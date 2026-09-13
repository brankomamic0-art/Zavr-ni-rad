"""HTTP rute.

Sustav je javan i bez prijave — stranica pri otvaranju odmah traži
`GET /api/matches` i prikazuje rezultat.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.ml.dataset import build_serving_matrix
from app.models import Fixture, IngestLog, Injury, League, Odd, Team
from app.schemas import MatchesResponse, StatusResponse
from app.services import predictions as pred_service
from app.services.scheduler import scheduler_status
from app.services.quality import classify_competition

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["matches"])

MIN_ROWS_FOR_TRAINING = 2000


def _jsonable(value):
    """NumPy skalari nisu serijalizabilni u JSON — svode se na Python tipove."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    item = getattr(value, "item", None)
    if callable(item):
        value = value.item()
    return None if pd.isna(value) else value


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@router.get("/matches", response_model=MatchesResponse)
def get_matches(
    response: Response,
    db: Session = Depends(get_db),
    day: str | None = Query(default=None, alias="date", description="YYYY-MM-DD, zadano: danas"),
    min_probability: float | None = Query(default=None, ge=0.0, le=1.0),
    market: str | None = Query(default=None, description="over25 | under25 | btts | btts_no"),
    country: str | None = Query(default=None),
    league_id: int | None = Query(default=None),
    only_senior: bool = Query(
        default=True,
        description="Izostavi juniorska, ženska i pričuvna natjecanja",
    ),
    only_predicted: bool = Query(
        default=True,
        description="Izostavi utakmice bez predikcije (premalo odigranih utakmica)",
    ),
    only_with_odds: bool = Query(
        default=False,
        description="Zadrži samo utakmice koje kladionice kotiraju",
    ),
) -> MatchesResponse:
    """Utakmice zadanog dana s predikcijama, formom, koeficijentima i ozljedama."""
    try:
        target_day = date.fromisoformat(day) if day else datetime.now().date()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Neispravan datum, ocekujem YYYY-MM-DD") from exc

    start = datetime.combine(target_day, datetime.min.time(), tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    stmt = (
        select(Fixture)
        .join(League, League.id == Fixture.league_id)
        .where(Fixture.kickoff >= start, Fixture.kickoff < end)
    )
    if country:
        stmt = stmt.where(League.country == country)
    if league_id:
        stmt = stmt.where(Fixture.league_id == league_id)

    fixtures = list(db.scalars(stmt.order_by(Fixture.kickoff)))
    if not fixtures:
        return MatchesResponse(
            date=target_day.isoformat(),
            generated_at=datetime.now(timezone.utc),
            count=0,
            model=pred_service.model_info(),
            matches=[],
        )

    fixture_ids = [f.id for f in fixtures]
    preds = pred_service.predict_fixtures(db, fixture_ids)
    matrix = build_serving_matrix(db, fixture_ids, with_odds=True)
    matrix = matrix.set_index("fixture_id") if not matrix.empty else matrix

    odds_rows = db.execute(select(Odd).where(Odd.fixture_id.in_(fixture_ids))).scalars().all()
    odds_by_fixture: dict[int, dict] = {}
    for row in odds_rows:
        odds_by_fixture.setdefault(row.fixture_id, {})[row.market] = {
            "price": row.price,
            "bookmaker": row.bookmaker,
        }

    injury_rows = db.execute(select(Injury).where(Injury.fixture_id.in_(fixture_ids))).scalars().all()
    injuries_by_fixture: dict[int, dict[int, list]] = {}
    for row in injury_rows:
        injuries_by_fixture.setdefault(row.fixture_id, {}).setdefault(row.team_id, []).append(
            {"name": row.player_name, "reason": row.reason, "type": row.type}
        )

    matches = []
    for fx in fixtures:
        entry = preds.get(fx.id, {}).get("markets", {})

        category = classify_competition(
            fx.league.name, fx.home_team.name, fx.away_team.name
        )
        if only_senior and category != "senior":
            continue

        fixture_odds = odds_by_fixture.get(fx.id, {})
        if only_with_odds and not fixture_odds:
            continue

        # Predikcija izostaje kad neka momčad nema dovoljno odigranih utakmica.
        # Takve utakmice se po zadanom ne prikazuju jer bi im sučelje pokazalo
        # praznu vrijednost, a ne zato što su nevažne.
        if only_predicted and all(v.get("p") is None for v in entry.values()):
            continue

        if min_probability is not None and market:
            selected = entry.get(market, {}).get("p")
            if selected is None or selected < min_probability:
                continue

        analysis = None
        if not matrix.empty and fx.id in matrix.index:
            analysis = pred_service.form_block(matrix.loc[fx.id])

        value = {}
        for market_key, payload in entry.items():
            prob = payload.get("p")
            price = (fixture_odds.get(market_key) or {}).get("price")
            if prob is not None and price:
                value[market_key] = round(prob * price - 1.0, 4)

        fixture_injuries = injuries_by_fixture.get(fx.id, {})
        matches.append(
            {
                "id": fx.id,
                "kickoff": fx.kickoff,
                "status": fx.status,
                "category": category,
                "league": {
                    "id": fx.league.id,
                    "name": fx.league.name,
                    "country": fx.league.country,
                    "type": fx.league.type,
                    "tier": fx.league.tier,
                },
                "home": {"id": fx.home_team.id, "name": fx.home_team.name, "logo": fx.home_team.logo},
                "away": {"id": fx.away_team.id, "name": fx.away_team.name, "logo": fx.away_team.logo},
                "analysis": analysis,
                "predictions": entry,
                "odds": fixture_odds,
                "injuries": {
                    "home": fixture_injuries.get(fx.home_team_id, []),
                    "away": fixture_injuries.get(fx.away_team_id, []),
                },
                "value": value,
                "home_goals": fx.home_goals,
                "away_goals": fx.away_goals,
            }
        )

    # Podaci se osvježavaju jednom dnevno, pa kratko keširanje na rubu štedi rad.
    response.headers["Cache-Control"] = "public, max-age=120"

    return MatchesResponse(
        date=target_day.isoformat(),
        generated_at=datetime.now(timezone.utc),
        count=len(matches),
        model=pred_service.model_info(),
        matches=matches,
    )


@router.get("/status", response_model=StatusResponse)
def get_status(db: Session = Depends(get_db)) -> StatusResponse:
    today = datetime.now().date()
    start = datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc)

    total = db.scalar(select(func.count(Fixture.id))) or 0
    finished = db.scalar(select(func.count(Fixture.id)).where(Fixture.finished.is_(True))) or 0
    today_count = (
        db.scalar(
            select(func.count(Fixture.id)).where(
                Fixture.kickoff >= start, Fixture.kickoff < start + timedelta(days=1)
            )
        )
        or 0
    )
    earliest = db.scalar(select(func.min(Fixture.kickoff)))
    latest = db.scalar(select(func.max(Fixture.kickoff)))

    jobs = db.execute(select(IngestLog).order_by(IngestLog.id.desc()).limit(5)).scalars().all()

    return StatusResponse(
        database=settings.sqlalchemy_url.split("://")[0],
        fixtures_total=total,
        fixtures_finished=finished,
        fixtures_today=today_count,
        teams=db.scalar(select(func.count(Team.id))) or 0,
        leagues=db.scalar(select(func.count(League.id))) or 0,
        earliest=earliest,
        latest=latest,
        last_jobs=[
            {
                "job": j.job,
                "status": j.status,
                "detail": j.detail,
                "rows": j.rows,
                "api_calls": j.api_calls,
                "finished_at": j.finished_at.isoformat() if j.finished_at else None,
            }
            for j in jobs
        ],
        model=pred_service.model_info(),
        scheduler=scheduler_status(),
        training_ready=finished >= MIN_ROWS_FOR_TRAINING,
    )


@router.get("/leagues")
def get_leagues(db: Session = Depends(get_db)) -> list[dict]:
    rows = db.execute(select(League).order_by(League.tier, League.country, League.name)).scalars()
    return [
        {"id": r.id, "name": r.name, "country": r.country, "type": r.type, "tier": r.tier}
        for r in rows
    ]


@router.get("/fixtures/{fixture_id}")
def get_fixture(fixture_id: int, db: Session = Depends(get_db)) -> dict:
    fx = db.get(Fixture, fixture_id)
    if fx is None:
        raise HTTPException(status_code=404, detail="Utakmica nije pronadena")

    matrix = build_serving_matrix(db, [fixture_id], with_odds=True)
    features = {}
    if not matrix.empty:
        row = matrix.set_index("fixture_id").loc[fixture_id]
        features = {k: _jsonable(v) for k, v in row.items()}

    return {
        "id": fx.id,
        "kickoff": fx.kickoff,
        "home": fx.home_team.name,
        "away": fx.away_team.name,
        "league": fx.league.name,
        "status": fx.status,
        "home_goals": fx.home_goals,
        "away_goals": fx.away_goals,
        "predictions": pred_service.predict_fixtures(db, [fixture_id]).get(fixture_id, {}),
        "features": features,
    }
