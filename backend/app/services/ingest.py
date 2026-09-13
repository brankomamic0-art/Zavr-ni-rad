"""Poslovi dohvata podataka.

Tri posla čine sloj podataka:

1. `backfill_history()`  — jednokratno povlačenje više sezona po ligama.
                           Ovo stvara skup za treniranje.
2. `fetch_upcoming()`    — dnevno, utakmice za zadani datum + koeficijenti + ozljede.
3. `update_results()`    — dnevno, upisuje ishode odigranih utakmica.
                           Ovo stvara OZNAKE (labels) bez kojih nema učenja.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import session_scope
from app.models import Fixture, FixtureStat, IngestLog, Injury, League, Odd, Team
from app.services.api_football import (
    FINISHED_STATUSES,
    ApiFootballClient,
    classify_tier,
    is_european,
    is_supported,
    normalize_country,
)

log = logging.getLogger(__name__)

# ID-evi tržišta u API-Football odds odgovoru
BET_GOALS_OVER_UNDER = 5
BET_BOTH_TEAMS_SCORE = 8

ODDS_LABELS = {
    BET_GOALS_OVER_UNDER: {"Over 2.5": "over25", "Under 2.5": "under25"},
    BET_BOTH_TEAMS_SCORE: {"Yes": "btts", "No": "btts_no"},
}

# Nazivi iz `statistics` bloka -> stupci u `fixture_stats`.
STAT_MAP = {
    "Shots on Goal": "shots_on",
    "Total Shots": "shots_total",
    "Shots insidebox": "shots_box",
    "Corner Kicks": "corners",
    "Ball Possession": "possession",
    "Goalkeeper Saves": "saves",
}


def _stat_value(value: Any) -> float | None:
    """API vraca broj, None, ili postotak kao tekst ('53%')."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().rstrip("%"))
    except ValueError:
        return None


def _upsert_fixture_stats(db: Session, item: dict) -> int:
    """Sprema statistiku utakmice ako je odgovor sadrzi.

    Blok `statistics` stize samo uz upite s parametrom `ids` — a upravo takav
    upit vec radi `update_results`. Statistika se time dobiva BEZ ijednog
    dodatnog poziva; prije se jednostavno bacala.
    """
    blocks = item.get("statistics") or []
    if not blocks:
        return 0

    fixture_id = (item.get("fixture") or {}).get("id")
    if not fixture_id:
        return 0

    written = 0
    for block in blocks:
        team_id = (block.get("team") or {}).get("id")
        if not team_id:
            continue
        values = {
            STAT_MAP[entry["type"]]: _stat_value(entry.get("value"))
            for entry in block.get("statistics") or []
            if entry.get("type") in STAT_MAP
        }
        if not values:
            continue

        existing = db.scalar(
            select(FixtureStat).where(
                FixtureStat.fixture_id == fixture_id, FixtureStat.team_id == team_id
            )
        )
        target = existing or FixtureStat(fixture_id=fixture_id, team_id=team_id)
        for column, value in values.items():
            setattr(target, column, value)
        if existing is None:
            db.add(target)
        written += 1
    return written


# ═══════════════════════════════════════════════════════════════
# Pomoćne funkcije za upis
# ═══════════════════════════════════════════════════════════════

def _upsert_league(db: Session, raw: dict) -> int:
    league_id = raw["id"]
    country = normalize_country(raw.get("country"))
    obj = db.get(League, league_id)
    is_new = obj is None
    if is_new:
        obj = League(id=league_id)

    obj.name = raw.get("name") or ""
    obj.country = country
    obj.type = raw.get("type")
    obj.logo = raw.get("logo")
    obj.tier = classify_tier(league_id, country, obj.name)

    if is_new:
        db.add(obj)
        # Flush tek nakon postavljanja atributa. Bez njega objekt ostaje
        # "pending", sljedeci db.get() ga ne vidi i ista liga se umetne dvaput
        # (UNIQUE constraint failed). Redoslijed je bitan: flush prije
        # postavljanja atributa pao bi na NOT NULL.
        db.flush()
    return league_id


def _upsert_team(db: Session, raw: dict, country: str | None = None) -> int:
    team_id = raw["id"]
    obj = db.get(Team, team_id)
    is_new = obj is None
    if is_new:
        obj = Team(id=team_id)

    obj.name = raw.get("name") or ""
    obj.logo = raw.get("logo")
    if country:
        obj.country = country

    if is_new:
        db.add(obj)
        db.flush()
    return team_id


def _parse_kickoff(raw: str) -> datetime:
    # API vraća ISO 8601 s pomakom, npr. 2024-08-16T20:00:00+02:00
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _upsert_fixture(db: Session, item: dict) -> Fixture:
    fx = item["fixture"]
    lg = item["league"]
    teams = item["teams"]
    goals = item.get("goals") or {}
    score = item.get("score") or {}
    halftime = score.get("halftime") or {}
    fulltime = score.get("fulltime") or {}

    country = normalize_country(lg.get("country"))
    _upsert_league(db, lg)
    home_id = _upsert_team(db, teams["home"], country)
    away_id = _upsert_team(db, teams["away"], country)

    status = (fx.get("status") or {}).get("short") or "NS"
    finished = status in FINISHED_STATUSES

    obj = db.get(Fixture, fx["id"])
    is_new = obj is None
    if is_new:
        obj = Fixture(id=fx["id"])

    obj.league_id = lg["id"]
    obj.season = lg.get("season")
    obj.kickoff = _parse_kickoff(fx["date"])
    obj.home_team_id = home_id
    obj.away_team_id = away_id
    obj.status = status
    obj.finished = finished
    # Rezultat se upisuje samo ako je utakmica stvarno završena — inače ostaje
    # NULL i ta utakmica nikad ne uđe u skup za treniranje.
    if finished:
        obj.home_goals = goals.get("home")
        obj.away_goals = goals.get("away")
        obj.home_ht = halftime.get("home")
        obj.away_ht = halftime.get("away")
        # Rezultat nakon 90 minuta — bez produzetaka. Ovo je oznaka na kojoj
        # model uci, jer se trziste Over/Under 2.5 tako i namiruje. Bez ovog
        # retka bi svaka nova utakmica s produzecima opet bila krivo oznacena.
        obj.home_ft = fulltime.get("home")
        obj.away_ft = fulltime.get("away")

    if is_new:
        db.add(obj)
        db.flush()

    # Statistika stize samo uz upite s `ids` (dakle iz `update_results`).
    _upsert_fixture_stats(db, item)
    return obj


def _log_job(
    db: Session,
    job: str,
    status: str,
    detail: str,
    api_calls: int,
    rows: int,
    started: datetime,
) -> None:
    db.add(
        IngestLog(
            job=job,
            status=status,
            detail=detail[:1000],
            api_calls=api_calls,
            rows=rows,
            started_at=started,
            finished_at=datetime.now(timezone.utc),
        )
    )


# ═══════════════════════════════════════════════════════════════
# 1. Povijesni backfill — stvara skup za treniranje
# ═══════════════════════════════════════════════════════════════

def backfill_history(
    league_ids: list[int] | None = None,
    seasons: list[int] | None = None,
) -> dict[str, Any]:
    """Povlači sve utakmice zadanih liga i sezona i trajno ih sprema.

    Jedan poziv po (liga, sezona) vraća cijelu sezonu (cca 380 utakmica), što je
    daleko štedljivije prema kvoti API-ja od dohvata po danima.
    """
    league_ids = league_ids or settings.league_ids
    seasons = seasons or settings.seasons
    started = datetime.now(timezone.utc)
    total_rows = 0
    errors: list[str] = []
    api_calls = 0

    with ApiFootballClient() as client:
        for league_id in league_ids:
            for season in seasons:
                try:
                    items = client.fixtures_by_league_season(league_id, season)
                except Exception as exc:  # noqa: BLE001
                    msg = f"liga {league_id}/{season}: {exc}"
                    log.warning("Preskacem %s", msg)
                    errors.append(msg)
                    continue

                with session_scope() as db:
                    for item in items:
                        try:
                            _upsert_fixture(db, item)
                            total_rows += 1
                        except Exception as exc:  # noqa: BLE001
                            log.debug("Neispravan zapis preskocen: %s", exc)
                log.info(
                    "Liga %s sezona %s: %s utakmica (ukupno %s, poziva %s)",
                    league_id,
                    season,
                    len(items),
                    total_rows,
                    client.calls,
                )
        api_calls = client.calls

    with session_scope() as db:
        _log_job(
            db,
            "backfill_history",
            "error" if errors else "ok",
            f"{total_rows} utakmica; greske: {len(errors)}",
            api_calls,
            total_rows,
            started,
        )

    return {"rows": total_rows, "api_calls": api_calls, "errors": errors}


# ═══════════════════════════════════════════════════════════════
# 2. Dnevni dohvat nadolazećih utakmica
# ═══════════════════════════════════════════════════════════════

def fetch_upcoming(
    target_date: date | None = None,
    region: str = "supported",
    # Koeficijenti se po zadanom NE dohvacaju: sustav ih ne koristi ni kao
    # znacajku ni kao mjerilo, a dohvat kosta jedan poziv po utakmici.
    with_odds: bool = False,
    with_injuries: bool = True,
    with_history: bool = True,
    history_last: int = 20,
) -> dict[str, Any]:
    """Dohvaća utakmice zadanog dana.

    Args:
        region: "supported" (Europa + Južna Amerika + međunarodna natjecanja),
            "europe" ili "all".
        with_history: dohvaća i posljednjih N odigranih utakmica svake momčadi.
            Bez toga nema forme, pa ni predikcija — nove momčadi u bazi imaju
            prazan prozor.
    """
    target_date = target_date or datetime.now().date()
    date_str = target_date.isoformat()
    started = datetime.now(timezone.utc)
    fixture_ids: list[int] = []
    team_ids: set[int] = set()

    with ApiFootballClient() as client:
        try:
            items = client.fixtures_by_date(date_str)
        except Exception as exc:  # noqa: BLE001
            with session_scope() as db:
                _log_job(db, "fetch_upcoming", "error", str(exc), client.calls, 0, started)
            raise

        total_raw = len(items)
        if region == "europe":
            keep = is_european
        elif region == "all":
            keep = lambda _country: True  # noqa: E731
        else:
            keep = is_supported
        items = [
            it for it in items
            if keep(normalize_country((it.get("league") or {}).get("country")))
        ]

        with session_scope() as db:
            for item in items:
                try:
                    fx = _upsert_fixture(db, item)
                    fixture_ids.append(fx.id)
                    team_ids.add(fx.home_team_id)
                    team_ids.add(fx.away_team_id)
                except Exception as exc:  # noqa: BLE001
                    log.debug("Neispravan zapis preskocen: %s", exc)

        log.info(
            "Datum %s: %s od %s utakmica u odabranim regijama (%s momcadi)",
            date_str, len(fixture_ids), total_raw, len(team_ids),
        )

        if with_history and team_ids:
            fetch_team_history(client, sorted(team_ids), last=history_last)
        if with_injuries and fixture_ids:
            _fetch_injuries(client, fixture_ids)
        if with_odds and fixture_ids:
            _fetch_odds(client, fixture_ids)

        api_calls = client.calls

    with session_scope() as db:
        _log_job(
            db,
            "fetch_upcoming",
            "ok",
            f"{date_str}: {len(fixture_ids)} utakmica",
            api_calls,
            len(fixture_ids),
            started,
        )
    return {"date": date_str, "fixtures": len(fixture_ids), "api_calls": api_calls}


def fetch_team_history(
    client: ApiFootballClient,
    team_ids: list[int],
    last: int = 20,
    min_existing: int | None = None,
) -> int:
    """Dohvaća posljednjih `last` odigranih utakmica po momčadi.

    Ovo je izvor forme za nadolazeće utakmice. Momčadi koje u bazi već imaju
    dovoljno odigranih utakmica preskaču se, pa svako sljedeće pokretanje troši
    znatno manje poziva.
    """
    if min_existing is None:
        min_existing = max(settings.form_window_overall, settings.form_window_venue) + 4

    # Koliko odigranih utakmica svaka momčad već ima u bazi.
    with session_scope() as db:
        rows = db.execute(
            select(Fixture.home_team_id, func.count(Fixture.id))
            .where(Fixture.finished.is_(True), Fixture.home_team_id.in_(team_ids))
            .group_by(Fixture.home_team_id)
        ).all()
        home_counts = dict(rows)
        rows = db.execute(
            select(Fixture.away_team_id, func.count(Fixture.id))
            .where(Fixture.finished.is_(True), Fixture.away_team_id.in_(team_ids))
            .group_by(Fixture.away_team_id)
        ).all()
        away_counts = dict(rows)

    pending = [
        tid for tid in team_ids
        if home_counts.get(tid, 0) + away_counts.get(tid, 0) < min_existing
    ]
    skipped = len(team_ids) - len(pending)
    log.info(
        "Povijest momcadi: %s za dohvat, %s vec ima dovoljno", len(pending), skipped
    )

    rows_written = 0
    for index, team_id in enumerate(pending, start=1):
        try:
            items = client.fixtures_by_team(team_id, last=last)
        except Exception as exc:  # noqa: BLE001
            log.debug("Povijest momcadi %s preskocena: %s", team_id, exc)
            continue
        with session_scope() as db:
            for item in items:
                try:
                    _upsert_fixture(db, item)
                    rows_written += 1
                except Exception as exc:  # noqa: BLE001
                    log.debug("Neispravan zapis preskocen: %s", exc)
        if index % 50 == 0:
            log.info(
                "  povijest %s/%s momcadi (poziva %s)", index, len(pending), client.calls
            )

    log.info("Povijest: upisano %s zapisa", rows_written)
    return rows_written


def _fetch_injuries(client: ApiFootballClient, fixture_ids: list[int]) -> int:
    count = 0
    for i in range(0, len(fixture_ids), 20):
        batch = fixture_ids[i : i + 20]
        try:
            rows = client.injuries_for_fixtures(batch)
        except Exception as exc:  # noqa: BLE001
            log.debug("Ozljede preskocene za skupinu: %s", exc)
            continue
        with session_scope() as db:
            # API zna vratiti istog igraca vise puta unutar iste serije, a
            # provjera u bazi ne vidi retke koji su dodani ali jos nisu upisani.
            seen: set[tuple[int, int, str]] = set()
            for row in rows:
                fixture_id = (row.get("fixture") or {}).get("id")
                team_id = (row.get("team") or {}).get("id")
                player = row.get("player") or {}
                name = player.get("name")
                if not (fixture_id and team_id and name):
                    continue

                key = (fixture_id, team_id, name)
                if key in seen:
                    continue
                seen.add(key)

                exists = db.scalar(
                    select(Injury).where(
                        Injury.fixture_id == fixture_id,
                        Injury.team_id == team_id,
                        Injury.player_name == name,
                    )
                )
                if exists:
                    continue
                db.add(
                    Injury(
                        fixture_id=fixture_id,
                        team_id=team_id,
                        player_name=name,
                        reason=player.get("reason"),
                        type=player.get("type"),
                    )
                )
                count += 1
    log.info("Ozljede: %s zapisa", count)
    return count


def _fetch_odds(client: ApiFootballClient, fixture_ids: list[int]) -> int:
    """Sprema najbolji (najveci) koeficijent po trzistu preko svih kladionica."""
    count = 0
    for fixture_id in fixture_ids:
        try:
            payload = client.odds_for_fixture(fixture_id)
        except Exception:  # noqa: BLE001
            continue
        if not payload:
            continue

        best: dict[str, tuple[float, str]] = {}
        for bookmaker in payload[0].get("bookmakers") or []:
            bk_name = bookmaker.get("name") or "?"
            for bet in bookmaker.get("bets") or []:
                label_map = ODDS_LABELS.get(bet.get("id"))
                if not label_map:
                    continue
                for value in bet.get("values") or []:
                    try:
                        price = float(value.get("odd"))
                    except (TypeError, ValueError):
                        continue
                    market = label_map.get(value.get("value"))
                    if market and (market not in best or price > best[market][0]):
                        best[market] = (price, bk_name)

        if not best:
            continue
        with session_scope() as db:
            for market, (price, bk_name) in best.items():
                existing = db.scalar(
                    select(Odd).where(Odd.fixture_id == fixture_id, Odd.market == market)
                )
                if existing:
                    existing.price = price
                    existing.bookmaker = bk_name
                else:
                    db.add(
                        Odd(
                            fixture_id=fixture_id,
                            market=market,
                            price=price,
                            bookmaker=bk_name,
                        )
                    )
                count += 1
    log.info("Koeficijenti: %s zapisa", count)
    return count


# ═══════════════════════════════════════════════════════════════
# 2b. Koeficijenti po datumu — jedini nacin da ih se skupi u kolicini
# ═══════════════════════════════════════════════════════════════

def fetch_odds_by_date(days_back: int = 7, days_ahead: int = 2) -> dict[str, Any]:
    """Dohvaca koeficijente za raspon datuma, stranično (10 utakmica po pozivu).

    Dvije mjerene činjenice iz API-ja koje određuju ovaj pristup:

    1. **Povijesni koeficijenti ne postoje.** Upit za prošlu sezonu ili datum
       stariji od tjedan dana vraća 0 rezultata — API čuva samo 7 dana
       povijesti. Skup koeficijenata se zato ne može "backfillati", nego se
       jedino gradi unaprijed, dan po dan.
    2. **Dohvat po datumu je 10x jeftiniji** od dohvata po utakmici: jedan poziv
       vraća stranicu od 10 utakmica umjesto jedne.

    Prozor unatrag je bitan jer su te utakmice već odigrane — dobiju se
    koeficijent i ishod zajedno, što je upravo ono što backtest traži.
    """
    started = datetime.now(timezone.utc)
    today = datetime.now().date()
    dates = [
        today + timedelta(days=offset)
        for offset in range(-days_back, days_ahead + 1)
    ]

    known: set[int] = set()
    with session_scope() as db:
        known = set(db.scalars(select(Fixture.id)))

    written = 0
    skipped_unknown = 0
    with ApiFootballClient() as client:
        for day in dates:
            page = 1
            while True:
                try:
                    # Bez filtra `bet=` — jedan poziv tako nosi i Over/Under 2.5
                    # (id 5) i "obje zabijaju" (id 8). Filtriranje po jednom
                    # trzistu udvostrucilo bi broj poziva.
                    payload = client.get("/odds", date=day.isoformat(), page=page)
                except Exception as exc:  # noqa: BLE001
                    log.warning("Koeficijenti %s str.%s preskoceni: %s", day, page, exc)
                    break
                if not payload:
                    break

                rows, unknown = _store_odds_payload(payload, known)
                written += rows
                skipped_unknown += unknown

                # API vraca 10 po stranici; kraca stranica znaci kraj.
                if len(payload) < 10:
                    break
                page += 1
                if page > 200:  # zastita od beskonacne petlje
                    break
            log.info("  %s: ukupno upisano %s cijena", day, written)
        api_calls = client.calls

    with session_scope() as db:
        _log_job(
            db,
            "fetch_odds_by_date",
            "ok",
            f"{dates[0]}..{dates[-1]}: {written} cijena, {skipped_unknown} utakmica izvan baze",
            api_calls,
            written,
            started,
        )
    return {"prices": written, "api_calls": api_calls, "unknown_fixtures": skipped_unknown}


def _store_odds_payload(payload: list[dict], known: set[int]) -> tuple[int, int]:
    """Iz odgovora `/odds` izvlaci najbolju cijenu po trzistu i sprema je."""
    written = 0
    unknown = 0
    best_by_fixture: dict[int, dict[str, tuple[float, str]]] = {}

    for entry in payload:
        fixture_id = (entry.get("fixture") or {}).get("id")
        if not fixture_id:
            continue
        if fixture_id not in known:
            unknown += 1
            continue

        best: dict[str, tuple[float, str]] = {}
        for bookmaker in entry.get("bookmakers") or []:
            bk_name = bookmaker.get("name") or "?"
            for bet in bookmaker.get("bets") or []:
                label_map = ODDS_LABELS.get(bet.get("id"))
                if not label_map:
                    continue
                for value in bet.get("values") or []:
                    try:
                        price = float(value.get("odd"))
                    except (TypeError, ValueError):
                        continue
                    market = label_map.get(value.get("value"))
                    if market and (market not in best or price > best[market][0]):
                        best[market] = (price, bk_name)
        if best:
            best_by_fixture[fixture_id] = best

    if not best_by_fixture:
        return 0, unknown

    with session_scope() as db:
        for fixture_id, markets in best_by_fixture.items():
            for market, (price, bk_name) in markets.items():
                existing = db.scalar(
                    select(Odd).where(Odd.fixture_id == fixture_id, Odd.market == market)
                )
                if existing:
                    existing.price = price
                    existing.bookmaker = bk_name
                else:
                    db.add(Odd(fixture_id=fixture_id, market=market, price=price, bookmaker=bk_name))
                written += 1
    return written, unknown


# ═══════════════════════════════════════════════════════════════
# 3. Upis ishoda — bez ovoga nema oznaka za učenje
# ═══════════════════════════════════════════════════════════════

def update_results(lookback_days: int = 3) -> dict[str, Any]:
    """Za utakmice iz zadnjih N dana koje jos nemaju rezultat dohvaca ishod."""
    started = datetime.now(timezone.utc)
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=lookback_days)

    with session_scope() as db:
        pending = list(
            db.scalars(
                select(Fixture.id).where(
                    Fixture.finished.is_(False),
                    Fixture.kickoff >= since,
                    Fixture.kickoff <= now,
                )
            )
        )

    if not pending:
        with session_scope() as db:
            _log_job(db, "update_results", "ok", "nema utakmica za azuriranje", 0, 0, started)
        return {"updated": 0, "pending": 0, "api_calls": 0}

    updated = 0
    with_stats = 0
    with ApiFootballClient() as client:
        for i in range(0, len(pending), 20):
            batch = pending[i : i + 20]
            try:
                items = client.fixtures_by_ids(batch)
            except Exception as exc:  # noqa: BLE001
                log.warning("Skupina preskocena: %s", exc)
                continue
            with session_scope() as db:
                for item in items:
                    fx = _upsert_fixture(db, item)
                    if fx.finished:
                        updated += 1
                    if item.get("statistics"):
                        with_stats += 1
        api_calls = client.calls

    log.info(
        "Upisano %s ishoda (od %s na cekanju), statistika za %s utakmica",
        updated, len(pending), with_stats,
    )
    with session_scope() as db:
        _log_job(
            db,
            "update_results",
            "ok",
            f"{updated}/{len(pending)} ishoda, {with_stats} sa statistikom",
            api_calls,
            updated,
            started,
        )
    return {"updated": updated, "pending": len(pending), "api_calls": api_calls}
