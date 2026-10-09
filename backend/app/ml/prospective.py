"""Prospektivna evaluacija — predikcije izdane PRIJE utakmice naspram ishoda.

Retrospektivni test (`train.py`) mjeri model na povijesnim utakmicama nakon
kronološke granice. To je pošteno samo ako u značajke nije procurila
budućnost, a i tada je granica izabrana unaprijed znajući cijeli skup.
Ovdje se evaluiraju predikcije koje je sustav spremio prije početka utakmice
(`snapshot_upcoming`), dakle u trenutku kad ishod nije postojao. Metrike su
iste kao u retrospektivnom testu, pa su dva broja izravno usporediva.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.ml.evaluate import backtest, calibration_table, classification_metrics
from app.models import Fixture, Odd, Prediction

MARKETS = ("over25", "btts")

# Ispod ovoliko utakmica metrike su previše šumovite da bi išta značile.
MIN_SAMPLE = 30


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def load_resolved(db: Session, days: int | None = None) -> pd.DataFrame:
    """Spremljene predikcije čije su utakmice odigrane, s ishodom i koeficijentom."""
    stmt = (
        select(
            Prediction.fixture_id,
            Prediction.market,
            Prediction.model_name,
            Prediction.probability,
            Prediction.created_at,
            Fixture.kickoff,
            Fixture.home_goals,
            Fixture.away_goals,
            Fixture.home_ft,
            Fixture.away_ft,
        )
        .join(Fixture, Fixture.id == Prediction.fixture_id)
        .where(Fixture.finished.is_(True))
    )
    if days:
        stmt = stmt.where(Fixture.kickoff >= datetime.now(timezone.utc) - timedelta(days=days))

    df = pd.DataFrame(db.execute(stmt).all())
    if df.empty:
        return df

    # Zaštita od curenja: računa se samo ono što je spremljeno prije početka.
    created = df["created_at"].map(_as_utc)
    kickoff = df["kickoff"].map(_as_utc)
    df = df[created < kickoff].copy()

    # Ista oznaka kao kod treniranja: rezultat nakon 90 minuta, a ako ga nema, konačni.
    home = df["home_ft"].fillna(df["home_goals"])
    away = df["away_ft"].fillna(df["away_goals"])
    df["y"] = np.where(
        df["market"] == "over25",
        (home + away > 2.5).astype(float),
        ((home > 0) & (away > 0)).astype(float),
    )
    df = df[home.notna() & away.notna()]

    odds = pd.DataFrame(
        db.execute(
            select(Odd.fixture_id, Odd.market, Odd.price).where(
                Odd.fixture_id.in_(df["fixture_id"].unique().tolist()),
                Odd.market.in_(MARKETS),
            )
        ).all(),
        columns=["fixture_id", "market", "price"],
    )
    return df.merge(odds, on=["fixture_id", "market"], how="left")


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _evaluate(df: pd.DataFrame) -> dict[str, Any]:
    y = df["y"].to_numpy()
    p = df["probability"].to_numpy()
    out: dict[str, Any] = {
        "metrics": {k: _clean(v) for k, v in classification_metrics(y, p).items()},
        "backtest": {k: _clean(v) for k, v in backtest(y, p, df["price"].to_numpy()).items()},
        "calibration": [
            {
                "bin": str(r["bin"]),
                "n": int(r["n"]),
                "predicted": round(float(r["predvideno"]), 4),
                "observed": round(float(r["ostvareno"]), 4),
            }
            for _, r in calibration_table(y, p).iterrows()
        ],
        "with_odds": int(df["price"].notna().sum()),
    }
    if len(df) < MIN_SAMPLE:
        out["warning"] = f"Uzorak je premalen ({len(df)} < {MIN_SAMPLE}) — metrike nisu pouzdane."
    return out


def evaluate_prospective(db: Session, days: int | None = None) -> dict[str, Any]:
    """Metrike po tržištu: ukupno (ono što je sustav poslužio) i po izvoru predikcije."""
    df = load_resolved(db, days)
    pending = db.scalar(
        select(func.count(Prediction.id))
        .join(Fixture, Fixture.id == Prediction.fixture_id)
        .where(Fixture.finished.is_(False))
    )
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days": days,
        "resolved": int(len(df)),
        "pending": int(pending or 0),  # spremljene, utakmica još nije odigrana
        "markets": {},
    }
    if df.empty:
        return report

    report["first_kickoff"] = _as_utc(df["kickoff"].min()).isoformat()
    report["last_kickoff"] = _as_utc(df["kickoff"].max()).isoformat()
    for market in MARKETS:
        sub = df[df["market"] == market]
        if sub.empty:
            continue
        entry = _evaluate(sub)
        entry["by_source"] = {
            source: _evaluate(group)["metrics"] for source, group in sub.groupby("model_name")
        }
        report["markets"][market] = entry
    return report
