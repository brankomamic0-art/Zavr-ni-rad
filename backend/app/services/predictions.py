"""Posluživanje predikcija.

Model se učitava jednom i drži u memoriji. Ako model još nije istreniran,
sustav se vraća na jednostavnu heuristiku (prosjek stopa iz forme) i to
izričito označava u odgovoru — tako je uvijek jasno gleda li korisnik izlaz
modela ili tek statistički prosjek.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.ml.dataset import build_serving_matrix
from app.ml.features import ODDS_FEATURES, SHOT_FEATURES
from app.models import Prediction

log = logging.getLogger(__name__)

MARKETS = ("over25", "btts")
_CACHE: dict[str, dict[str, Any]] = {}

# Sufiks varijante modela koja koristi i znacajke iz udaraca.
SHOT_SUFFIX = "_shots"


def _artifact_path(market: str, model_name: str) -> Path:
    return Path(settings.model_dir) / f"{market}__{model_name}.joblib"


def _has_shot_history(row: pd.Series) -> bool:
    """Postoji li povijest udaraca za OBJE momcadi te utakmice."""
    return bool(pd.notna(row.get("h_shots_on_f")) and pd.notna(row.get("a_shots_on_f")))


def load_artifact(market: str, model_name: str | None = None) -> dict[str, Any] | None:
    model_name = model_name or settings.active_model
    key = f"{market}__{model_name}"
    if key in _CACHE:
        return _CACHE[key]

    path = _artifact_path(market, model_name)
    if not path.exists():
        return None
    try:
        import joblib  # lokalni uvoz — nije potreban ako model ne postoji

        artifact = joblib.load(path)
        _CACHE[key] = artifact
        log.info("Ucitan model %s (treniran %s)", key, artifact.get("trained_at"))
        return artifact
    except Exception as exc:  # noqa: BLE001
        log.error("Model %s se ne moze ucitati: %s", key, exc)
        return None


def clear_cache() -> None:
    _CACHE.clear()


def model_info() -> dict[str, Any]:
    info: dict[str, Any] = {"active_model": settings.active_model, "markets": {}}
    for market in MARKETS:
        artifact = load_artifact(market)
        shot_art = load_artifact(market, f"{settings.active_model}{SHOT_SUFFIX}")
        if artifact is None:
            info["markets"][market] = {"available": False, "mode": "heuristika"}
        else:
            metrics = artifact.get("metrics") or {}
            info["markets"][market] = {
                "available": True,
                "mode": "model",
                "trained_at": artifact.get("trained_at"),
                "cutoff": artifact.get("cutoff"),
                "log_loss": metrics.get("log_loss"),
                "roc_auc": metrics.get("roc_auc"),
                "brier": metrics.get("brier"),
                "skill": metrics.get("log_loss_skill"),
                # Varijanta s udarcima postoji samo ako je istrenirana; sucelje
                # tako moze pokazati koji je model dao koju predikciju.
                "shots_variant": (
                    None if shot_art is None
                    else {
                        "trained_at": shot_art.get("trained_at"),
                        "roc_auc": (shot_art.get("metrics") or {}).get("roc_auc"),
                        "skill": (shot_art.get("metrics") or {}).get("log_loss_skill"),
                    }
                ),
            }
    return info


# ═══════════════════════════════════════════════════════════════
# Heuristika (zamjena dok model nije istreniran)
# ═══════════════════════════════════════════════════════════════

def _heuristic(row: pd.Series, market: str) -> float:
    """Prosjek četiriju stopa iz forme — isti pristup koji je sustav imao prije
    uvođenja strojnog učenja. Služi kao vidljiva referentna vrijednost."""
    parts = []
    for prefix in ("h_ov", "a_ov", "h_vn", "a_vn"):
        count = row.get(f"{prefix}_{market}")
        n = row.get(f"{prefix}_n")
        if pd.notna(count) and pd.notna(n) and n > 0:
            parts.append(float(count) / float(n))
    if not parts:
        return float("nan")
    return float(np.clip(np.mean(parts), 0.01, 0.99))


# ═══════════════════════════════════════════════════════════════
# Glavni ulaz
# ═══════════════════════════════════════════════════════════════

def _insufficient_history(matrix: pd.DataFrame) -> dict[int, bool]:
    """Utakmice u kojima neka momčad ima premalo odigranih utakmica."""
    threshold = settings.serving_min_history
    home_n = matrix.get("h_ov_n")
    away_n = matrix.get("a_ov_n")
    if home_n is None or away_n is None:
        return {}
    flagged = (home_n.fillna(0) < threshold) | (away_n.fillna(0) < threshold)
    return {int(k): bool(v) for k, v in flagged.items()}


def predict_fixtures(
    db: Session,
    fixture_ids: list[int],
    persist: bool = False,
) -> dict[int, dict[str, Any]]:
    """Vraća predikcije po utakmici, uz značajke na kojima se temelje."""
    if not fixture_ids:
        return {}

    matrix = build_serving_matrix(db, fixture_ids, with_odds=True)
    if matrix.empty:
        return {}

    results: dict[int, dict[str, Any]] = {}
    matrix = matrix.set_index("fixture_id", drop=False)

    # Utakmice se dijele u dvije skupine prema tome postoji li povijest udaraca
    # za obje momčadi. API statistiku nudi samo za dio liga, a izmjereno je da
    # model s udarcima pobjeđuje ondje gdje ih ima (skill 1,82 % -> 2,33 % za
    # Over 2.5), ali gubi ako se za ostale utakmice vrijednosti imputiraju.
    # Zato se ne bira jedan model za sve, nego model po utakmici.
    has_shots = matrix.apply(_has_shot_history, axis=1)

    for market in MARKETS:
        base_art = load_artifact(market)
        shot_art = load_artifact(market, f"{settings.active_model}{SHOT_SUFFIX}")

        probs = np.full(len(matrix), np.nan)
        sources = np.array(["form_average"] * len(matrix), dtype=object)
        modes = np.array(["heuristika"] * len(matrix), dtype=object)

        groups = [(shot_art, has_shots.to_numpy()), (base_art, ~has_shots.to_numpy())]
        if shot_art is None:  # varijanta s udarcima nije istrenirana
            groups = [(base_art, np.ones(len(matrix), dtype=bool))]

        for artifact, mask in groups:
            if artifact is None or not mask.any():
                continue
            feature_cols = artifact["features"]
            for col in [c for c in feature_cols if c not in matrix.columns]:
                matrix[col] = np.nan
            try:
                probs[mask] = artifact["model"].predict_proba(matrix.loc[mask, feature_cols])[:, 1]
                sources[mask] = artifact["model_name"]
                modes[mask] = "model"
            except Exception as exc:  # noqa: BLE001
                log.error("Predikcija nije uspjela (%s / %s): %s", market, artifact["model_name"], exc)

        # Sve sto nijedan model nije pokrio pada na heuristiku iz forme.
        gap = ~np.isfinite(probs)
        if gap.any():
            fallback = np.array([_heuristic(r, market) for _, r in matrix.loc[gap].iterrows()])
            probs[gap] = fallback

        insufficient = _insufficient_history(matrix)

        for idx, (fixture_id, prob) in enumerate(zip(matrix.index, probs, strict=False)):
            entry = results.setdefault(int(fixture_id), {"markets": {}})
            mode, source = modes[idx], sources[idx]
            value = None if not np.isfinite(prob) else round(float(prob), 4)
            # Predikcija se ne izdaje ako neka od momčadi nema dovoljno
            # odigranih utakmica — inače bi 2/2 izgledalo kao pouzdanih 100 %.
            if bool(insufficient.get(fixture_id, False)):
                value = None
            entry["markets"][market] = {"p": value, "mode": mode, "source": source}
            # Komplementarna tržišta se izvode, ne modeliraju zasebno:
            # under 2.5 je po definiciji suprotan događaj od over 2.5.
            complement = {"over25": "under25", "btts": "btts_no"}[market]
            entry["markets"][complement] = {
                "p": None if value is None else round(1.0 - value, 4),
                "mode": mode,
                "source": source,
            }

    if persist:
        _persist(db, matrix, results)

    return results


def _persist(db: Session, matrix: pd.DataFrame, results: dict[int, dict]) -> None:
    """Sprema izdanu predikciju radi kasnije poštene (prospektivne) evaluacije."""
    for fixture_id, payload in results.items():
        for market in MARKETS:
            entry = payload["markets"].get(market) or {}
            prob = entry.get("p")
            if prob is None:
                continue
            existing = db.scalar(
                select(Prediction).where(
                    Prediction.fixture_id == fixture_id,
                    Prediction.market == market,
                    Prediction.model_name == entry["source"],
                )
            )
            if existing:
                existing.probability = prob
                continue
            row = matrix.loc[fixture_id]
            db.add(
                Prediction(
                    fixture_id=fixture_id,
                    market=market,
                    model_name=entry["source"],
                    probability=prob,
                    features={
                        k: (None if pd.isna(v) else float(v))
                        for k, v in row.items()
                        if isinstance(v, (int, float, np.floating)) and k not in ODDS_FEATURES
                    },
                    created_at=datetime.now(timezone.utc),
                )
            )
    db.commit()


def form_block(matrix_row: pd.Series) -> dict[str, Any]:
    """Pretvara redak matrice značajki u strukturu koju prikazuje sučelje."""

    def block(prefix: str) -> dict[str, Any]:
        n = matrix_row.get(f"{prefix}_n")
        n = 0 if pd.isna(n) else int(n)

        def val(stat: str, digits: int = 2) -> float | int | None:
            raw = matrix_row.get(f"{prefix}_{stat}")
            if pd.isna(raw):
                return None
            return int(raw) if digits == 0 else round(float(raw), digits)

        return {
            "n": n,
            "over25": val("over25", 0),
            "btts": val("btts", 0),
            "cs": val("cs", 0),
            "fts": val("fts", 0),
            "gf": val("gf"),
            "ga": val("ga"),
            "tot": val("tot"),
            "ppg": val("pts"),
        }

    return {
        "hO": block("h_ov"),
        "aO": block("a_ov"),
        "hV": block("h_vn"),
        "aV": block("a_vn"),
    }
