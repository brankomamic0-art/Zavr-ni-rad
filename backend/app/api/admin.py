"""Administrativne rute za ručno pokretanje poslova.

Onemogućene su dok se ne postavi `ADMIN_TOKEN` u okolini, jer troše kvotu
API-ja i mijenjaju sadržaj baze.
"""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query

from app.config import settings
from app.services import predictions as pred_service
from app.services.ingest import backfill_history, fetch_upcoming, update_results

router = APIRouter(prefix="/api/admin", tags=["admin"])


def _require_token(x_admin_token: str | None) -> None:
    if not settings.admin_token:
        raise HTTPException(
            status_code=404, detail="Admin rute su onemogucene (ADMIN_TOKEN nije postavljen)"
        )
    if x_admin_token != settings.admin_token:
        raise HTTPException(status_code=401, detail="Neispravan admin token")


@router.post("/fetch-upcoming")
def trigger_fetch(
    background: BackgroundTasks,
    x_admin_token: str | None = Header(default=None),
    day: str | None = Query(default=None, alias="date"),
) -> dict:
    _require_token(x_admin_token)
    target = date.fromisoformat(day) if day else None
    background.add_task(fetch_upcoming, target)
    return {"queued": "fetch_upcoming", "date": day or "danas"}


@router.post("/update-results")
def trigger_results(
    background: BackgroundTasks,
    x_admin_token: str | None = Header(default=None),
    lookback_days: int = Query(default=3, ge=1, le=30),
) -> dict:
    _require_token(x_admin_token)
    background.add_task(update_results, lookback_days)
    return {"queued": "update_results", "lookback_days": lookback_days}


@router.post("/backfill")
def trigger_backfill(
    background: BackgroundTasks,
    x_admin_token: str | None = Header(default=None),
) -> dict:
    _require_token(x_admin_token)
    background.add_task(backfill_history)
    return {
        "queued": "backfill_history",
        "leagues": settings.league_ids,
        "seasons": settings.seasons,
        "napomena": "Dugotrajno; prati napredak preko /api/status",
    }


@router.post("/reload-model")
def reload_model(x_admin_token: str | None = Header(default=None)) -> dict:
    """Učitava nanovo artefakte modela nakon treniranja, bez restarta servera."""
    _require_token(x_admin_token)
    pred_service.clear_cache()
    return {"reloaded": True, "model": pred_service.model_info()}
