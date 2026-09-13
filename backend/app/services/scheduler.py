"""Raspoređivanje dnevnih poslova (zamjena za node-cron).

Dva redovita posla:
  00:05  dohvat utakmica za tekući dan, povijesti momčadi i izostanaka
  02:30  upis ishoda odigranih utakmica  ← ovo stvara oznake za učenje

Uz to i nadoknada propuštenog: raspoređivač živi samo dok radi poslužitelj, pa
bi se svaki dan kad je računalo u 00:05 bilo ugašeno trajno preskočio. Zato se
pri pokretanju provjerava jesu li podaci za danas svježi i, ako nisu, dohvat se
pokreće odmah.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import func, select

from app.config import settings
from app.db import session_scope
from app.models import Fixture, IngestLog
from app.services.ingest import fetch_odds_by_date, fetch_upcoming, update_results

log = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None

# Nakon koliko sati se dnevni dohvat smatra zastarjelim.
STALE_AFTER_HOURS = 20

# Koliko često se provjerava svježina podataka (minute).
WATCHDOG_INTERVAL_MINUTES = 30

# Najmanji razmak između dva pokušaja dohvata, da se pri trajnoj grešci
# (npr. istekla kvota) ne troše pozivi u krug.
RETRY_COOLDOWN_HOURS = 2


def _safe(job_name: str, fn, *args) -> None:
    try:
        log.info("Pokrecem posao: %s", job_name)
        result = fn(*args)
        log.info("Posao %s zavrsen: %s", job_name, result)
    except Exception as exc:  # noqa: BLE001
        # Greška u jednom pokretanju ne smije srušiti raspoređivač.
        log.exception("Posao %s nije uspio: %s", job_name, exc)


def _parse_cron(value: str) -> tuple[int, int]:
    minute, hour = value.split()
    return int(minute), int(hour)


def needs_daily_fetch() -> tuple[bool, str]:
    """Jesu li podaci za današnji dan svježi? Vraća (treba li dohvat, razlog)."""
    now = datetime.now(timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    def _age_hours(value: datetime | None) -> float | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return (now - value).total_seconds() / 3600

    with session_scope() as db:
        today_count = db.scalar(
            select(func.count(Fixture.id)).where(
                Fixture.kickoff >= start, Fixture.kickoff < start + timedelta(days=1)
            )
        ) or 0
        last_ok = db.scalar(
            select(IngestLog.finished_at)
            .where(IngestLog.job == "fetch_upcoming", IngestLog.status == "ok")
            .order_by(IngestLog.id.desc())
            .limit(1)
        )
        last_any = db.scalar(
            select(IngestLog.finished_at)
            .where(IngestLog.job == "fetch_upcoming")
            .order_by(IngestLog.id.desc())
            .limit(1)
        )

    ok_age = _age_hours(last_ok)
    any_age = _age_hours(last_any)

    # Zaustavi vrtnju u krug ako dohvat trajno pada (istekla kvota, nema mreze).
    if any_age is not None and any_age < RETRY_COOLDOWN_HOURS and (
        ok_age is None or ok_age > STALE_AFTER_HOURS
    ):
        return False, f"nedavni pokusaj prije {any_age:.1f} h — cekam"

    if today_count == 0:
        return True, "nema utakmica za danas u bazi"
    if ok_age is None:
        return True, "nema zabiljezenog uspjesnog dohvata"
    if ok_age > STALE_AFTER_HOURS:
        return True, f"zadnji dohvat prije {ok_age:.0f} h"

    return False, f"podaci su svjezi ({today_count} utakmica, dohvat prije {ok_age:.0f} h)"


def _watchdog() -> None:
    """Periodična provjera svježine.

    Cron sam po sebi nije dovoljan na prijenosnom računalu: ako uređaj spava u
    00:05, termin se propusti, a APScheduler ga odbija naknadno pokrenuti kad
    kašnjenje premaši dopušteno. Ova provjera hvata upravo taj slučaj — čim se
    računalo probudi, primijeti da podaci nisu svježi i pokrene dohvat.
    """
    try:
        needed, reason = needs_daily_fetch()
    except Exception as exc:  # noqa: BLE001
        log.warning("Provjera svjezine nije uspjela: %s", exc)
        return
    if not needed:
        log.debug("Provjera svjezine: %s", reason)
        return
    log.info("Provjera svjezine je pokrenula dohvat: %s", reason)
    _safe("fetch_upcoming (provjera svjezine)", fetch_upcoming)
    _safe("update_results (provjera svjezine)", update_results)


def start_scheduler() -> BackgroundScheduler | None:
    global _scheduler
    if not settings.enable_scheduler:
        log.info("Rasporedivac je iskljucen (ENABLE_SCHEDULER=false)")
        return None
    if not settings.football_api_key:
        log.warning("FOOTBALL_API_KEY nije postavljen — rasporedivac se ne pokrece")
        return None
    if _scheduler is not None:
        return _scheduler

    scheduler = BackgroundScheduler(timezone=settings.timezone)

    fetch_min, fetch_hour = _parse_cron(settings.daily_fixtures_cron)
    results_min, results_hour = _parse_cron(settings.daily_results_cron)

    scheduler.add_job(
        _safe,
        CronTrigger(hour=fetch_hour, minute=fetch_min, timezone=settings.timezone),
        args=["fetch_upcoming", fetch_upcoming],
        id="fetch_upcoming",
        replace_existing=True,
        # `None` = izvrsi propusteni termin bez obzira na kasnjenje. Uz
        # `coalesce` se vise propustenih termina svede na jedno pokretanje.
        # Bez ovoga bi se posao nakon dugog spavanja uredaja tiho preskocio.
        misfire_grace_time=None,
        coalesce=True,
    )
    scheduler.add_job(
        _safe,
        CronTrigger(hour=results_hour, minute=results_min, timezone=settings.timezone),
        args=["update_results", update_results],
        id="update_results",
        replace_existing=True,
        misfire_grace_time=None,
        coalesce=True,
    )

    odds_min, odds_hour = _parse_cron(settings.daily_odds_cron)
    scheduler.add_job(
        _safe,
        CronTrigger(hour=odds_hour, minute=odds_min, timezone=settings.timezone),
        args=["fetch_odds_by_date", fetch_odds_by_date,
              settings.odds_days_back, settings.odds_days_ahead],
        id="fetch_odds_by_date",
        replace_existing=True,
        misfire_grace_time=None,
        coalesce=True,
    )

    scheduler.add_job(
        _watchdog,
        IntervalTrigger(minutes=WATCHDOG_INTERVAL_MINUTES),
        id="freshness_watchdog",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
    )

    scheduler.start()
    _scheduler = scheduler
    log.info(
        "Rasporedivac pokrenut (%s): dohvat %02d:%02d, ishodi %02d:%02d, koeficijenti %02d:%02d",
        settings.timezone, fetch_hour, fetch_min, results_hour, results_min, odds_hour, odds_min,
    )

    # ── Nadoknada propuštenog ──
    try:
        needed, reason = needs_daily_fetch()
    except Exception as exc:  # noqa: BLE001
        log.warning("Provjera svjezine podataka nije uspjela: %s", exc)
        needed, reason = False, str(exc)

    if needed:
        log.info("Nadoknada dohvata pri pokretanju: %s", reason)
        # Kratka odgoda da poslužitelj prvo do kraja podigne HTTP sloj.
        scheduler.add_job(
            _safe,
            DateTrigger(run_date=datetime.now(timezone.utc) + timedelta(seconds=10)),
            args=["fetch_upcoming (nadoknada)", fetch_upcoming],
            id="catchup_fetch",
            replace_existing=True,
        )
        scheduler.add_job(
            _safe,
            DateTrigger(run_date=datetime.now(timezone.utc) + timedelta(minutes=30)),
            args=["update_results (nadoknada)", update_results],
            id="catchup_results",
            replace_existing=True,
        )
    else:
        log.info("Nadoknada nije potrebna: %s", reason)

    return scheduler


def scheduler_status() -> dict:
    """Stanje raspoređivača za /api/status."""
    if _scheduler is None:
        return {"running": False, "enabled": settings.enable_scheduler, "jobs": []}
    return {
        "running": True,
        "enabled": True,
        "timezone": settings.timezone,
        "jobs": [
            {
                "id": job.id,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            }
            for job in _scheduler.get_jobs()
        ],
    }


def shutdown_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
