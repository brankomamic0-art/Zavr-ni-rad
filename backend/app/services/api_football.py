"""Klijent za javno dostupni API-Football (v3.football.api-sports.io)."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from app.config import settings

log = logging.getLogger(__name__)

# Lige koje se u sustavu tretiraju kao najviša razina (utječe na značajku `tier`).
# 39 Premier League, 140 La Liga, 135 Serie A, 78 Bundesliga, 61 Ligue 1,
# 2 Liga prvaka, 3 Europska liga, 848 Konferencijska liga,
# 71 Brazil Serie A, 128 Argentina Liga Profesional,
# 13 Copa Libertadores, 11 Copa Sudamericana
TIER1_LEAGUE_IDS = {39, 140, 135, 78, 61, 2, 3, 848, 71, 128, 13, 11}
TIER2_LEAGUE_IDS = {40, 79, 136, 141, 62, 88, 94, 144, 203, 179, 239, 265, 268, 281, 72, 129}
TIER2_COUNTRIES = {
    "Netherlands", "Portugal", "Belgium", "Turkey", "Scotland", "Austria",
    "Switzerland", "Brazil", "Argentina",
}

EUROPEAN_COUNTRIES = {
    "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus", "Belgium",
    "Bosnia", "Bosnia-And-Herzegovina", "Bulgaria", "Croatia", "Cyprus",
    "Czech-Republic", "Denmark", "England", "Estonia", "Faroe-Islands", "Finland",
    "France", "Georgia", "Germany", "Gibraltar", "Greece", "Hungary", "Iceland",
    "Ireland", "Israel", "Italy", "Kazakhstan", "Kosovo", "Latvia", "Liechtenstein",
    "Lithuania", "Luxembourg", "Malta", "Moldova", "Monaco", "Montenegro",
    "Netherlands", "North-Macedonia", "Northern-Ireland", "Norway", "Poland",
    "Portugal", "Romania", "Russia", "San-Marino", "Scotland", "Serbia", "Slovakia",
    "Slovenia", "Spain", "Sweden", "Switzerland", "Turkey", "Ukraine", "Wales",
}

SOUTH_AMERICAN_COUNTRIES = {
    "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
    "Paraguay", "Peru", "Uruguay", "Venezuela",
}

# API-Football pod "World" vodi međunarodna natjecanja — Ligu prvaka, Europsku
# ligu, Copa Libertadores, Copa Sudamericana i slično.
INTERNATIONAL_COUNTRIES = {"World"}

SUPPORTED_COUNTRIES = (
    EUROPEAN_COUNTRIES | SOUTH_AMERICAN_COUNTRIES | INTERNATIONAL_COUNTRIES
)

# Status kodovi API-ja koji znače "utakmica je regularno odigrana do kraja".
FINISHED_STATUSES = {"FT", "AET", "PEN"}


class ApiFootballError(RuntimeError):
    pass


class ApiFootballClient:
    """Tanki sinkroni omotač oko REST API-ja, s brojačem poziva i pauzama."""

    def __init__(self, api_key: str | None = None, timeout: float = 30.0) -> None:
        self.api_key = api_key or settings.football_api_key
        if not self.api_key:
            raise ApiFootballError(
                "FOOTBALL_API_KEY nije postavljen — dodaj ga u backend/.env"
            )
        self.base_url = f"https://{settings.football_api_host}"
        self.calls = 0
        self._client = httpx.Client(
            timeout=timeout,
            headers={"x-apisports-key": self.api_key, "Accept": "application/json"},
        )

    def __enter__(self) -> "ApiFootballClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def get(self, endpoint: str, **params: Any) -> list[dict]:
        url = f"{self.base_url}{endpoint}"
        for attempt in range(3):
            try:
                resp = self._client.get(url, params=params)
                self.calls += 1
                if resp.status_code == 429:
                    wait = 5 * (attempt + 1)
                    log.warning("Rate limit, čekam %ss...", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                payload = resp.json()
                errors = payload.get("errors")
                # API vraća [] kad nema grešaka, a {} ili {"key": "poruka"} kad ima.
                if isinstance(errors, dict) and errors:
                    raise ApiFootballError("; ".join(str(v) for v in errors.values()))
                return payload.get("response") or []
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt == 2:
                    raise ApiFootballError(f"Mrežna greška: {exc}") from exc
                time.sleep(2 * (attempt + 1))
            finally:
                time.sleep(settings.api_request_delay)
        return []

    # ── Endpointi ──

    def fixtures_by_date(self, date_str: str, timezone: str | None = None) -> list[dict]:
        return self.get("/fixtures", date=date_str, timezone=timezone or settings.timezone)

    def fixtures_by_league_season(self, league_id: int, season: int) -> list[dict]:
        """Sve utakmice jedne lige u jednoj sezoni — temelj povijesnog dataseta."""
        return self.get("/fixtures", league=league_id, season=season)

    def fixtures_by_team(self, team_id: int, last: int = 20) -> list[dict]:
        """Zadnjih N odigranih utakmica jedne momcadi — izvor forme."""
        return self.get("/fixtures", team=team_id, last=last, status="FT")

    def fixtures_by_ids(self, fixture_ids: list[int]) -> list[dict]:
        """Do 20 ID-eva po pozivu — koristi se za naknadni upis rezultata."""
        return self.get("/fixtures", ids="-".join(str(i) for i in fixture_ids))

    def odds_for_fixture(self, fixture_id: int) -> list[dict]:
        return self.get("/odds", fixture=fixture_id)

    def injuries_for_fixtures(self, fixture_ids: list[int]) -> list[dict]:
        return self.get("/injuries", ids="-".join(str(i) for i in fixture_ids))


def classify_tier(league_id: int, country: str, name: str) -> int:
    if league_id in TIER1_LEAGUE_IDS:
        return 1
    lowered = (name or "").lower()
    if country == "World" and any(
        k in lowered for k in ("champion", "europa", "conference")
    ):
        return 1
    if league_id in TIER2_LEAGUE_IDS or country in TIER2_COUNTRIES:
        return 2
    return 3


def normalize_country(raw: str | None) -> str:
    return (raw or "").replace(" ", "-")


def is_european(country: str) -> bool:
    return country in EUROPEAN_COUNTRIES


def is_supported(country: str) -> bool:
    """Europa + Južna Amerika + međunarodna natjecanja."""
    return country in SUPPORTED_COUNTRIES
