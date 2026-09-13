"""Konfiguracija aplikacije — sve se čita iz okoline (.env)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # ── Server ──
    app_name: str = "Football Predictor API"
    port: int = 8000
    cors_origins: str = "*"

    # ── Baza ──
    # Prazno -> lokalni SQLite (nula konfiguracije za razvoj i pisanje rada).
    database_url: str = ""

    # ── API-Football ──
    football_api_key: str = ""
    football_api_host: str = "v3.football.api-sports.io"
    api_request_delay: float = 0.18  # sekunde između poziva (rate limit)

    # ── Domena ──
    # Lige koje ulaze u povijesni dataset (backfill). Manje liga = čistiji podaci.
    backfill_league_ids: str = "39,140,135,78,61,88,94,144,203,2,3"
    backfill_seasons: str = "2019,2020,2021,2022,2023,2024,2025"

    # Veličine prozora forme nisu proizvoljne — izmjerene su usporedbom
    # 5/10/20/40 na istoj kronološkoj podjeli: skill za Over 2.5 raste
    # 1,64 % -> 2,29 % -> 2,62 %, a od 20 naviše se zasićuje (40 daje 2,65 %).
    # Duži prozor opisuje kvalitetu momčadi, kratki njezinu trenutnu formu.
    form_window_overall: int = 20  # zadnjih N utakmica ukupno
    form_window_venue: int = 10    # zadnjih N kod kuće / u gostima
    form_window_recent: int = 5    # kratki prozor — dopuna, ne zamjena
    # Najmanji broj ranijih utakmica koje momčad mora imati da bi se za nju
    # uopće izdala predikcija. Momčad s dvije odigrane utakmice može imati 2/2
    # na Over 2.5, što izgleda kao 100 % a nije nikakav dokaz.
    serving_min_history: int = 5

    # ── ML ──
    model_dir: str = str(BASE_DIR / "models")
    active_model: str = "logreg"   # koji se model poslužuje preko API-ja
    # Kronološka granica train/test podjele (sve prije = train).
    train_cutoff: str = "2024-07-01"

    # ── Raspored (APScheduler) ──
    timezone: str = "Europe/Zagreb"
    enable_scheduler: bool = True
    daily_fixtures_cron: str = "5 0"    # 00:05 — dohvat današnjih utakmica
    daily_results_cron: str = "30 2"    # 02:30 — upis ishoda jučerašnjih utakmica
    daily_odds_cron: str = "15 3"       # 03:15 — koeficijenti (7 dana unatrag + 2 unaprijed)
    # Koeficijenti se dohvaćaju po datumu (10 utakmica po pozivu). API čuva
    # samo 7 dana povijesti, pa je ovo jedini način da se skup gradi.
    odds_days_back: int = 7
    odds_days_ahead: int = 2

    # ── Zaštita admin ruta (prazno = rute su onemogućene) ──
    admin_token: str = ""

    @property
    def sqlalchemy_url(self) -> str:
        url = self.database_url.strip()
        if not url:
            return f"sqlite:///{BASE_DIR / 'football.db'}"
        # Heroku/Railway stil "postgres://" -> SQLAlchemy 2.0 + psycopg3
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+psycopg://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        return url

    @property
    def league_ids(self) -> list[int]:
        return [int(x) for x in self.backfill_league_ids.split(",") if x.strip()]

    @property
    def seasons(self) -> list[int]:
        return [int(x) for x in self.backfill_seasons.split(",") if x.strip()]

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
