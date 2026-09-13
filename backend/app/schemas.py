"""Pydantic sheme — ujedno i automatska dokumentacija API-ja na /docs."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TeamOut(BaseModel):
    id: int
    name: str
    logo: str | None = None


class LeagueOut(BaseModel):
    id: int
    name: str
    country: str
    type: str | None = None
    tier: int = 3


class FormBlock(BaseModel):
    """Forma jedne momčadi u jednom prozoru (ukupno ili dom/gost)."""

    n: int = Field(description="Broj utakmica u prozoru")
    over25: int | None = Field(default=None, description="Koliko ih je imalo 3+ gola")
    btts: int | None = Field(default=None, description="Koliko ih je imalo pogodak obiju momčadi")
    cs: int | None = Field(default=None, description="Utakmice bez primljenog gola")
    fts: int | None = Field(default=None, description="Utakmice bez zabijenog gola")
    gf: float | None = Field(default=None, description="Prosjek zabijenih")
    ga: float | None = Field(default=None, description="Prosjek primljenih")
    tot: float | None = Field(default=None, description="Prosjek ukupnih golova")
    ppg: float | None = Field(default=None, description="Bodovi po utakmici")


class AnalysisOut(BaseModel):
    hO: FormBlock
    aO: FormBlock
    hV: FormBlock
    aV: FormBlock


class MarketPrediction(BaseModel):
    p: float | None = Field(default=None, description="Procijenjena vjerojatnost 0-1")
    mode: str = Field(description="'model' ili 'heuristika'")
    source: str = Field(description="Naziv modela koji je dao procjenu")


class OddOut(BaseModel):
    price: float
    bookmaker: str | None = None


class InjuryOut(BaseModel):
    name: str
    reason: str | None = None
    type: str | None = None


class MatchOut(BaseModel):
    id: int
    kickoff: datetime
    status: str
    category: str = Field(
        default="senior",
        description="senior | youth | women | reserve — vrsta natjecanja",
    )
    league: LeagueOut
    home: TeamOut
    away: TeamOut
    analysis: AnalysisOut | None = None
    predictions: dict[str, MarketPrediction] = {}
    odds: dict[str, OddOut] = {}
    injuries: dict[str, list[InjuryOut]] = {}
    value: dict[str, float] = Field(
        default={}, description="Prednost nad tržištem: p * koeficijent - 1"
    )
    home_goals: int | None = None
    away_goals: int | None = None


class MatchesResponse(BaseModel):
    date: str
    generated_at: datetime
    count: int
    model: dict
    matches: list[MatchOut]


class StatusResponse(BaseModel):
    database: str
    fixtures_total: int
    fixtures_finished: int
    fixtures_today: int
    teams: int
    leagues: int
    earliest: datetime | None = None
    latest: datetime | None = None
    last_jobs: list[dict] = []
    model: dict = {}
    scheduler: dict = Field(default={}, description="Stanje rasporedivaca i sljedeca pokretanja")
    training_ready: bool = Field(
        description="Ima li baza dovoljno odigranih utakmica za smisleno treniranje"
    )
