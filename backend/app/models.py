"""ORM model — trajna pohrana povijesnih utakmica i njihovih ishoda.

Ovo je ključna razlika u odnosu na prethodnu verziju sustava: raniji
`fetchData.js` je svaki dan prepisivao jedan JSON s današnjim utakmicama, pa
oznake (ishodi) nikada nisu bile sačuvane i model se nije imao na čemu učiti.
Ovdje se svaka utakmica trajno zapisuje, a zaseban posao naknadno upisuje
rezultat kada utakmica završi.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class League(Base):
    __tablename__ = "leagues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    name: Mapped[str] = mapped_column(String(160))
    country: Mapped[str] = mapped_column(String(80), index=True)
    type: Mapped[str | None] = mapped_column(String(20))
    tier: Mapped[int] = mapped_column(Integer, default=3)
    logo: Mapped[str | None] = mapped_column(String(300))


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    name: Mapped[str] = mapped_column(String(160))
    country: Mapped[str | None] = mapped_column(String(80))
    logo: Mapped[str | None] = mapped_column(String(300))


class Fixture(Base):
    """Jedna utakmica. `home_goals`/`away_goals` su NULL dok se ne odigra."""

    __tablename__ = "fixtures"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"), index=True)
    season: Mapped[int | None] = mapped_column(Integer, index=True)
    kickoff: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)

    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)

    status: Mapped[str] = mapped_column(String(10), default="NS", index=True)
    finished: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    home_goals: Mapped[int | None] = mapped_column(Integer)
    away_goals: Mapped[int | None] = mapped_column(Integer)
    home_ht: Mapped[int | None] = mapped_column(Integer)
    away_ht: Mapped[int | None] = mapped_column(Integer)
    # Rezultat nakon 90 minuta (`score.fulltime`), BEZ produzetaka i penala.
    # `home_goals`/`away_goals` su konacni rezultat i kod statusa AET ukljucuju
    # golove iz produzetaka — a kladionice trziste Over/Under 2.5 namiruju
    # iskljucivo na regularnih 90 minuta. Zato je ovo ispravna oznaka za ucenje.
    home_ft: Mapped[int | None] = mapped_column(Integer)
    away_ft: Mapped[int | None] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    league: Mapped[League] = relationship(lazy="joined")
    home_team: Mapped[Team] = relationship(foreign_keys=[home_team_id], lazy="joined")
    away_team: Mapped[Team] = relationship(foreign_keys=[away_team_id], lazy="joined")

    __table_args__ = (Index("ix_fixtures_finished_kickoff", "finished", "kickoff"),)


class Odd(Base):
    """Najbolji koeficijent po tržištu (max preko svih kladionica)."""

    __tablename__ = "odds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(ForeignKey("fixtures.id", ondelete="CASCADE"), index=True)
    market: Mapped[str] = mapped_column(String(20))  # over25 | under25 | btts | btts_no
    price: Mapped[float] = mapped_column(Float)
    bookmaker: Mapped[str | None] = mapped_column(String(80))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (UniqueConstraint("fixture_id", "market", name="uq_odds_fixture_market"),)


class Injury(Base):
    __tablename__ = "injuries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(ForeignKey("fixtures.id", ondelete="CASCADE"), index=True)
    team_id: Mapped[int] = mapped_column(Integer, index=True)
    player_name: Mapped[str] = mapped_column(String(160))
    reason: Mapped[str | None] = mapped_column(String(160))
    type: Mapped[str | None] = mapped_column(String(80))

    __table_args__ = (
        UniqueConstraint("fixture_id", "team_id", "player_name", name="uq_injury_row"),
    )


class FixtureStat(Base):
    """Statistika jedne momcadi na jednoj utakmici.

    Izvor je `/fixtures?ids=` koji uz utakmice vraca i `statistics` — do 20
    utakmica po jednom pozivu. Bez toga model vidi samo golove, a gol je rijedak
    dogadaj (~2,8 po utakmici); udarci mjere istu namjeru s desetak puta vise
    uzoraka, pa je procjena napadacke snage znatno stabilnija.
    """

    __tablename__ = "fixture_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(ForeignKey("fixtures.id", ondelete="CASCADE"), index=True)
    team_id: Mapped[int] = mapped_column(Integer, index=True)

    shots_on: Mapped[float | None] = mapped_column(Float)      # Shots on Goal
    shots_total: Mapped[float | None] = mapped_column(Float)   # Total Shots
    shots_box: Mapped[float | None] = mapped_column(Float)     # Shots insidebox
    corners: Mapped[float | None] = mapped_column(Float)       # Corner Kicks
    possession: Mapped[float | None] = mapped_column(Float)    # Ball Possession (%)
    saves: Mapped[float | None] = mapped_column(Float)         # Goalkeeper Saves

    __table_args__ = (
        UniqueConstraint("fixture_id", "team_id", name="uq_fixture_stat"),
    )


class Prediction(Base):
    """Predikcija spremljena u trenutku izdavanja.

    Bitno za rad: čuva se ono što je model tvrdio PRIJE utakmice, pa se poslije
    može pošteno evaluirati na stvarno neviđenim podacima (prospektivna
    evaluacija, za razliku od retrospektivnog backtesta).
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(ForeignKey("fixtures.id", ondelete="CASCADE"), index=True)
    market: Mapped[str] = mapped_column(String(20))       # over25 | btts
    model_name: Mapped[str] = mapped_column(String(60))
    model_version: Mapped[str] = mapped_column(String(40), default="v1")
    probability: Mapped[float] = mapped_column(Float)
    features: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("fixture_id", "market", "model_name", name="uq_pred_fixture_market_model"),
    )


class IngestLog(Base):
    """Trag izvršavanja poslova dohvata — za dijagnostiku i za poglavlje o sustavu."""

    __tablename__ = "ingest_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job: Mapped[str] = mapped_column(String(60), index=True)
    status: Mapped[str] = mapped_column(String(20))  # ok | error
    detail: Mapped[str | None] = mapped_column(String(1000))
    api_calls: Mapped[int] = mapped_column(Integer, default=0)
    rows: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
