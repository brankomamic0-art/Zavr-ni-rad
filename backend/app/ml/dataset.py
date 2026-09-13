"""Učitavanje podataka iz baze u pandas i sastavljanje skupa za učenje."""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import engine
from app.ml.features import attach_odds_features, build_feature_matrix

log = logging.getLogger(__name__)

_FIXTURE_SQL = """
    SELECT  f.id            AS fixture_id,
            f.kickoff       AS kickoff,
            f.league_id     AS league_id,
            f.season        AS season,
            f.home_team_id  AS home_team_id,
            f.away_team_id  AS away_team_id,
            f.home_goals    AS home_goals,
            f.away_goals    AS away_goals,
            f.home_ft       AS home_ft,
            f.away_ft       AS away_ft,
            f.finished      AS finished,
            l.tier          AS tier,
            l.country       AS country
    FROM fixtures f
    JOIN leagues l ON l.id = f.league_id
"""


def load_fixtures(
    conn=None,
    finished_only: bool = False,
    team_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Učitava utakmice iz baze."""
    sql = _FIXTURE_SQL
    clauses = []
    params: dict = {}

    if finished_only:
        clauses.append("f.finished = TRUE AND f.home_goals IS NOT NULL")
    if team_ids:
        ids = ",".join(str(int(t)) for t in team_ids)
        clauses.append(f"(f.home_team_id IN ({ids}) OR f.away_team_id IN ({ids}))")
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY f.kickoff"

    target = conn if conn is not None else engine
    df = pd.read_sql(text(sql), target, params=params)
    if df.empty:
        return df

    df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True)
    df["finished"] = df["finished"].astype(bool)
    return df


def load_odds(conn=None, fixture_ids: list[int] | None = None) -> pd.DataFrame:
    sql = "SELECT fixture_id, market, price, bookmaker FROM odds"
    if fixture_ids:
        ids = ",".join(str(int(f)) for f in fixture_ids)
        sql += f" WHERE fixture_id IN ({ids})"
    target = conn if conn is not None else engine
    return pd.read_sql(text(sql), target)


def load_fixture_stats(conn=None, fixture_ids: list[int] | None = None) -> pd.DataFrame:
    """Statistika utakmica (udarci, kutovi) — izvor značajki iz udaraca."""
    sql = "SELECT fixture_id, team_id, shots_on, shots_total, shots_box, corners FROM fixture_stats"
    if fixture_ids:
        ids = ",".join(str(int(f)) for f in fixture_ids)
        sql += f" WHERE fixture_id IN ({ids})"
    target = conn if conn is not None else engine
    return pd.read_sql(text(sql), target)


def load_injury_counts(conn=None, fixture_ids: list[int] | None = None) -> pd.DataFrame:
    sql = "SELECT fixture_id, team_id, COUNT(*) AS n FROM injuries"
    if fixture_ids:
        ids = ",".join(str(int(f)) for f in fixture_ids)
        sql += f" WHERE fixture_id IN ({ids})"
    sql += " GROUP BY fixture_id, team_id"
    target = conn if conn is not None else engine
    return pd.read_sql(text(sql), target)


def _attach_injuries(matrix: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """Broj izostanaka po momčadi. Za povijesne utakmice podatak najčešće ne
    postoji (API ga ne nudi retroaktivno), pa ostaje 0."""
    matrix = matrix.copy()
    matrix["h_injuries"] = 0.0
    matrix["a_injuries"] = 0.0
    if injuries is None or injuries.empty:
        return matrix

    home = injuries.rename(columns={"team_id": "home_team_id", "n": "h_inj"})
    away = injuries.rename(columns={"team_id": "away_team_id", "n": "a_inj"})
    matrix = matrix.merge(home, on=["fixture_id", "home_team_id"], how="left")
    matrix = matrix.merge(away, on=["fixture_id", "away_team_id"], how="left")
    matrix["h_injuries"] = matrix.pop("h_inj").fillna(0.0)
    matrix["a_injuries"] = matrix.pop("a_inj").fillna(0.0)
    return matrix


def build_training_set(
    min_history: int = 5,
    with_odds: bool = False,
    since: datetime | None = None,
) -> pd.DataFrame:
    """Sastavlja skup za treniranje iz svih odigranih utakmica u bazi.

    Args:
        min_history: koliko ranijih utakmica momčad mora imati da bi utakmica
            ušla u skup. Prvih nekoliko kola svake momčadi ima nepouzdanu formu
            (prosjek od jedne ili dvije utakmice), pa se izbacuju.
        with_odds: dodaje li se skup značajki izveden iz koeficijenata.
        since: opcionalno ograničenje na utakmice od zadanog datuma.
    """
    fixtures = load_fixtures(finished_only=True)
    if fixtures.empty:
        log.warning("Baza ne sadrži nijednu odigranu utakmicu — pokreni backfill.")
        return pd.DataFrame()

    if since is not None:
        targets = fixtures[fixtures["kickoff"] >= pd.Timestamp(since, tz="UTC")]
    else:
        targets = fixtures

    matrix = build_feature_matrix(fixtures, targets=targets, stats=load_fixture_stats())
    if matrix.empty:
        return matrix

    matrix = _attach_injuries(matrix, load_injury_counts())
    if with_odds:
        matrix = attach_odds_features(matrix, load_odds())

    before = len(matrix)
    matrix = matrix[
        (matrix["h_ov_n"] >= min_history) & (matrix["a_ov_n"] >= min_history)
    ].copy()
    matrix = matrix.dropna(subset=["y_over25", "y_btts"])
    log.info(
        "Skup za treniranje: %s redaka (odbačeno %s zbog prekratke povijesti)",
        len(matrix),
        before - len(matrix),
    )
    return matrix.sort_values("kickoff").reset_index(drop=True)


def build_serving_matrix(
    db: Session,
    fixture_ids: list[int],
    with_odds: bool = False,
) -> pd.DataFrame:
    """Gradi značajke za utakmice koje tek slijede.

    Koristi identičan kod kao treniranje — zato je forma za nadolazeću utakmicu
    izračunata na potpuno isti način kao i za povijesnu.
    """
    if not fixture_ids:
        return pd.DataFrame()

    conn = db.connection()
    all_fixtures = load_fixtures(conn=conn)
    if all_fixtures.empty:
        return pd.DataFrame()

    targets = all_fixtures[all_fixtures["fixture_id"].isin(fixture_ids)]
    if targets.empty:
        return pd.DataFrame()

    # Statistika se ucitava za SVE utakmice, ne samo ciljane: klizni prosjek
    # nadolazece utakmice racuna se iz ranijih utakmica tih momcadi.
    matrix = build_feature_matrix(
        all_fixtures, targets=targets, stats=load_fixture_stats(conn=conn)
    )
    if matrix.empty:
        return matrix

    matrix = _attach_injuries(matrix, load_injury_counts(conn=conn, fixture_ids=fixture_ids))
    if with_odds:
        matrix = attach_odds_features(matrix, load_odds(conn=conn, fixture_ids=fixture_ids))
    return matrix


def chronological_split(
    df: pd.DataFrame, cutoff: str | datetime
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kronološka podjela na skup za treniranje i skup za testiranje.

    Nasumična podjela (`train_test_split(shuffle=True)`) je kod vremenskih
    podataka pogrešna: model bi učio na budućim utakmicama i predviđao prošle,
    što u stvarnoj primjeni nikad nije moguće i daje lažno dobre rezultate.
    """
    ts = pd.Timestamp(cutoff)
    # Granica smije stići kao tekst bez zone ili kao Timestamp sa zonom.
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    train = df[df["kickoff"] < ts].copy()
    test = df[df["kickoff"] >= ts].copy()
    return train, test
