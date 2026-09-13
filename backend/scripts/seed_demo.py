"""Puni bazu demo podacima da se sučelje može vidjeti bez API ključa.

Podaci su IZMIŠLJENI — nazivi liga i momčadi su stvarni radi realnog izgleda,
ali svi rezultati, koeficijenti i izostanci su generirani. Skripta služi
isključivo za pregled sučelja i za razvoj; za rad se koriste stvarni podaci
dohvaćeni preko `scripts.backfill`.

    python -m scripts.seed_demo            # dvije sezone povijesti + danasnje utakmice
    python -m scripts.seed_demo --reset    # prvo obrise postojece podatke
    python -m scripts.seed_demo --seasons 3

Nakon toga:
    python -m app.ml.train --cutoff <datum>
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import delete

from app.db import init_db, session_scope
from app.models import Fixture, Injury, League, Odd, Prediction, Team

# ── Demo prvenstva ──
LEAGUES = [
    (39, "Premier League", "England", 1),
    (140, "La Liga", "Spain", 1),
    (135, "Serie A", "Italy", 1),
    (78, "Bundesliga", "Germany", 1),
]

TEAMS = {
    39: [
        "Arsenal", "Aston Villa", "Brentford", "Brighton", "Chelsea",
        "Crystal Palace", "Everton", "Fulham", "Liverpool", "Manchester City",
        "Manchester United", "Newcastle", "Nottingham Forest", "Tottenham",
        "West Ham", "Wolves",
    ],
    140: [
        "Athletic Club", "Atletico Madrid", "Barcelona", "Betis", "Celta Vigo",
        "Getafe", "Girona", "Mallorca", "Osasuna", "Rayo Vallecano",
        "Real Madrid", "Real Sociedad", "Sevilla", "Valencia", "Villarreal",
        "Alaves",
    ],
    135: [
        "Atalanta", "Bologna", "Cagliari", "Empoli", "Fiorentina", "Genoa",
        "Inter", "Juventus", "Lazio", "Lecce", "Milan", "Napoli", "Roma",
        "Torino", "Udinese", "Verona",
    ],
    78: [
        "Augsburg", "Bayer Leverkusen", "Bayern Munchen", "Bochum",
        "Borussia Dortmund", "Eintracht Frankfurt", "Freiburg", "Heidenheim",
        "Hoffenheim", "Mainz", "Monchengladbach", "RB Leipzig", "St. Pauli",
        "Stuttgart", "Union Berlin", "Werder Bremen",
    ],
}

INJURY_REASONS = [
    "Ozljeda koljena", "Istegnuce misica", "Ozljeda glezna", "Suspenzija",
    "Bolest", "Ozljeda ledja", "Problemi s preponama",
]

# Koliko dana unaprijed se generiraju neodigrane utakmice.
UPCOMING_DAYS = 4

HOME_ADVANTAGE = 1.22
rng = random.Random(2026)
nprng = np.random.default_rng(2026)


def _wipe() -> None:
    with session_scope() as db:
        for model in (Prediction, Odd, Injury, Fixture, Team, League):
            db.execute(delete(model))
    print("  Postojeci podaci obrisani.")


def _build_entities() -> dict[int, dict[int, str]]:
    """Upisuje lige i momčadi, vraća {league_id: {team_id: naziv}}."""
    registry: dict[int, dict[int, str]] = {}
    team_id = 1000

    with session_scope() as db:
        for league_id, name, country, tier in LEAGUES:
            db.add(League(id=league_id, name=name, country=country, type="League", tier=tier))
            registry[league_id] = {}
            for team_name in TEAMS[league_id]:
                team_id += 1
                db.add(Team(id=team_id, name=team_name, country=country))
                registry[league_id][team_id] = team_name
    return registry


def _strengths(team_ids: list[int]) -> dict[int, tuple[float, float]]:
    """Skrivena jakost napada i obrane po momčadi (ono što model treba nazrijeti)."""
    strengths = {}
    for tid in team_ids:
        strengths[tid] = (rng.uniform(0.75, 2.05), rng.uniform(0.65, 1.55))
    return strengths


def _score(attack_home, defence_away, attack_away, defence_home) -> tuple[int, int]:
    lam_home = attack_home * defence_away * HOME_ADVANTAGE
    lam_away = attack_away * defence_home
    return int(nprng.poisson(lam_home)), int(nprng.poisson(lam_away))


def _round_robin(team_ids: list[int]) -> list[list[tuple[int, int]]]:
    """Klasičan kružni raspored — svaka momčad igra svaku, jednom po krugu."""
    teams = list(team_ids)
    n = len(teams)
    rounds = []
    for _ in range(n - 1):
        pairs = [(teams[i], teams[n - 1 - i]) for i in range(n // 2)]
        rounds.append(pairs)
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    return rounds


def seed(seasons: int) -> None:
    init_db()
    registry = _build_entities()
    total_teams = sum(len(v) for v in registry.values())
    print(f"  Lige: {len(LEAGUES)}, momcadi: {total_teams}")

    fixture_id = 900_000_000
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    # Povijest završava jučer; današnje utakmice se dodaju posebno.
    history_end = today - timedelta(days=1)

    finished = 0
    upcoming: list[tuple[int, int, int, int]] = []  # (fixture_id, league_id, home, away)

    for league_id, _, _, _ in LEAGUES:
        team_ids = list(registry[league_id])
        strength = _strengths(team_ids)
        rounds = _round_robin(team_ids)

        # Svaka sezona = dva kruga (doma i u gostima), jedno kolo tjedno.
        weeks_per_season = len(rounds) * 2
        total_weeks = weeks_per_season * seasons
        start = history_end - timedelta(weeks=total_weeks)

        week = 0
        with session_scope() as db:
            for season_index in range(seasons):
                season_year = today.year - seasons + season_index
                for leg in range(2):
                    for pairs in rounds:
                        kickoff_base = start + timedelta(weeks=week)
                        week += 1
                        if kickoff_base >= history_end:
                            continue
                        for slot, (a, b) in enumerate(pairs):
                            home, away = (a, b) if leg == 0 else (b, a)
                            kickoff = kickoff_base + timedelta(days=slot % 3, hours=13 + slot % 6)
                            if kickoff >= history_end:
                                continue
                            hg, ag = _score(
                                strength[home][0], strength[away][1],
                                strength[away][0], strength[home][1],
                            )
                            fixture_id += 1
                            db.add(
                                Fixture(
                                    id=fixture_id, league_id=league_id, season=season_year,
                                    kickoff=kickoff, home_team_id=home, away_team_id=away,
                                    status="FT", finished=True,
                                    home_goals=hg, away_goals=ag,
                                    home_ht=min(hg, nprng.integers(0, hg + 1)),
                                    away_ht=min(ag, nprng.integers(0, ag + 1)),
                                )
                            )
                            finished += 1

        # ── Nadolazeće utakmice (neodigrane) ──
        # Generira se nekoliko dana unaprijed kako demo ne bi ostao prazan
        # ako se stranica otvori dan-dva nakon sijanja podataka.
        for day_offset in range(UPCOMING_DAYS):
            shuffled = list(team_ids)
            rng.shuffle(shuffled)
            with session_scope() as db:
                for i in range(0, len(shuffled) - 1, 2):
                    home, away = shuffled[i], shuffled[i + 1]
                    fixture_id += 1
                    kickoff = today + timedelta(
                        days=day_offset, hours=15 + (i // 2) % 6, minutes=(i * 15) % 60
                    )
                    db.add(
                        Fixture(
                            id=fixture_id, league_id=league_id, season=today.year,
                            kickoff=kickoff, home_team_id=home, away_team_id=away,
                            status="NS", finished=False,
                        )
                    )
                    upcoming.append((fixture_id, league_id, home, away))

    print(f"  Odigranih utakmica: {finished}")
    print(f"  Nadolazecih utakmica: {len(upcoming)} (kroz {UPCOMING_DAYS} dana)")

    _seed_odds_and_injuries(upcoming, registry)


def _seed_odds_and_injuries(
    upcoming: list[tuple[int, int, int, int]],
    registry: dict[int, dict[int, str]],
) -> None:
    """Koeficijenti se izvode iz izmišljene 'prave' vjerojatnosti uz maržu."""
    bookmakers = ["Bet365", "Pinnacle", "William Hill", "Unibet", "Betano"]
    odds_rows = 0
    injury_rows = 0

    with session_scope() as db:
        for fixture_id, league_id, home, away in upcoming:
            p_over = min(max(nprng.normal(0.53, 0.10), 0.20), 0.85)
            p_btts = min(max(nprng.normal(0.51, 0.10), 0.20), 0.85)
            margin = 1.06  # ~6 % marže kladionice

            for market, prob in (
                ("over25", p_over), ("under25", 1 - p_over),
                ("btts", p_btts), ("btts_no", 1 - p_btts),
            ):
                db.add(
                    Odd(
                        fixture_id=fixture_id,
                        market=market,
                        price=round(1.0 / (prob * margin), 2),
                        bookmaker=rng.choice(bookmakers),
                    )
                )
                odds_rows += 1

            # Izostanci — otprilike svaka druga momčad ih ima.
            for team_id in (home, away):
                if rng.random() > 0.5:
                    continue
                for n in range(rng.randint(1, 4)):
                    db.add(
                        Injury(
                            fixture_id=fixture_id,
                            team_id=team_id,
                            player_name=f"{registry[league_id][team_id].split()[0]} igrac {n + 1}",
                            reason=rng.choice(INJURY_REASONS),
                            type="Missing Fixture",
                        )
                    )
                    injury_rows += 1

    print(f"  Koeficijenata: {odds_rows}")
    print(f"  Izostanaka: {injury_rows}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo podaci za pregled sucelja")
    parser.add_argument("--seasons", type=int, default=2, help="Broj sezona povijesti")
    parser.add_argument("--reset", action="store_true", help="Obrisi postojece podatke")
    args = parser.parse_args()

    print("=" * 60)
    print(" DEMO PODACI (izmisljeni - samo za pregled sucelja)")
    print("=" * 60)

    init_db()
    if args.reset:
        _wipe()

    seed(args.seasons)

    cutoff = (datetime.now(timezone.utc) - timedelta(days=120)).date().isoformat()
    print()
    print("Gotovo. Sljedeci koraci:")
    print(f"  python -m app.ml.train --cutoff {cutoff}")
    print("  uvicorn app.main:app --reload --port 8000")
    print()


if __name__ == "__main__":
    main()
