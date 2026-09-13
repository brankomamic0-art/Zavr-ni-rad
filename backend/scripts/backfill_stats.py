"""Povlacenje statistike utakmica (udarci, kutovi, posjed) i 90-minutnog rezultata.

Kljucna stvar zbog koje je ovo uopce izvedivo: endpoint `/fixtures` uz parametar
`ids` prima do 20 ID-eva po pozivu i u istom odgovoru vraca `statistics`,
`events`, `lineups` i `players`. Dohvat statistike za cijelu povijest zato kosta
oko 2 100 poziva umjesto 42 000 — dakle jedan popodnevni posao unutar dnevne
kvote, a ne 500 dana.

Usput se sprema i `score.fulltime` (rezultat nakon 90 minuta). Kod utakmica sa
statusom AET polje `goals` sadrzi i golove iz produzetaka, a kladionice trziste
Over/Under 2.5 namiruju na regularnih 90 minuta — bez ovoga su takve utakmice
krivo oznacene.

Pokretanje:
    python -m scripts.backfill_stats                 # sve lige iz .env
    python -m scripts.backfill_stats --limit 200     # samo prvih N serija
    python -m scripts.backfill_stats --leagues 39 140
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.services.api_football import ApiFootballClient  # noqa: E402

log = logging.getLogger("backfill_stats")

BATCH = 20  # maksimum koji API dopusta za parametar `ids`

# Nazivi iz odgovora API-ja -> stupci u tablici `fixture_stats`.
STAT_MAP = {
    "Shots on Goal": "shots_on",
    "Total Shots": "shots_total",
    "Shots insidebox": "shots_box",
    "Corner Kicks": "corners",
    "Ball Possession": "possession",
    "Goalkeeper Saves": "saves",
}


def _num(value) -> float | None:
    """API vraca brojeve, None, ili postotak kao tekst ('53%')."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().rstrip("%")
    try:
        return float(text)
    except ValueError:
        return None


def _db_path() -> str:
    url = settings.sqlalchemy_url
    if not url.startswith("sqlite"):
        raise SystemExit("Ova skripta radi nad SQLite bazom (DATABASE_URL je prazan).")
    return url.split("sqlite:///")[1]


def pending_fixtures(conn: sqlite3.Connection, league_ids: list[int]) -> list[int]:
    """Odigrane utakmice zadanih liga koje jos nemaju statistiku."""
    ids = ",".join(str(int(x)) for x in league_ids)
    rows = conn.execute(
        f"""
        SELECT f.id
        FROM fixtures f
        LEFT JOIN fixture_stats s ON s.fixture_id = f.id
        WHERE f.finished = 1
          AND f.home_goals IS NOT NULL
          AND f.league_id IN ({ids})
          AND s.fixture_id IS NULL
        ORDER BY f.kickoff DESC
        """
    ).fetchall()
    return [r[0] for r in rows]


def store_batch(conn: sqlite3.Connection, items: list[dict]) -> tuple[int, int]:
    """Upisuje statistiku i 90-minutni rezultat. Vraca (redaka statistike, utakmica sa 90')."""
    stat_rows: list[tuple] = []
    ft_rows: list[tuple] = []

    for item in items:
        fixture_id = (item.get("fixture") or {}).get("id")
        if not fixture_id:
            continue

        fulltime = (item.get("score") or {}).get("fulltime") or {}
        if fulltime.get("home") is not None and fulltime.get("away") is not None:
            ft_rows.append((int(fulltime["home"]), int(fulltime["away"]), fixture_id))

        for block in item.get("statistics") or []:
            team_id = (block.get("team") or {}).get("id")
            if not team_id:
                continue
            values = {
                STAT_MAP[s["type"]]: _num(s.get("value"))
                for s in block.get("statistics") or []
                if s.get("type") in STAT_MAP
            }
            if not values:
                continue
            stat_rows.append(
                (
                    fixture_id,
                    team_id,
                    values.get("shots_on"),
                    values.get("shots_total"),
                    values.get("shots_box"),
                    values.get("corners"),
                    values.get("possession"),
                    values.get("saves"),
                )
            )

    if stat_rows:
        conn.executemany(
            """INSERT OR IGNORE INTO fixture_stats
               (fixture_id, team_id, shots_on, shots_total, shots_box, corners, possession, saves)
               VALUES (?,?,?,?,?,?,?,?)""",
            stat_rows,
        )
    if ft_rows:
        conn.executemany(
            "UPDATE fixtures SET home_ft = ?, away_ft = ? WHERE id = ?", ft_rows
        )
    conn.commit()
    return len(stat_rows), len(ft_rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Backfill statistike utakmica")
    parser.add_argument("--leagues", nargs="+", type=int, default=settings.league_ids)
    parser.add_argument("--limit", type=int, default=0, help="Najvise N serija (0 = sve)")
    parser.add_argument("--delay", type=float, default=0.25, help="Pauza izmedu poziva (s)")
    args = parser.parse_args()

    conn = sqlite3.connect(_db_path())
    todo = pending_fixtures(conn, args.leagues)
    batches = [todo[i : i + BATCH] for i in range(0, len(todo), BATCH)]
    if args.limit:
        batches = batches[: args.limit]

    log.info(
        "Za dohvat: %s utakmica u %s serija (~%s poziva)", len(todo), len(batches), len(batches)
    )
    if not batches:
        log.info("Nema nista za dohvatiti.")
        return

    total_stats = total_ft = 0
    started = time.time()
    with ApiFootballClient() as client:
        for i, batch in enumerate(batches, start=1):
            try:
                items = client.get("/fixtures", ids="-".join(str(x) for x in batch))
            except Exception as exc:  # noqa: BLE001
                log.warning("Serija %s preskocena: %s", i, exc)
                continue
            s, f = store_batch(conn, items)
            total_stats += s
            total_ft += f
            if i % 25 == 0 or i == len(batches):
                elapsed = time.time() - started
                pace = i / elapsed if elapsed else 0
                left = (len(batches) - i) / pace if pace else 0
                log.info(
                    "  %s/%s serija | %s redaka statistike | %s rezultata 90' | preostalo ~%.0f min",
                    i, len(batches), total_stats, total_ft, left / 60,
                )
            time.sleep(args.delay)

    log.info("Gotovo: %s redaka statistike, %s upisanih 90-minutnih rezultata", total_stats, total_ft)


if __name__ == "__main__":
    main()
