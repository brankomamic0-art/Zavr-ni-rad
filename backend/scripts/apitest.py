"""Provjera HTTP sloja nad sintetičkom bazom koju ostavlja `scripts.selftest`.

    python -m scripts.selftest && python -m scripts.apitest
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_TMP_DB = Path(tempfile.gettempdir()) / "selftest_football.db"
if not _TMP_DB.exists():
    print("Nema sinteticke baze — pokreni najprije: python -m scripts.selftest")
    sys.exit(1)

os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB}"
os.environ["FOOTBALL_API_KEY"] = "apitest"
os.environ["ENABLE_SCHEDULER"] = "false"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

PASS, FAIL = "  [OK]  ", "  [X]   "
failures: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    print(f"{PASS if condition else FAIL}{label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)


def _busiest_day() -> str:
    """Dan s najviše utakmica iz druge polovice raspona podataka."""
    from sqlalchemy import func, select

    from app.db import SessionLocal
    from app.models import Fixture

    with SessionLocal() as db:
        midpoint = db.scalar(
            select(func.min(Fixture.kickoff))
        ), db.scalar(select(func.max(Fixture.kickoff)))
        day_col = func.date(Fixture.kickoff)
        row = db.execute(
            select(day_col, func.count(Fixture.id))
            .where(Fixture.kickoff >= midpoint[0] + (midpoint[1] - midpoint[0]) / 2)
            .group_by(day_col)
            .order_by(func.count(Fixture.id).desc())
            .limit(1)
        ).first()
    return str(row[0])


def main() -> int:
    print("=" * 62)
    print(" PROVJERA HTTP SLOJA")
    print("=" * 62 + "\n")

    with TestClient(app) as client:
        r = client.get("/api/health")
        check(r.status_code == 200 and r.json()["status"] == "ok", "GET /api/health")

        r = client.get("/api/status")
        check(r.status_code == 200, "GET /api/status", f"HTTP {r.status_code}")
        status = r.json()
        check(status["fixtures_total"] > 0, "baza sadrzi utakmice", str(status["fixtures_total"]))
        check(
            status["model"]["markets"]["over25"]["mode"] in ("model", "heuristika"),
            "status prijavljuje nacin rada modela",
            status["model"]["markets"]["over25"]["mode"],
        )

        # Datum se uzima iz baze: bira se dan s najviše utakmica, i to iz druge
        # polovice raspona kako bi momčadi već imale formu.
        busiest = _busiest_day()
        print(f"        (testni datum iz baze: {busiest})\n")
        r = client.get("/api/matches", params={"date": busiest})
        check(r.status_code == 200, "GET /api/matches", f"HTTP {r.status_code}")
        payload = r.json()
        check(payload["count"] > 0, "vraceno je utakmica", str(payload["count"]))

        if payload["count"]:
            match = payload["matches"][0]
            check("analysis" in match and match["analysis"] is not None, "utakmica ima analizu forme")
            check(
                set(match["predictions"]) >= {"over25", "under25", "btts", "btts_no"},
                "sva cetiri trzista su prisutna",
                ", ".join(sorted(match["predictions"])),
            )
            over = match["predictions"]["over25"]["p"]
            under = match["predictions"]["under25"]["p"]
            check(
                over is not None and under is not None and abs(over + under - 1.0) < 1e-6,
                "komplementarna trzista se zbrajaju u 1",
                f"{over} + {under}",
            )
            check(
                match["analysis"]["hO"]["n"] > 0,
                "forma domacina je izracunata",
                f"n={match['analysis']['hO']['n']}, gf={match['analysis']['hO']['gf']}",
            )

            r = client.get(f"/api/fixtures/{match['id']}")
            check(r.status_code == 200, "GET /api/fixtures/{id}")

        r = client.get("/api/matches", params={"date": "1990-01-01"})
        check(
            r.status_code == 200 and r.json()["count"] == 0,
            "prazan datum vraca prazan popis, ne gresku",
        )

        r = client.get("/api/matches", params={"date": "neispravno"})
        check(r.status_code == 400, "neispravan datum vraca HTTP 400", f"HTTP {r.status_code}")

        r = client.get("/api/leagues")
        check(r.status_code == 200 and len(r.json()) > 0, "GET /api/leagues")

        r = client.post("/api/admin/backfill")
        check(r.status_code == 404, "admin rute su onemogucene bez ADMIN_TOKEN", f"HTTP {r.status_code}")

        r = client.get("/openapi.json")
        check(r.status_code == 200, "OpenAPI shema se generira")

    print()
    print("=" * 62)
    if failures:
        print(f" NEUSPJESNO: {len(failures)}")
        for f in failures:
            print(f"   - {f}")
        return 1
    print(" SVE PROVJERE PROSLE")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
