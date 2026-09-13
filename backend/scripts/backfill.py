"""Jednokratno punjenje baze povijesnim utakmicama.

    python -m scripts.backfill
    python -m scripts.backfill --leagues 39 140 --seasons 2022 2023 2024

Ovo je preduvjet za treniranje: bez povijesti s upisanim ishodima model nema
oznake na kojima bi ucio.
"""

from __future__ import annotations

import argparse
import logging

from app.db import init_db
from app.services.ingest import backfill_history


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    parser = argparse.ArgumentParser(description="Povijesni backfill utakmica")
    parser.add_argument("--leagues", nargs="+", type=int, default=None)
    parser.add_argument("--seasons", nargs="+", type=int, default=None)
    args = parser.parse_args()

    init_db()
    result = backfill_history(league_ids=args.leagues, seasons=args.seasons)

    print()
    print(f"Upisano utakmica : {result['rows']}")
    print(f"API poziva       : {result['api_calls']}")
    if result["errors"]:
        print(f"Preskoceno       : {len(result['errors'])}")
        for err in result["errors"][:10]:
            print(f"  - {err}")
    print()
    print("Sljedeci korak: python -m app.ml.train")


if __name__ == "__main__":
    main()
