"""Rucno pokretanje dnevnih poslova (isto sto radi rasporedivac).

    python -m scripts.daily              # utakmice za danas i iduća 2 dana + ishodi
    python -m scripts.daily --days 3
    python -m scripts.daily --date 2026-08-20
    python -m scripts.daily --results-only
"""

from __future__ import annotations

import argparse
import logging
from datetime import date

from app.db import init_db
from app.services.ingest import fetch_upcoming, fetch_upcoming_days, update_results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    parser = argparse.ArgumentParser(description="Dnevni poslovi dohvata")
    parser.add_argument("--date", dest="day", default=None, help="YYYY-MM-DD (samo taj dan)")
    parser.add_argument("--days", type=int, default=None,
                        help="bez --date: danas + N-1 dana (zadano FETCH_DAYS_AHEAD)")
    parser.add_argument("--results-only", action="store_true")
    parser.add_argument("--fixtures-only", action="store_true")
    parser.add_argument("--lookback-days", type=int, default=3)
    parser.add_argument("--region", default="supported",
                        choices=["supported", "europe", "all"],
                        help="supported = Europa + J. Amerika + medunarodna natjecanja")
    parser.add_argument("--with-odds", action="store_true",
                        help="dohvati i koeficijente (po zadanom se preskacu)")
    parser.add_argument("--no-history", action="store_true",
                        help="preskoci dohvat povijesti momcadi (nema forme!)")
    args = parser.parse_args()

    init_db()

    if not args.results_only:
        options = dict(
            region=args.region,
            with_history=not args.no_history,
            with_odds=args.with_odds,
        )
        if args.day:
            print(fetch_upcoming(date.fromisoformat(args.day), **options))
        else:
            print(fetch_upcoming_days(args.days, **options))
    if not args.fixtures_only:
        print(update_results(lookback_days=args.lookback_days))


if __name__ == "__main__":
    main()
