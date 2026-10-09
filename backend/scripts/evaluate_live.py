"""Prospektivna evaluacija iz naredbenog retka.

    python -m scripts.evaluate_live              # sve spremljene predikcije
    python -m scripts.evaluate_live --days 30    # samo zadnjih 30 dana
    python -m scripts.evaluate_live --snapshot   # najprije zamrzni nadolazeće

Ispis je u istom obliku kao izvještaj `app.ml.train`, pa se retrospektivni i
prospektivni rezultat mogu staviti jedan uz drugi.
"""

from __future__ import annotations

import argparse
import json
import sys

from app.db import init_db, session_scope
from app.ml.evaluate import format_report
from app.ml.prospective import evaluate_prospective
from app.services.predictions import model_info, snapshot_upcoming


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Prospektivna evaluacija predikcija")
    parser.add_argument("--days", type=int, default=None, help="samo zadnjih N dana")
    parser.add_argument("--snapshot", action="store_true",
                        help="prije evaluacije spremi predikcije za nadolazece utakmice")
    parser.add_argument("--json", action="store_true", help="ispis kao JSON")
    args = parser.parse_args()

    init_db()
    with session_scope() as db:
        if args.snapshot:
            print("Zamrznuto:", snapshot_upcoming(db))
        report = evaluate_prospective(db, args.days)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        return

    print(
        f"Odigrano s predikcijom: {report['resolved']} | "
        f"ceka ishod: {report['pending']}"
    )
    if not report["markets"]:
        print("Jos nema odigranih utakmica sa spremljenom predikcijom.")
        return
    print(f"Raspon: {report['first_kickoff'][:10]} - {report['last_kickoff'][:10]}\n")

    retro = model_info()["markets"]
    for market, entry in report["markets"].items():
        metrics = {k: float("nan") if v is None else v for k, v in entry["metrics"].items()}
        print(format_report(market, "prospektivno", metrics, entry["backtest"]))
        if entry.get("warning"):
            print(f"  ! {entry['warning']}")
        r = retro.get(market) or {}
        if r.get("available"):
            print(
                f"  retrospektivno    log loss {r['log_loss']:.4f} | skill {r['skill']:+.2%} | "
                f"ROC-AUC {r['roc_auc']:.4f}"
            )
        for source, m in entry["by_source"].items():
            print(f"    {source:<16} n={m['n']:<5} skill {m['log_loss_skill']:+.2%}")
        print("  kalibracija (predvideno -> ostvareno):")
        for row in entry["calibration"]:
            print(f"    {row['bin']:<14} n={row['n']:<5} {row['predicted']:.3f} -> {row['observed']:.3f}")
        print()


if __name__ == "__main__":
    main()
