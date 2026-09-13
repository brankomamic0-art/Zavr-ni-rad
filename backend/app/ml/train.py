"""Treniranje i evaluacija modela.

Pokretanje:
    python -m app.ml.train                      # svi modeli, oba tržišta
    python -m app.ml.train --models logreg      # samo jedan model
    python -m app.ml.train --with-odds          # uključi koeficijente u značajke
    python -m app.ml.train --cutoff 2024-07-01  # granica kronološke podjele

Rezultat: `backend/models/<trziste>__<model>.joblib` + `metrics.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from app.config import settings
from app.ml.dataset import build_training_set, chronological_split
from app.ml.evaluate import backtest, calibration_table, classification_metrics, format_report
from app.ml.features import MODEL_FEATURES, ODDS_FEATURES, SHOT_MODEL_FEATURES, TARGETS
from app.ml.registry import available_models, build_model

log = logging.getLogger(__name__)

MODEL_DIR = Path(settings.model_dir)


def _force_utf8_stdout() -> None:
    """Windows konzola po zadanom koristi cp1252 i rusi se na dijakritiku."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


ODDS_COLUMN = {"over25": "over25", "btts": "btts"}

# Ispod ovog broja redaka za treniranje koristi se Plattova umjesto
# izotonicne kalibracije (vidi obrazlozenje u `train_one`).
ISOTONIC_MIN_ROWS = 5000


# Sufiks kojim se razlikuju dvije varijante istog algoritma. Varijanta s
# udarcima nije bolja svugdje — samo ondje gdje statistika postoji — pa se
# obje treniraju i posluzuju, a izbor se radi po utakmici.
SHOT_SUFFIX = "_shots"


def _artifact_path(market: str, model_name: str, with_shots: bool = False) -> Path:
    suffix = SHOT_SUFFIX if with_shots else ""
    return MODEL_DIR / f"{market}__{model_name}{suffix}.joblib"


def train_one(
    df: pd.DataFrame,
    market: str,
    model_name: str,
    cutoff: str,
    feature_cols: list[str],
    calibrate: bool = True,
    with_shots: bool = False,
) -> dict:
    """Trenira jedan model za jedno tržište i vraća izvještaj o evaluaciji."""
    target_col = TARGETS[market]
    data = df.dropna(subset=[target_col]).copy()
    if with_shots:
        # Varijanta s udarcima uci ISKLJUCIVO na utakmicama gdje statistika
        # stvarno postoji. Imputacija medijanom na ostalima mjerljivo steti
        # (skill 2,71 % -> 2,41 %), pa se te utakmice ne uce i ne posluzuju.
        data = data[data["h_shots_on_f"].notna() & data["a_shots_on_f"].notna()]
    train_df, test_df = chronological_split(data, cutoff)

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Kronoloska podjela na {cutoff} daje prazan skup "
            f"(train={len(train_df)}, test={len(test_df)}). Provjeri raspon podataka."
        )

    x_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(int)
    x_test = test_df[feature_cols]
    y_test = test_df[target_col].astype(int)

    model = build_model(model_name)

    calibration_method = None
    if calibrate and model_name != "prior":
        # Kalibracija se uči unakrsnom provjerom UNUTAR skupa za treniranje,
        # nikad na testnom skupu — inače bi testni skup prestao biti neviđen.
        #
        # Izbor metode ovisi o količini podataka: izotonična regresija je
        # neparametarska i na malim uzorcima se prenauči, dok je Plattova
        # (sigmoidna) metoda s dva parametra znatno stabilnija.
        calibration_method = "isotonic" if len(train_df) >= ISOTONIC_MIN_ROWS else "sigmoid"
        model = CalibratedClassifierCV(model, method=calibration_method, cv=3)

    model.fit(x_train, y_train)
    prob = model.predict_proba(x_test)[:, 1]

    metrics = classification_metrics(y_test.to_numpy(), prob)
    metrics["train_n"] = int(len(train_df))
    metrics["cutoff"] = cutoff
    metrics["calibration"] = calibration_method or "bez kalibracije"

    bt = None
    odds_col = f"odds_{ODDS_COLUMN[market]}"
    if odds_col in test_df.columns:
        bt = backtest(y_test.to_numpy(), prob, test_df[odds_col].to_numpy())

    calib = calibration_table(y_test.to_numpy(), prob)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "market": market,
            "model_name": model_name + (SHOT_SUFFIX if with_shots else ""),
            "with_shots": with_shots,
            "features": feature_cols,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "cutoff": cutoff,
            "calibration": calibration_method,
            "metrics": metrics,
        },
        _artifact_path(market, model_name, with_shots),
    )

    print(format_report(market, model_name + (SHOT_SUFFIX if with_shots else ""), metrics, bt))
    print(f"  kalibracija       {metrics['calibration']}")
    print(calib.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()

    return {"metrics": metrics, "backtest": bt, "calibration": calib.to_dict(orient="records")}


def main() -> None:
    _force_utf8_stdout()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Treniranje prediktivnih modela")
    parser.add_argument("--models", nargs="+", default=available_models(),
                        help=f"Modeli za treniranje. Dostupni: {', '.join(available_models())}")
    parser.add_argument("--markets", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    parser.add_argument("--cutoff", default=settings.train_cutoff,
                        help="Datum kronoloske podjele (YYYY-MM-DD)")
    parser.add_argument("--with-odds", action="store_true",
                        help="Ukljuci implicirane vjerojatnosti iz koeficijenata u znacajke")
    parser.add_argument("--min-history", type=int, default=5)
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--no-shots", action="store_true",
                        help="Preskoci varijantu modela sa znacajkama iz udaraca")
    args = parser.parse_args()

    print("Ucitavam skup za treniranje iz baze...")
    df = build_training_set(min_history=args.min_history, with_odds=True)
    if df.empty:
        print(
            "Baza je prazna. Pokreni najprije:\n"
            "    python -m scripts.backfill"
        )
        return

    # Koeficijenti se dohvaćaju uvijek (radi backtesta), ali u značajke ulaze
    # samo ako je to izričito traženo — usporedba "sa" i "bez" je dio rada.
    feature_cols = list(MODEL_FEATURES)
    if args.with_odds:
        feature_cols += ODDS_FEATURES

    for market in ("over25", "btts"):
        src = f"imp_{market}"
        if src in df.columns:
            df[f"odds_{market}"] = np.where(
                df[src].notna() & (df[src] > 0), 1.0 / df[src], np.nan
            )

    print(
        f"Redaka: {len(df)} | raspon: {df['kickoff'].min().date()} - {df['kickoff'].max().date()} | "
        f"znacajki: {len(feature_cols)}\n"
    )

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(df)),
        "cutoff": args.cutoff,
        "with_odds_features": bool(args.with_odds),
        "features": feature_cols,
        "results": {},
    }

    # Dvije varijante: osnovna (sve lige) i varijanta s udarcima (samo lige
    # gdje API nudi statistiku). Posluzivanje bira po utakmici.
    variants = [(False, feature_cols)]
    if not args.no_shots:
        variants.append((True, feature_cols + SHOT_MODEL_FEATURES[len(MODEL_FEATURES):]))

    for market in args.markets:
        report["results"][market] = {}
        for model_name in args.models:
            for with_shots, cols in variants:
                key = model_name + (SHOT_SUFFIX if with_shots else "")
                try:
                    report["results"][market][key] = train_one(
                        df,
                        market=market,
                        model_name=model_name,
                        cutoff=args.cutoff,
                        feature_cols=cols,
                        calibrate=not args.no_calibration,
                        with_shots=with_shots,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.error("Model %s / %s nije istreniran: %s", market, key, exc)
                    report["results"][market][key] = {"error": str(exc)}

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / "metrics.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Izvjestaj spremljen: {out}")
    print(
        f"\nAktivni model za posluzivanje je '{settings.active_model}' "
        "(promjena preko ACTIVE_MODEL u .env)."
    )


if __name__ == "__main__":
    main()
