"""Metrike i backtest.

Kod predikcije sportskih ishoda točnost klasifikacije nije dovoljna mjera.
Zanima nas koliko su dobro kalibrirane vjerojatnosti: ako model kaže 70 %,
treba li se to doista ostvariti u otprilike 70 % slučajeva. Zato su primarne
mjere log loss i Brier score, a ne accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)

    base_rate = float(y_true.mean())
    # Referentna vrijednost: model koji uvijek vraća osnovnu stopu.
    baseline_ll = float(log_loss(y_true, np.full_like(y_prob, base_rate), labels=[0, 1]))
    model_ll = float(log_loss(y_true, y_prob, labels=[0, 1]))

    metrics = {
        "n": int(len(y_true)),
        "base_rate": base_rate,
        "log_loss": model_ll,
        "log_loss_baseline": baseline_ll,
        # Koliko je model smanjio log loss u odnosu na osnovnu stopu.
        # Vrijednost <= 0 znači da model nije naučio ništa korisno.
        "log_loss_skill": float(1.0 - model_ll / baseline_ll) if baseline_ll else 0.0,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, (y_prob >= 0.5).astype(int))),
    }
    # ROC-AUC nije definiran ako su svi ishodi isti.
    metrics["roc_auc"] = (
        float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    )
    return metrics


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predviđena vjerojatnost naspram stvarno ostvarene, po razredima.

    Ovo je tablica koja u radu najizravnije pokazuje je li model upotrebljiv:
    u dobro kalibriranom modelu su stupci `predviđeno` i `ostvareno` bliski.
    """
    df = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "p": np.asarray(y_prob, dtype=float)})
    edges = np.linspace(0.0, 1.0, bins + 1)
    df["bin"] = pd.cut(df["p"], bins=edges, include_lowest=True)
    grouped = df.groupby("bin", observed=True).agg(
        n=("y", "size"), predvideno=("p", "mean"), ostvareno=("y", "mean")
    )
    grouped["odstupanje"] = grouped["predvideno"] - grouped["ostvareno"]
    return grouped.reset_index()


def backtest(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    odds: np.ndarray,
    edge_threshold: float = 0.05,
    stake: float = 1.0,
) -> dict[str, float]:
    """Simulira ravni ulog na sve prilike gdje model vidi prednost nad tržištem.

    Oklada se igra samo ako je `p_model * koeficijent - 1 > prag`, dakle kad
    model procjenjuje pozitivnu očekivanu vrijednost. Ovo je najstroži test:
    pobijediti tržište je bitno teže nego imati dobru točnost.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    odds = np.asarray(odds, dtype=float)

    valid = np.isfinite(odds) & (odds > 1.0) & np.isfinite(y_prob)
    if not valid.any():
        return {"bets": 0, "staked": 0.0, "profit": 0.0, "roi": float("nan"), "hit_rate": float("nan")}

    edge = y_prob * odds - 1.0
    selected = valid & (edge > edge_threshold)
    n_bets = int(selected.sum())
    if n_bets == 0:
        return {"bets": 0, "staked": 0.0, "profit": 0.0, "roi": float("nan"), "hit_rate": float("nan")}

    wins = y_true[selected] == 1
    staked = stake * n_bets
    returns = np.where(wins, stake * odds[selected], 0.0).sum()
    profit = float(returns - staked)

    return {
        "bets": n_bets,
        "staked": float(staked),
        "profit": profit,
        "roi": float(profit / staked),
        "hit_rate": float(wins.mean()),
        "avg_odds": float(odds[selected].mean()),
    }


def format_report(market: str, model_name: str, metrics: dict, bt: dict | None = None) -> str:
    lines = [
        f"-- {market.upper()} | {model_name} " + "-" * 28,
        f"  uzorak (test)     {metrics['n']}",
        f"  osnovna stopa     {metrics['base_rate']:.3f}",
        f"  log loss          {metrics['log_loss']:.4f}   (referenca {metrics['log_loss_baseline']:.4f})",
        f"  skill vs osnovna  {metrics['log_loss_skill']:+.2%}",
        f"  Brier             {metrics['brier']:.4f}",
        f"  ROC-AUC           {metrics['roc_auc']:.4f}",
        f"  tocnost           {metrics['accuracy']:.4f}",
    ]
    if bt and bt.get("bets"):
        lines += [
            f"  backtest oklada   {bt['bets']} @ prosj. {bt.get('avg_odds', float('nan')):.2f}",
            f"  ROI               {bt['roi']:+.2%}  (pogodak {bt['hit_rate']:.2%})",
        ]
    return "\n".join(lines)
