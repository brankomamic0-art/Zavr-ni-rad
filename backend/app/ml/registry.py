"""Registar modela.

Algoritam još nije odabran — zato je ovdje sve svedeno na jedan rječnik
tvornica. Dodavanje novog kandidata znači jednu funkciju i jedan unos, bez
diranja ostatka sustava (treniranje, evaluacija i posluživanje rade nad
`sklearn`-kompatibilnim sučeljem `fit` / `predict_proba`).

Trenutno registrirano:
    prior    — uvijek predviđa osnovnu stopu; donja granica koju svaki
               ozbiljan model mora nadmašiti
    logreg   — logistička regresija, interpretabilna referentna vrijednost
    hist_gb  — gradient boosting nad stablima (scikit-learn izvedba)

Pripremljeno, uključuje se instalacijom biblioteke:
    xgboost  — vidi `requirements.txt`
"""

from __future__ import annotations

from collections.abc import Callable

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ModelFactory = Callable[[], Pipeline]


def _prior() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("clf", DummyClassifier(strategy="prior")),
        ]
    )


def _logreg() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")),
        ]
    )


def _hist_gb() -> Pipeline:
    # HistGradientBoosting sam obrađuje nedostajuće vrijednosti, pa imputacija
    # nije potrebna; ostavljena je izvan cjevovoda namjerno.
    return Pipeline(
        [
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=400,
                    learning_rate=0.05,
                    max_depth=None,
                    max_leaf_nodes=31,
                    min_samples_leaf=40,
                    l2_regularization=1.0,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=42,
                ),
            )
        ]
    )


REGISTRY: dict[str, ModelFactory] = {
    "prior": _prior,
    "logreg": _logreg,
    "hist_gb": _hist_gb,
}


def _register_optional_models() -> None:
    """Modeli koji ovise o neobaveznim bibliotekama."""
    try:
        from xgboost import XGBClassifier  # noqa: PLC0415
    except ImportError:
        return

    def _xgb() -> Pipeline:
        return Pipeline(
            [
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=600,
                        learning_rate=0.03,
                        max_depth=4,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        min_child_weight=20,
                        reg_lambda=2.0,
                        eval_metric="logloss",
                        tree_method="hist",
                        random_state=42,
                    ),
                )
            ]
        )

    REGISTRY["xgb"] = _xgb


_register_optional_models()


def build_model(name: str) -> Pipeline:
    if name not in REGISTRY:
        raise KeyError(
            f"Nepoznat model '{name}'. Dostupni: {', '.join(sorted(REGISTRY))}"
        )
    return REGISTRY[name]()


def available_models() -> list[str]:
    return sorted(REGISTRY)
