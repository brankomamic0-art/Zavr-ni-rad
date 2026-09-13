"""Provjera ispravnosti cjevovoda na sintetičkim podacima.

Ne dira API ni pravu bazu — generira izmišljeno prvenstvo, puni privremenu
bazu i provjerava tri stvari:

  1. Značajke se uopće izgrade i imaju očekivani oblik.
  2. NEMA CURENJA PODATAKA — forma za utakmicu ne smije sadržavati ništa iz te
     ili bilo koje kasnije utakmice. Provjerava se ručnim preračunom.
  3. Treniranje i evaluacija prolaze do kraja i model nadmašuje osnovnu stopu.

Pokretanje:
    python -m scripts.selftest
"""

from __future__ import annotations

import random
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Privremena baza mora biti postavljena PRIJE uvoza aplikacijskih modula,
# jer se engine gradi pri uvozu.
_TMP_DB = Path(tempfile.gettempdir()) / "selftest_football.db"
_TMP_DB.unlink(missing_ok=True)
import os  # noqa: E402

os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB}"
os.environ["FOOTBALL_API_KEY"] = "selftest"
os.environ["ENABLE_SCHEDULER"] = "false"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import init_db, session_scope  # noqa: E402
from app.ml.dataset import build_training_set, chronological_split  # noqa: E402
from app.ml.evaluate import classification_metrics  # noqa: E402
from app.ml.features import MODEL_FEATURES, build_feature_matrix  # noqa: E402
from app.ml.registry import build_model  # noqa: E402
from app.models import Fixture, League, Team  # noqa: E402

N_TEAMS = 20
N_SEASONS = 6
PASS, FAIL = "  [OK]  ", "  [X]   "
failures: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    print(f"{PASS if condition else FAIL}{label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)


# ═══════════════════════════════════════════════════════════════
# Generiranje sintetičkog prvenstva
# ═══════════════════════════════════════════════════════════════

def generate() -> None:
    rng = random.Random(42)
    init_db()

    # Svaka momčad ima skrivenu jakost napada i obrane; golovi se izvlače iz
    # Poissonove razdiobe. Ako model radi, mora naučiti nazrijeti te jakosti.
    attack = {t: rng.uniform(0.7, 2.1) for t in range(1, N_TEAMS + 1)}
    defence = {t: rng.uniform(0.6, 1.6) for t in range(1, N_TEAMS + 1)}

    with session_scope() as db:
        db.add(League(id=1, name="Testna liga", country="Croatia", type="League", tier=1))
        for tid in range(1, N_TEAMS + 1):
            db.add(Team(id=tid, name=f"Klub {tid}", country="Croatia"))

    fixture_id = 1
    start = datetime(2019, 8, 1, 18, 0, tzinfo=timezone.utc)

    with session_scope() as db:
        for season in range(N_SEASONS):
            season_start = start + timedelta(days=365 * season)
            teams = list(range(1, N_TEAMS + 1))
            round_no = 0
            for home in teams:
                for away in teams:
                    if home == away:
                        continue
                    kickoff = season_start + timedelta(days=round_no // 10 * 7, hours=round_no % 10)
                    round_no += 1

                    lam_home = attack[home] * defence[away] * 1.25  # prednost domaćeg terena
                    lam_away = attack[away] * defence[home]
                    hg = int(np.random.default_rng(fixture_id).poisson(lam_home))
                    ag = int(np.random.default_rng(fixture_id + 100000).poisson(lam_away))

                    db.add(
                        Fixture(
                            id=fixture_id,
                            league_id=1,
                            season=2019 + season,
                            kickoff=kickoff,
                            home_team_id=home,
                            away_team_id=away,
                            status="FT",
                            finished=True,
                            home_goals=hg,
                            away_goals=ag,
                        )
                    )
                    fixture_id += 1
    print(f"        Generirano {fixture_id - 1} utakmica, {N_TEAMS} momcadi, {N_SEASONS} sezona\n")


# ═══════════════════════════════════════════════════════════════
# Provjere
# ═══════════════════════════════════════════════════════════════

def test_features() -> pd.DataFrame:
    print("1) Izgradnja znacajki")
    df = build_training_set(min_history=5)
    check(not df.empty, "matrica znacajki nije prazna", f"{len(df)} redaka")
    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    check(not missing, "sve ocekivane znacajke postoje", f"{len(MODEL_FEATURES)} znacajki")
    if missing:
        print(f"        nedostaje: {missing}")

    nan_share = df[MODEL_FEATURES].isna().mean().mean()
    check(nan_share < 0.35, "udio nedostajucih vrijednosti je prihvatljiv", f"{nan_share:.1%}")
    check(
        df["y_over25"].between(0, 1).all() and df["y_btts"].between(0, 1).all(),
        "ciljne varijable su binarne",
        f"over25={df['y_over25'].mean():.3f}, btts={df['y_btts'].mean():.3f}",
    )
    print()
    return df


def test_no_leakage(df: pd.DataFrame) -> None:
    """Ključna provjera rada: rucni preracun forme za nasumicne utakmice."""
    print("2) Odsutnost curenja podataka (point-in-time ispravnost)")

    from app.ml.dataset import load_fixtures

    fixtures = load_fixtures(finished_only=True)
    rng = random.Random(7)
    sample = df.sample(min(40, len(df)), random_state=1)

    mismatches = 0
    self_inclusions = 0

    for _, row in sample.iterrows():
        fid = row["fixture_id"]
        team = row["home_team_id"]
        kickoff = row["kickoff"]

        # Ručno: zadnjih N utakmica te momčadi STROGO prije ovog termina.
        # Veličina prozora se čita iz postavki, a ne upisuje ovdje — inače test
        # pukne svaki put kad se prozor promijeni, i to lažno prijavi curenje
        # podataka umjesto da provjeri ono zbog čega postoji.
        window = settings.form_window_overall
        prior = fixtures[
            ((fixtures["home_team_id"] == team) | (fixtures["away_team_id"] == team))
            & (fixtures["kickoff"] < kickoff)
        ].sort_values("kickoff").tail(window)

        if prior.empty:
            continue

        # Sama utakmica ne smije biti u prozoru.
        if fid in set(prior["fixture_id"]):
            self_inclusions += 1

        gf = np.where(
            prior["home_team_id"] == team, prior["home_goals"], prior["away_goals"]
        ).astype(float)
        expected_gf = gf.mean()
        actual_gf = row["h_ov_gf"]

        if pd.notna(actual_gf) and abs(expected_gf - actual_gf) > 1e-6:
            mismatches += 1

        expected_n = len(prior)
        if pd.notna(row["h_ov_n"]) and int(row["h_ov_n"]) != expected_n:
            mismatches += 1

    check(self_inclusions == 0, "vlastita utakmica nije u prozoru forme")
    check(mismatches == 0, "rucni preracun forme se podudara", f"{len(sample)} uzoraka")

    # Dodatna provjera: buduće utakmice ne smiju utjecati na prošle značajke.
    fixtures_trunc = fixtures[fixtures["kickoff"] < fixtures["kickoff"].quantile(0.6)]
    full = build_feature_matrix(fixtures, targets=fixtures_trunc)
    partial = build_feature_matrix(fixtures_trunc, targets=fixtures_trunc)
    cols = ["h_ov_gf", "a_ov_gf", "h_vn_over25"]
    merged = full[["fixture_id", *cols]].merge(
        partial[["fixture_id", *cols]], on="fixture_id", suffixes=("_full", "_part")
    )
    identical = all(
        np.allclose(
            merged[f"{c}_full"].fillna(-999), merged[f"{c}_part"].fillna(-999), atol=1e-9
        )
        for c in cols
    )
    check(
        identical,
        "znacajke se ne mijenjaju kad se dodaju kasnije utakmice",
        f"{len(merged)} redaka usporedeno",
    )
    print()


def test_training(df: pd.DataFrame) -> None:
    print("3) Treniranje i evaluacija")
    cutoff = df["kickoff"].quantile(0.75)
    train, test = chronological_split(df, cutoff)
    check(len(train) > 0 and len(test) > 0, "kronoloska podjela daje oba skupa",
          f"train={len(train)}, test={len(test)}")
    check(
        train["kickoff"].max() < test["kickoff"].min(),
        "skup za testiranje je u cijelosti nakon skupa za treniranje",
    )

    # Ovo je provjera cjevovoda, ne natjecanje algoritama. Sintetički skup ima
    # svega ~1600 redaka za treniranje, pa je normalno da se fleksibilniji
    # modeli prenauče — odabir algoritma se radi na stvarnim podacima.
    for market, target in (("over25", "y_over25"), ("btts", "y_btts")):
        skills = {}
        for name in ("prior", "logreg", "hist_gb"):
            model = build_model(name)
            model.fit(train[MODEL_FEATURES], train[target].astype(int))
            prob = model.predict_proba(test[MODEL_FEATURES])[:, 1]

            check(
                np.all(np.isfinite(prob)) and prob.min() >= 0.0 and prob.max() <= 1.0,
                f"{market:7s} {name:8s} daje valjane vjerojatnosti",
            )
            metrics = classification_metrics(test[target].to_numpy(), prob)
            skills[name] = metrics["log_loss_skill"]
            print(
                f"        {market:7s} {name:8s} log_loss={metrics['log_loss']:.4f} "
                f"skill={metrics['log_loss_skill']:+.2%}"
            )

        learners = {k: v for k, v in skills.items() if k != "prior"}
        best = max(learners, key=learners.get)
        check(
            learners[best] > 0,
            f"{market:7s} barem jedan model nadmasuje osnovnu stopu",
            f"najbolji: {best} {learners[best]:+.2%}",
        )
    print()


def test_competition_classification() -> None:
    """Razvrstavanje natjecanja — cisto tekstualna heuristika, bez baze.

    Lazni pozitiv ovdje izbacuje prave utakmice iz prikaza, pa su nazivi koji
    su vec jednom zavarali klasifikator zapisani kao trajna provjera.
    """
    print("4) Razvrstavanje natjecanja")
    from app.services.quality import classify_competition

    cases = [
        # Nazivi na kojima se klasifikator vec jednom prevario:
        ("Serie B", "Londrina", "Atletico Goianiense", "senior"),          # rang lige, ne pricuve
        ("Liga Profesional", "Boca Juniors", "River Plate", "senior"),     # "Juniors" u imenu kluba
        ("Primera A", "Junior FC", "Millonarios", "senior"),
        ("CONCACAF Caribbean Club Championship", "Mount Pleasant Academy", "Cibao", "senior"),
        ("Primera B Nacional", "Atlanta", "Almagro", "senior"),
        ("2. Bundesliga", "Hertha BSC", "Schalke 04", "senior"),
        ("CONMEBOL Libertadores", "U. Catolica", "Estudiantes L.P.", "senior"),
        ("Premier League", "Chelsea", "Arsenal", "senior"),
        # Moraju biti prepoznati:
        ("U19 Bundesliga", "Dynamo Dresden U19", "Eintracht U19", "youth"),
        ("Primavera 1", "Inter Primavera", "Milan Primavera", "youth"),
        ("Cup", "Keila", "Flora III", "reserve"),
        ("Friendlies Clubs", "Elche II", "FC Cartagena", "reserve"),
        ("Regionalliga", "Bayern Munchen II", "Wurzburger", "reserve"),
        ("Premier Liga", "Tobol 2", "Khan Tengri", "reserve"),          # brojcana pricuva
        ("Primera B", "Quilmes 2", "Ferro 2", "reserve"),
        ("Reserve Liga", "Boca Juniors Res.", "River Res.", "reserve"), # "Res." nastavak
        ("Toppserien", "Valerenga W", "Honefoss W", "women"),
        ("Kvindeliga", "Brondby W", "Koge W", "women"),
        ("DFB Junioren Pokal", "Team A", "Team B", "youth"),
        # Seniorski klubovi s brojkom ili "Juniors" u imenu ne smiju propasti.
        # "Juniors" se namjerno NE gleda u nazivu momcadi: u Argentini i
        # Skotskoj to su seniorski klubovi (Boca Juniors, Cumnock Juniors).
        ("Bundesliga", "Schalke 04", "Bayer 04 Leverkusen", "senior"),
        ("Primera Division", "Chacarita Juniors", "Rampla Juniors", "senior"),
        ("Lowland League", "Cumnock Juniors", "Tranent Juniors", "senior"),
    ]

    wrong = []
    for league, home, away, expected in cases:
        got = classify_competition(league, home, away)
        if got != expected:
            wrong.append(f"{league} | {home}-{away}: {got}, ocekivano {expected}")

    check(not wrong, "svi nazivi razvrstani tocno", f"{len(cases)} slucajeva")
    for item in wrong:
        print(f"        {item}")
    print()


def main() -> int:
    print("=" * 62)
    print(" SAMOPROVJERA CJEVOVODA (sinteticki podaci)")
    print("=" * 62 + "\n")

    generate()
    df = test_features()
    if df.empty:
        print("Matrica je prazna — daljnje provjere se preskacu.")
        return 1
    test_no_leakage(df)
    test_training(df)
    test_competition_classification()

    print("=" * 62)
    if failures:
        print(f" NEUSPJESNO: {len(failures)} provjera")
        for f in failures:
            print(f"   - {f}")
        return 1
    print(" SVE PROVJERE PROSLE")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
