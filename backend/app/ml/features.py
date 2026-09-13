"""Inženjering značajki s ispravnošću u vremenu (point-in-time).

Najvažnije pravilo cijelog rada je ovdje: značajke jedne utakmice smiju biti
izračunate ISKLJUČIVO iz utakmica koje su odigrane STROGO PRIJE njezina
početka. Ako se u prosjek forme uvuče i sama utakmica koja se predviđa (ili
bilo koja kasnija), model "vidi budućnost" i rezultati evaluacije su lažno
dobri — klasična pogreška curenja podataka (data leakage).

Tehnički se to ovdje postiže s `pandas.merge_asof(direction="backward",
allow_exact_matches=False)`: za svaku utakmicu se dohvaća stanje forme momčadi
nakon njezine posljednje ranije odigrane utakmice.

Isti se kod koristi i pri treniranju i pri posluživanju predikcija, čime se
izbjegava neslaganje značajki između treniranja i produkcije (train/serve skew).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import settings

# ── Nazivi stupaca ──

BASE_STATS = ["gf", "ga", "tot", "over25", "btts", "cs", "fts", "pts"]

# Značajke koje ulaze u model (bez koeficijenata).
FORM_FEATURES = [
    f"{side}_{scope}_{stat}"
    for side in ("h", "a")
    for scope in ("ov", "vn")
    for stat in ("gf", "ga", "tot", "over25", "btts", "cs", "fts", "pts", "n")
]

DERIVED_FEATURES = [
    "exp_goals_home",
    "exp_goals_away",
    "exp_total",
    "attack_diff",
    "form_diff",
    "league_tier",
    "h_rest_days",
    "a_rest_days",
    "h_injuries",
    "a_injuries",
]

# Kratki prozor: trenutna forma. Mjereno je da dugi prozor nosi vise signala od
# kratkog (20 utakmica > 5), ali da kratki uz njega jos malo doprinosi — zato
# oba, a ne samo jedan.
RECENT_STATS = ["gf", "ga", "tot", "pts", "over25", "btts"]
RECENT_FEATURES = [f"{side}_re_{stat}" for side in ("h", "a") for stat in RECENT_STATS]

# Statistika udaraca iz `/fixtures?ids=` (`_f` = vlastiti, `_a` = protivnikovi).
# Gol je rijedak dogadaj; udarci mjere istu namjeru s oko deset puta vise
# uzoraka po utakmici, pa je procjena napadacke snage stabilnija.
SHOT_STATS = ["shots_on", "shots_total", "shots_box", "corners"]
SHOT_COLS = [f"{stat}_{side}" for stat in SHOT_STATS for side in ("f", "a")]
SHOT_FEATURES = [f"{side}_{col}" for side in ("h", "a") for col in SHOT_COLS]

ODDS_FEATURES = ["imp_over25", "imp_btts", "odds_margin"]

# Brojači veličine prozora (`*_n`) ostaju u matrici jer ih trebaju prikaz forme
# i filter dovoljnosti podataka, ali NE ulaze u model. Oni ne govore ništa o
# nogometu — mjere koliko je podataka prikupljeno — pa model preko njih uči
# osobitosti prikupljanja umjesto igre. Izmjereno: bez njih je Over 2.5 neznatno
# bolji (AUC 0,599 -> 0,601), a težine ostalih značajki postaju tumačive.
WINDOW_SIZE_FEATURES = [c for c in FORM_FEATURES if c.endswith("_n")]

MODEL_FEATURES = (
    [c for c in FORM_FEATURES if not c.endswith("_n")] + DERIVED_FEATURES + RECENT_FEATURES
)

# Druga varijanta modela: ista osnova plus udarci. Postoji zasebno jer statistika
# udaraca ne postoji za sve lige — na utakmicama bez nje imputacija medijanom
# unosi vise stete nego koristi (izmjereno: skill 2,71 % -> 2,41 % na punom
# skupu). Zato se ta varijanta poslužuje samo ondje gdje podaci stvarno postoje.
SHOT_MODEL_FEATURES = MODEL_FEATURES + SHOT_FEATURES

TARGETS = {
    "over25": "y_over25",
    "btts": "y_btts",
}


# ═══════════════════════════════════════════════════════════════
# 1. Dugački oblik: jedan redak po (momčad, utakmica)
# ═══════════════════════════════════════════════════════════════

def to_long_frame(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Pretvara tablicu utakmica u perspektivu pojedine momčadi.

    Svaka utakmica daje dva retka — jedan iz kuta domaćina, jedan iz kuta gosta.
    Time se forma računa jednom po momčadi, neovisno o tome je li ta momčad u
    ciljanoj utakmici domaćin ili gost.
    """
    if fixtures.empty:
        return pd.DataFrame(columns=["fixture_id", "kickoff", "team_id", "is_home", *BASE_STATS])

    home = pd.DataFrame(
        {
            "fixture_id": fixtures["fixture_id"],
            "kickoff": fixtures["kickoff"],
            "team_id": fixtures["home_team_id"],
            "opponent_id": fixtures["away_team_id"],
            "is_home": True,
            "gf": fixtures["home_goals"],
            "ga": fixtures["away_goals"],
        }
    )
    away = pd.DataFrame(
        {
            "fixture_id": fixtures["fixture_id"],
            "kickoff": fixtures["kickoff"],
            "team_id": fixtures["away_team_id"],
            "opponent_id": fixtures["home_team_id"],
            "is_home": False,
            "gf": fixtures["away_goals"],
            "ga": fixtures["home_goals"],
        }
    )
    long = pd.concat([home, away], ignore_index=True)

    long["tot"] = long["gf"] + long["ga"]
    long["over25"] = (long["tot"] > 2.5).astype(float)
    long["btts"] = ((long["gf"] > 0) & (long["ga"] > 0)).astype(float)
    long["cs"] = (long["ga"] == 0).astype(float)          # clean sheet
    long["fts"] = (long["gf"] == 0).astype(float)          # failed to score
    long["pts"] = np.where(long["gf"] > long["ga"], 3.0, np.where(long["gf"] == long["ga"], 1.0, 0.0))

    return long.sort_values(["team_id", "kickoff"]).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
# 2. Klizni prosjeci — stanje forme NAKON svake odigrane utakmice
# ═══════════════════════════════════════════════════════════════

def _rolling_state(long: pd.DataFrame, window: int) -> pd.DataFrame:
    """Za svaki redak vraća formu momčadi uključujući taj redak.

    Rezultat se poslije spaja unatrag (`merge_asof`), pa "uključujući taj redak"
    znači "stanje nakon posljednje ranije utakmice" iz perspektive cilja.
    """
    if long.empty:
        cols = ["team_id", "kickoff", *[f"f_{s}" for s in BASE_STATS], "f_n"]
        return pd.DataFrame(columns=cols)

    long = long.sort_values(["team_id", "kickoff"]).reset_index(drop=True)
    grouped = long.groupby("team_id", sort=False)
    out = long[["team_id", "kickoff"]].copy()

    for stat in BASE_STATS:
        rolled = grouped[stat].rolling(window, min_periods=1)
        if stat in ("over25", "btts", "cs", "fts"):
            # Za binarne pokazatelje čuvamo apsolutni broj (npr. "7 od 10"),
            # jer se isti broj prikazuje i u sučelju.
            values = rolled.sum()
        else:
            values = rolled.mean()
        out[f"f_{stat}"] = values.reset_index(level=0, drop=True).to_numpy()

    out["f_n"] = grouped["gf"].rolling(window, min_periods=1).count().reset_index(level=0, drop=True).to_numpy()
    out["last_kickoff"] = long["kickoff"].to_numpy()
    return out


def _as_of_join(
    targets: pd.DataFrame,
    state: pd.DataFrame,
    team_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Spaja stanje forme na ciljane utakmice bez curenja podataka."""
    if targets.empty:
        return targets

    left = (
        targets[["fixture_id", "kickoff", team_col]]
        .rename(columns={team_col: "team_id"})
        .sort_values("kickoff")
        .reset_index(drop=True)
    )
    if state.empty:
        merged = left.copy()
        for stat in [*BASE_STATS, "n"]:
            merged[f"f_{stat}"] = np.nan
        merged["last_kickoff"] = pd.NaT
    else:
        right = state.sort_values("kickoff").reset_index(drop=True)
        merged = pd.merge_asof(
            left,
            right,
            on="kickoff",
            by="team_id",
            direction="backward",
            allow_exact_matches=False,  # ključno: vlastita utakmica se isključuje
        )

    rename = {f"f_{stat}": f"{prefix}_{stat}" for stat in [*BASE_STATS, "n"]}
    merged = merged.rename(columns=rename)
    merged[f"{prefix}_rest_days"] = (
        merged["kickoff"] - merged["last_kickoff"]
    ).dt.total_seconds() / 86400.0

    keep = ["fixture_id", *rename.values(), f"{prefix}_rest_days"]
    return merged[keep]


# ═══════════════════════════════════════════════════════════════
# 2b. Udarci — isti point-in-time postupak kao za golove
# ═══════════════════════════════════════════════════════════════

def shots_to_long(fixtures: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Za svaki par (utakmica, momčad) vraća vlastite i protivnikove udarce.

    Tablica `fixture_stats` ima jedan redak po momčadi. Spajanjem iste tablice
    dvaput — jednom po vlastitom, jednom po protivničkom ID-u — dobiva se i
    napadačka i obrambena strana istog pokazatelja.
    """
    if stats is None or stats.empty or fixtures.empty:
        return pd.DataFrame(columns=["fixture_id", "team_id", *SHOT_COLS])

    pairs = pd.concat(
        [
            fixtures[["fixture_id", "home_team_id", "away_team_id"]].rename(
                columns={"home_team_id": "team_id", "away_team_id": "opp_id"}
            ),
            fixtures[["fixture_id", "away_team_id", "home_team_id"]].rename(
                columns={"away_team_id": "team_id", "home_team_id": "opp_id"}
            ),
        ],
        ignore_index=True,
    )
    own = stats.rename(columns={c: f"{c}_f" for c in SHOT_STATS})
    opp = stats.rename(columns={"team_id": "opp_id", **{c: f"{c}_a" for c in SHOT_STATS}})
    out = pairs.merge(own, on=["fixture_id", "team_id"], how="left")
    out = out.merge(opp, on=["fixture_id", "opp_id"], how="left")
    return out.drop(columns=["opp_id"])


def _rolling_shots(long_shots: pd.DataFrame, kickoffs: pd.DataFrame, window: int) -> pd.DataFrame:
    """Klizni prosjek udaraca po momčadi, uključujući tekući redak."""
    if long_shots.empty:
        return pd.DataFrame(columns=["team_id", "kickoff", *SHOT_COLS])

    df = long_shots.merge(kickoffs, on="fixture_id", how="left")
    df = df.sort_values(["team_id", "kickoff"]).reset_index(drop=True)
    grouped = df.groupby("team_id", sort=False)
    out = df[["team_id", "kickoff"]].copy()
    for col in SHOT_COLS:
        out[col] = (
            grouped[col].rolling(window, min_periods=1).mean()
            .reset_index(level=0, drop=True).to_numpy()
        )
    return out


def _join_shots(targets: pd.DataFrame, state: pd.DataFrame, team_col: str, prefix: str) -> pd.DataFrame:
    """Spaja stanje udaraca unatrag — isto pravilo kao kod forme."""
    left = (
        targets[["fixture_id", "kickoff", team_col]]
        .rename(columns={team_col: "team_id"})
        .sort_values("kickoff")
        .reset_index(drop=True)
    )
    rename = {c: f"{prefix}_{c}" for c in SHOT_COLS}
    if state.empty:
        merged = left.copy()
        for col in rename.values():
            merged[col] = np.nan
        return merged[["fixture_id", *rename.values()]]

    merged = pd.merge_asof(
        left,
        state.sort_values("kickoff").reset_index(drop=True),
        on="kickoff",
        by="team_id",
        direction="backward",
        allow_exact_matches=False,  # isto pravilo: vlastita utakmica se iskljucuje
    )
    return merged.rename(columns=rename)[["fixture_id", *rename.values()]]


# ═══════════════════════════════════════════════════════════════
# 3. Široka matrica značajki — jedan redak po utakmici
# ═══════════════════════════════════════════════════════════════

def build_feature_matrix(
    fixtures: pd.DataFrame,
    targets: pd.DataFrame | None = None,
    window_overall: int | None = None,
    window_venue: int | None = None,
    window_recent: int | None = None,
    stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Gradi matricu značajki.

    Args:
        fixtures: SVE poznate utakmice (povijest) — izvor forme. Neodigrane
            utakmice se automatski ignoriraju kao izvor.
        targets: utakmice za koje se traže značajke. Ako je None, uzimaju se
            sve odigrane utakmice iz `fixtures` (režim treniranja).
        window_overall: veličina prozora za ukupnu formu.
        window_venue: veličina prozora za formu kod kuće / u gostima.
    """
    window_overall = window_overall or settings.form_window_overall
    window_venue = window_venue or settings.form_window_venue
    window_recent = window_recent or settings.form_window_recent

    if fixtures.empty:
        return pd.DataFrame()

    fixtures = fixtures.copy()
    fixtures["kickoff"] = pd.to_datetime(fixtures["kickoff"], utc=True)

    played = fixtures[fixtures["finished"] & fixtures["home_goals"].notna()].copy()

    if targets is None:
        targets = played.copy()
    else:
        targets = targets.copy()
        targets["kickoff"] = pd.to_datetime(targets["kickoff"], utc=True)

    if targets.empty:
        return pd.DataFrame()

    long = to_long_frame(played)
    long_home = long[long["is_home"]].copy()
    long_away = long[~long["is_home"]].copy()

    state_overall = _rolling_state(long, window_overall)
    state_home = _rolling_state(long_home, window_venue)
    state_away = _rolling_state(long_away, window_venue)
    state_recent = _rolling_state(long, window_recent)

    parts = [
        _as_of_join(targets, state_overall, "home_team_id", "h_ov"),
        _as_of_join(targets, state_overall, "away_team_id", "a_ov"),
        _as_of_join(targets, state_home, "home_team_id", "h_vn"),
        _as_of_join(targets, state_away, "away_team_id", "a_vn"),
        _as_of_join(targets, state_recent, "home_team_id", "h_re"),
        _as_of_join(targets, state_recent, "away_team_id", "a_re"),
    ]

    keep_cols = [
        "fixture_id",
        "kickoff",
        "league_id",
        "home_team_id",
        "away_team_id",
        "home_goals",
        "away_goals",
    ]
    out = targets[keep_cols].copy()
    if "tier" in targets.columns:
        out["league_tier"] = targets["tier"].fillna(3)
    else:
        out["league_tier"] = 3

    for part in parts:
        out = out.merge(part, on="fixture_id", how="left")

    # `rest_days` se javlja četiri puta (po jednom za svaki spoj) — zadržavamo
    # ukupnu formu kao mjerodavnu jer pokriva sve utakmice, ne samo domaće/gostujuće.
    out["h_rest_days"] = out["h_ov_rest_days"]
    out["a_rest_days"] = out["a_ov_rest_days"]
    out = out.drop(columns=[c for c in out.columns if c.endswith("_vn_rest_days")], errors="ignore")
    out = out.drop(columns=["h_ov_rest_days", "a_ov_rest_days"], errors="ignore")

    # ── Izvedene značajke ──
    # Očekivani golovi kao prosjek napada jedne i obrane druge momčadi.
    out["exp_goals_home"] = (out["h_ov_gf"] + out["a_ov_ga"]) / 2.0
    out["exp_goals_away"] = (out["a_ov_gf"] + out["h_ov_ga"]) / 2.0
    out["exp_total"] = out["exp_goals_home"] + out["exp_goals_away"]
    out["attack_diff"] = out["h_ov_gf"] - out["a_ov_gf"]
    out["form_diff"] = out["h_ov_pts"] - out["a_ov_pts"]

    # Kratki prozor sluzi kao dopuna dugom: zadrzava se podskup pokazatelja,
    # jer bi puni blok udvostrucio broj znacajki bez mjerljive koristi.
    keep_recent = {f"{side}_re_{stat}" for side in ("h", "a") for stat in RECENT_STATS}
    out = out.drop(
        columns=[c for c in out.columns if "_re_" in c and c not in keep_recent],
        errors="ignore",
    )

    # ── Udarci (samo ako statistika postoji za te utakmice) ──
    if stats is not None and not stats.empty:
        long_shots = shots_to_long(played, stats)
        shot_state = _rolling_shots(
            long_shots, played[["fixture_id", "kickoff"]], window_overall
        )
        out = out.merge(_join_shots(targets, shot_state, "home_team_id", "h"), on="fixture_id", how="left")
        out = out.merge(_join_shots(targets, shot_state, "away_team_id", "a"), on="fixture_id", how="left")
    else:
        for col in SHOT_FEATURES:
            out[col] = np.nan

    for col in ("h_injuries", "a_injuries"):
        if col not in out.columns:
            out[col] = 0.0

    # ── Ciljne varijable (samo za odigrane utakmice) ──
    # Oznaka se racuna iz rezultata nakon 90 MINUTA, ne iz konacnog rezultata.
    # Kod utakmica sa statusom AET `home_goals` ukljucuje i golove iz produzetaka
    # (izmjereno: prosjek 3,90 gola naspram 2,46 u regularnom dijelu), a
    # kladionice trziste Over/Under 2.5 namiruju na 90 minuta. Ako 90-minutni
    # rezultat nije poznat, pada se natrag na konacni.
    if "home_ft" in targets.columns:
        out["home_ft"] = targets["home_ft"]
        out["away_ft"] = targets["away_ft"]
        home_final = out["home_ft"].fillna(out["home_goals"])
        away_final = out["away_ft"].fillna(out["away_goals"])
    else:
        home_final, away_final = out["home_goals"], out["away_goals"]

    total_goals = home_final + away_final
    out["y_over25"] = np.where(total_goals.notna(), (total_goals > 2.5).astype(float), np.nan)
    out["y_btts"] = np.where(
        home_final.notna() & away_final.notna(),
        ((home_final > 0) & (away_final > 0)).astype(float),
        np.nan,
    )

    for col in MODEL_FEATURES:
        if col not in out.columns:
            out[col] = np.nan

    return out


def attach_odds_features(matrix: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Dodaje implicirane vjerojatnosti iz koeficijenata.

    Kladionički koeficijent je najjači pojedinačni prediktor koji postoji jer u
    sebi sadrži informaciju cijelog tržišta. Zato se drži u zasebnom skupu
    značajki: model se trenira i sa i bez njega, pa se rezultati uspoređuju.
    """
    matrix = matrix.copy()
    for col in ODDS_FEATURES:
        matrix[col] = np.nan
    if odds is None or odds.empty:
        return matrix

    wide = odds.pivot_table(index="fixture_id", columns="market", values="price", aggfunc="max")
    for market in ("over25", "under25", "btts", "btts_no"):
        if market not in wide.columns:
            wide[market] = np.nan

    inv_over, inv_under = 1.0 / wide["over25"], 1.0 / wide["under25"]
    inv_btts, inv_btts_no = 1.0 / wide["btts"], 1.0 / wide["btts_no"]
    book_over = inv_over + inv_under
    book_btts = inv_btts + inv_btts_no

    derived = pd.DataFrame(
        {
            # Normalizacijom se uklanja marža kladionice ("overround").
            "imp_over25": inv_over / book_over,
            "imp_btts": inv_btts / book_btts,
            "odds_margin": book_over - 1.0,
        },
        index=wide.index,
    ).reset_index()

    matrix = matrix.drop(columns=ODDS_FEATURES).merge(derived, on="fixture_id", how="left")
    return matrix
