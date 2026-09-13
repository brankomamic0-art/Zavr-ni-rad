"""Razvrstavanje natjecanja po vrsti.

API-Football pod "sve lige jedne zemlje" vraća i juniorska, ženska i pričuvna
natjecanja. Ona nisu greška u podacima, ali kvare rang-listu: u juniorskom
nogometu pada osjetno više golova, pa model takve utakmice ispravno gura na vrh
liste za Over 2.5 — a to nije ono što korisnik traži kad pita za najbolje
utakmice dana.

Prepoznavanje je heuristika nad tekstom, pa vrijede dva pravila:

1. Obrasci za NAZIV MOMČADI i NAZIV NATJECANJA drže se odvojeno. Nastavak " B"
   kod momčadi znači pričuvnu momčad ("Elche B"), ali kod lige znači drugi rang
   ("Serie B") — miješanje to dvoje izbacilo bi cijelu brazilsku i talijansku
   drugu ligu kao da su pričuve.
2. Kad nije sigurno, utakmica ostaje `senior`. Radije propustiti pokoju
   juniorsku utakmicu nego izbaciti pravu.
"""

from __future__ import annotations

import re

SENIOR = "senior"
YOUTH = "youth"
WOMEN = "women"
RESERVE = "reserve"

CATEGORIES = (SENIOR, YOUTH, WOMEN, RESERVE)

# ═══════════════════════════════════════════════════════════════
# Obrasci za naziv NATJECANJA
# ═══════════════════════════════════════════════════════════════

_LEAGUE_YOUTH = re.compile(
    r"\bu-?(?:1[5-9]|2[0-3])\b"          # U17, U-19, U21…
    r"|\bsub-?(?:1[5-9]|2[0-3])\b"       # sub-20 (šp./port.)
    r"|\byouth\b|\bjunior\w*\b"
    r"|\bprimavera\b"                     # Italija
    r"|\bjugend\b"                        # Njemačka
    r"|\bjuvenil\b",                      # Španjolska
    re.IGNORECASE,
)

_LEAGUE_WOMEN = re.compile(
    r"\bwomen\w*\b|\bfemale\b|\bfemenin\w*\b|\bfeminin\w*\b|\bfeminino\b"
    r"|\bfrauen\b|\bdamen\b|\bdames\b"
    r"|\bkvinne\w*\b|\bkvinde\w*\b|\bnaiset\b"     # Norveška, Danska, Finska
    r"|\bdamallsvenskan\b|\btoppserien\b|\bnwsl\b"
    r"|\bžensk\w*\b|\bzensk\w*\b",
    re.IGNORECASE,
)

# Namjerno BEZ nastavka " B"/" II" — to su kod liga oznake ranga, ne pričuva.
_LEAGUE_RESERVE = re.compile(
    r"\breserves?\b|\brezerv\w*\b|\bb-?team\b",
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════════
# Obrasci za naziv MOMČADI
# ═══════════════════════════════════════════════════════════════

# Bez "junior": Boca Juniors, Argentinos Juniors i kolumbijski Junior FC su
# seniorski klubovi. Ta riječ je pouzdana samo u nazivu natjecanja.
_TEAM_YOUTH = re.compile(
    r"\bu-?(?:1[5-9]|2[0-3])\b"
    r"|\bsub-?(?:1[5-9]|2[0-3])\b"
    r"|\byouth\b|\bprimavera\b|\bjugend\b",
    re.IGNORECASE,
)

_TEAM_WOMEN = re.compile(
    r"\bwomen\w*\b|\bfemenin\w*\b|\bfeminin\w*\b|\bfrauen\b"
    r"|\(w\)\s*$"
    r"|\sw\.?\s*$"          # nastavak " W" — npr. "Brøndby W"
    r"|\sq\s*$",            # skandinavski zapis — npr. "KoldingQ W", "Odense Q"
    re.IGNORECASE,
)

_TEAM_RESERVE = re.compile(
    r"\breserves?\b|\brezerv\w*\b"
    r"|\sres\.?\s*$"              # "Boca Juniors Res."
    r"|\s(?:ii|iii|iv)\s*$"       # "Flora III", "Vasas II"
    r"|\s(?:b|c)\s*$"             # "Elche B"
    r"|\s[23]\s*$",               # "Tobol 2", "Zenit 2" — brojcana oznaka pricuve
    re.IGNORECASE,
)


def _classify_league(name: str | None) -> str | None:
    if not name:
        return None
    if _LEAGUE_WOMEN.search(name):
        return WOMEN
    if _LEAGUE_YOUTH.search(name):
        return YOUTH
    if _LEAGUE_RESERVE.search(name):
        return RESERVE
    return None


def _classify_team(name: str | None) -> str | None:
    if not name:
        return None
    if _TEAM_WOMEN.search(name):
        return WOMEN
    if _TEAM_YOUTH.search(name):
        return YOUTH
    if _TEAM_RESERVE.search(name):
        return RESERVE
    return None


def classify_competition(
    league_name: str | None,
    home_name: str | None = None,
    away_name: str | None = None,
) -> str:
    """Vraća `senior`, `youth`, `women` ili `reserve`.

    Gledaju se i naziv natjecanja i nazivi obiju momčadi, jer se juniorske i
    pričuvne momčadi znaju pojaviti u kupovima koji se zovu samo "Cup".
    """
    found = _classify_league(league_name)
    if found:
        return found
    for team in (home_name, away_name):
        found = _classify_team(team)
        if found:
            return found
    return SENIOR


def is_senior(
    league_name: str | None,
    home_name: str | None = None,
    away_name: str | None = None,
) -> bool:
    return classify_competition(league_name, home_name, away_name) == SENIOR
