# src/build_dob_fallback.py
"""
Fills missing birth dates from the local Transfermarkt dump.

Players who debuted after the FBref era ended and whose Sofascore profile is
not scraped yet have no age in the pipeline output. Many of them exist in
Data/Raw/Transfermarkt/tm_players.csv though. This matches them by
normalized name (exact, then fuzzy with a unique-hit requirement) and writes

    Data/Raw/Sofascore/player_dob_fallback.csv  (player_id, date_of_birth, source)

which processing_sofa picks up as a third fallback after FBref inheritance
and Sofascore profiles. Safe to re-run; rebuilt from scratch each time.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process as rfprocess

from .match_sofascore import _normalize

ROOT = Path(__file__).resolve().parents[2]
SOFA_DIR = ROOT / "Data" / "Raw" / "Sofascore"
TM_PATH = ROOT / "Data" / "Raw" / "Transfermarkt" / "tm_players.csv"
OUT = SOFA_DIR / "player_dob_fallback.csv"


def main():
    frames = [pd.read_csv(p, usecols=["player_id", "player_name"])
              for p in glob.glob(str(SOFA_DIR / "sofascore_player_stats-*-2025-2026.csv"))]
    players = pd.concat(frames).drop_duplicates("player_id")

    # Read the scraped profiles directly — NOT via load_profiles, which merges
    # this script's own previous output and would empty it on re-runs.
    prof_path = SOFA_DIR / "player_profiles.csv"
    if prof_path.exists():
        profiles = pd.read_csv(prof_path, usecols=["player_id", "date_of_birth"])
        have_dob = set(profiles.loc[profiles["date_of_birth"].notna(), "player_id"])
    else:
        have_dob = set()
    todo = players[~players["player_id"].isin(have_dob)]

    tm = pd.read_csv(TM_PATH, usecols=["name", "date_of_birth"]).dropna()
    tm["norm"] = tm["name"].map(_normalize)
    tm = tm.drop_duplicates("norm", keep=False)  # ambiguous names are useless here
    tm_lookup = dict(zip(tm["norm"], tm["date_of_birth"]))
    tm_norms = list(tm_lookup)

    rows = []
    for p in todo.itertuples(index=False):
        norm = _normalize(p.player_name)
        dob = tm_lookup.get(norm)
        if dob is None:
            hits = rfprocess.extract(norm, tm_norms, scorer=fuzz.token_set_ratio,
                                     score_cutoff=93, limit=2)
            if len(hits) == 1:  # only accept unambiguous fuzzy hits
                dob = tm_lookup[hits[0][0]]
        if dob is not None:
            rows.append({"player_id": p.player_id,
                         "date_of_birth": str(dob)[:10],
                         "source": "transfermarkt"})

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"{len(out)}/{len(todo)} fehlende Geburtsdaten via Transfermarkt gefunden -> {OUT}")


if __name__ == "__main__":
    main()
