# src/processing_sofa.py
"""
Role classification and feature preparation for the Sofascore-based pipeline.

Replaces the FBref position logic (prepare_positions / refine_mf_with_zones):
roles now come from the Sofascore player profile's detailed positions
(e.g. ["DM", "MC"]), with the coarse position group (G/D/M/F) from the
season statistics as fallback for players without a profile.

Roles match the existing scoring vocabulary: GK, DF, Def_MF, MF, Off_MF, FW.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Sofascore detailed position -> role. First entry of positions_detailed wins.
DETAILED_TO_ROLE = {
    "GK": "GK",
    "DC": "DF", "DL": "DF", "DR": "DF", "LB": "DF", "RB": "DF",
    "LWB": "DF", "RWB": "DF",
    "DM": "Def_MF",
    "MC": "MF", "ML": "MF", "MR": "MF",
    "AM": "Off_MF", "AML": "Off_MF", "AMR": "Off_MF",
    "LW": "FW", "RW": "FW", "ST": "FW", "FW": "FW",
}

# Fallback: coarse position group from season stats
GROUP_TO_ROLE = {"G": "GK", "D": "DF", "M": "MF", "F": "FW"}

# Sofascore league slug -> FBref-style Comp label (keeps the website filters stable)
LEAGUE_TO_COMP = {
    "premier-league": "eng Premier League",
    "laliga": "es La Liga",
    "bundesliga": "de Bundesliga",
    "serie-a": "it Serie A",
    "ligue-1": "fr Ligue 1",
}


def role_from_profile(positions_detailed: str | float, position_group: str) -> str:
    """positions_detailed: pipe-joined string from player_profiles.csv (may be NaN/empty)."""
    if isinstance(positions_detailed, str) and positions_detailed:
        for pos in positions_detailed.split("|"):
            role = DETAILED_TO_ROLE.get(pos.strip())
            if role:
                return role
    return GROUP_TO_ROLE.get(str(position_group), "MF")


def age_in_season(date_of_birth: pd.Series, season: str) -> pd.Series:
    """Age on Jan 1 of the season's closing year (matches FBref's convention closely)."""
    ref = pd.Timestamp(f"{season.split('-')[1]}-01-01")
    dob = pd.to_datetime(date_of_birth, errors="coerce")
    return ((ref - dob).dt.days / 365.25).round(1)


def load_profiles(sofascore_dir: Path) -> pd.DataFrame:
    path = Path(sofascore_dir) / "player_profiles.csv"
    if not path.exists():
        return pd.DataFrame(columns=["player_id", "positions_detailed", "date_of_birth",
                                     "height", "preferred_foot", "country", "market_value_eur"])
    df = pd.read_csv(path)
    return df.drop_duplicates("player_id", keep="last")


def fbref_meta_by_sofa_id(season: str, processed_dir: Path) -> dict[int, tuple[str, float]]:
    """
    {sofa_player_id: (Pos, Age)} for one season, inherited from the FBref era
    via the matching table. Keeps roles and ages consistent with the frozen
    history; players without an FBref match (e.g. debuts after the last FBref
    run) fall back to profile/coarse classification.
    """
    processed_dir = Path(processed_dir)
    match_path = processed_dir / "player_sofascore_stats.csv"
    fbref_path = processed_dir / f"player_scores-{season}.csv"
    if not match_path.exists() or not fbref_path.exists():
        return {}

    m = pd.read_csv(match_path, usecols=["season", "Player", "sofa_player_id"])
    m = m[m["season"] == season].drop_duplicates("sofa_player_id")
    fb = pd.read_csv(fbref_path, usecols=["Player", "Pos", "Age", "Min"])
    # One row per player: keep the row with most minutes (transfers)
    fb = fb.sort_values("Min").drop_duplicates("Player", keep="last")

    joined = m.merge(fb, on="Player", how="inner")
    return {
        int(r.sofa_player_id): (r.Pos, r.Age)
        for r in joined.itertuples(index=False)
        if isinstance(r.Pos, str)
    }


def build_season_table(season: str, sofascore_dir: Path,
                       processed_dir: Path | None = None) -> pd.DataFrame:
    """
    One row per player-league for `season`, with role, age and Comp label.
    Numeric stat columns stay untouched (scoring derives per-90s itself).
    """
    sofascore_dir = Path(sofascore_dir)
    frames = []
    for slug, comp in LEAGUE_TO_COMP.items():
        path = sofascore_dir / f"sofascore_player_stats-{slug}-{season}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["Comp"] = comp
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    table = pd.concat(frames, ignore_index=True)
    profiles = load_profiles(sofascore_dir)
    table = table.merge(
        profiles[["player_id", "positions_detailed", "date_of_birth", "height",
                  "preferred_foot", "country", "market_value_eur"]],
        on="player_id", how="left",
    )

    table["Pos"] = [
        role_from_profile(pd_str, grp)
        for pd_str, grp in zip(table["positions_detailed"], table["position_group"])
    ]
    table["Age"] = age_in_season(table["date_of_birth"], season)

    # Inherit role and age from the FBref era where a match exists
    if processed_dir is not None:
        meta = fbref_meta_by_sofa_id(season, processed_dir)
        if meta:
            fb_pos = table["player_id"].map(lambda pid: meta.get(pid, (None, None))[0])
            fb_age = table["player_id"].map(lambda pid: meta.get(pid, (None, None))[1])
            table["Pos"] = fb_pos.fillna(table["Pos"])
            table["Age"] = fb_age.fillna(table["Age"])
    table["Player"] = table["player_name"]
    table["Squad"] = table["team_name"]
    table["Min"] = table["minutesPlayed"]
    table["90s"] = (table["minutesPlayed"] / 90).round(1)
    return table
