# src/export_gk_shots_frontend.py
"""
Builds the goalkeeper faced-shots data for the website.

For every Big-5 goalkeeper since 2022-23, collects all ON-TARGET shots
(goal or save) against his team in league play, with pitch coordinates,
goal-mouth coordinates and xGOT.

Attribution caveat: the raw shotmaps identify the shooter, not the keeper.
Shots are attributed via the keeper's team, so they are only exported for
keepers who played at least half of their team's league minutes; the
MinShare column lets the frontend label the chart honestly.

Output: Data/Processed/player_gk_shots-{season}.csv (lazy-loaded per season).

Usage:
    python -m src.export_gk_shots_frontend
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "Data" / "Raw" / "Sofascore"
OUT_DIR = ROOT / "Data" / "Processed"

BIG5 = {"premier-league", "laliga", "bundesliga", "serie-a", "ligue-1"}
FIRST_SEASON = "2022-2023"
MIN_SHARE = 0.5


def main():
    shot_files: dict[str, list[str]] = {}
    for path in sorted(glob.glob(str(RAW / "Shotmaps" / "shotmaps-*-????-????.csv"))):
        m = re.search(r"shotmaps-([a-z0-9-]+)-(\d{4}-\d{4})\.csv$", path)
        if m and m.group(1) in BIG5 and m.group(2) >= FIRST_SEASON:
            shot_files.setdefault(m.group(2), []).append(path)

    for season, paths in sorted(shot_files.items()):
        shots = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
        shots["FacedTeam"] = shots["away_team"].where(shots["isHome"] == True, shots["home_team"])
        on_target = shots[shots["shotType"].isin(["goal", "save"])].copy()

        # Matches per team (for the minutes share)
        home = shots[["event_id", "home_team"]].rename(columns={"home_team": "team"})
        away = shots[["event_id", "away_team"]].rename(columns={"away_team": "team"})
        matches_per_team = (
            pd.concat([home, away]).drop_duplicates().groupby("team").size()
        )

        gk_frames = []
        for path in glob.glob(str(RAW / f"sofascore_player_stats-*-{season}.csv")):
            lm = re.search(rf"stats-(.+)-{season}\.csv$", path)
            if lm and lm.group(1) in BIG5:
                df = pd.read_csv(path, usecols=["player_id", "player_name", "team_name",
                                                "position_group", "minutesPlayed"])
                gk_frames.append(df[df["position_group"] == "G"])
        gks = pd.concat(gk_frames, ignore_index=True)
        gks["team_minutes"] = gks["team_name"].map(matches_per_team) * 90
        gks["share"] = gks["minutesPlayed"] / gks["team_minutes"]
        gks = gks[gks["share"] >= MIN_SHARE]

        rows = []
        for gk in gks.itertuples(index=False):
            faced = on_target[on_target["FacedTeam"] == gk.team_name]
            if faced.empty:
                continue
            rows.append(pd.DataFrame({
                "PlayerId": gk.player_id,
                "X": faced["x"].round(1),
                "Y": faced["y"].round(1),
                "GMY": faced["goal_mouth_y"].round(1),
                "GMZ": faced["goal_mouth_z"].round(1),
                "Result": faced["shotType"],
                "xGOT": faced["xgot"].round(3),
                "Minute": faced["time"],
                "Share": round(gk.share, 2),
            }))

        out = pd.concat(rows, ignore_index=True)
        dest = OUT_DIR / f"player_gk_shots-{season}.csv"
        out.to_csv(dest, index=False)
        print(f"{season}: {len(gks)} GKs, {len(out)} faced shots -> {dest.name} "
              f"({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
