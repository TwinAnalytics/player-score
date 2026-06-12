# src/export_shots_frontend.py
"""
Builds the website-facing shot data from the raw Sofascore shotmaps.

Output: Data/Processed/player_shots-{season}.csv — one row per shot, Big-5
leagues only, lean column set (~3-4 MB per season). The frontend lazy-loads
one season at a time when a player's shotmap is opened (same pattern as the
pizza CSV). Keyed by sofa player_id; the profile finds it via the PlayerId
column of player_sofa_metrics.csv.

Coordinates: Sofascore convention — x = distance from the attacked goal line
(0 = goal line, ~16 = edge of the box), y = pitch width position (0-100).

Usage:
    python -m src.export_shots_frontend
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SHOT_DIR = ROOT / "Data" / "Raw" / "Sofascore" / "Shotmaps"
OUT_DIR = ROOT / "Data" / "Processed"

BIG5 = {"premier-league", "laliga", "bundesliga", "serie-a", "ligue-1"}

COLUMNS = {
    "player_id": "PlayerId",
    "x": "X",
    "y": "Y",
    "goal_mouth_y": "GoalMouthY",
    "goal_mouth_z": "GoalMouthZ",
    "xg": "xG",
    "xgot": "xGOT",
    "shotType": "Result",         # goal / save / miss / block / post
    "situation": "Situation",     # regular / assisted / corner / set-piece / fast-break / penalty
    "bodyPart": "BodyPart",
    "time": "Minute",
    "isHome": "IsHome",
    "home_team": "HomeTeam",
    "away_team": "AwayTeam",
    "start_timestamp": "Timestamp",
}


def main():
    by_season: dict[str, list[pd.DataFrame]] = {}
    for path in sorted(glob.glob(str(SHOT_DIR / "shotmaps-*-????-????.csv"))):
        m = re.search(r"shotmaps-([a-z0-9-]+)-(\d{4}-\d{4})\.csv$", path)
        if not m or m.group(1) not in BIG5:
            continue
        df = pd.read_csv(path, usecols=list(COLUMNS))
        by_season.setdefault(m.group(2), []).append(df)

    for season, frames in sorted(by_season.items()):
        df = pd.concat(frames, ignore_index=True).rename(columns=COLUMNS)
        df["xG"] = df["xG"].round(3)
        df["xGOT"] = df["xGOT"].round(3)
        for col in ("X", "Y", "GoalMouthY", "GoalMouthZ"):
            df[col] = df[col].round(1)
        out = OUT_DIR / f"player_shots-{season}.csv"
        df.to_csv(out, index=False)
        print(f"{season}: {len(df)} shots -> {out.name} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
