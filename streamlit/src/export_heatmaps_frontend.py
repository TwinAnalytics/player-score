# src/export_heatmaps_frontend.py
"""
Builds the website-facing heatmap data from the raw Sofascore heatmaps.

Raw input is ~1000 grid points per player-season (x, y, count on a 100x100
pitch grid) — too heavy for the browser. This bins them into 5x5 cells
(20x20 grid) and writes one CSV per season:

    Data/Processed/player_heat-{season}.csv   (PlayerId, BX, BY, C)

The frontend lazy-loads one season at a time (same pattern as the shots).
Re-run any time; seasons are rebuilt from whatever league files exist, so
the data grows as the remaining league scrapes finish.

Usage:
    python -m src.export_heatmaps_frontend
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HEAT_DIR = ROOT / "Data" / "Raw" / "Sofascore" / "Heatmaps"
OUT_DIR = ROOT / "Data" / "Processed"

BIG5 = {"premier-league", "laliga", "bundesliga", "serie-a", "ligue-1"}
BIN = 5  # 100/BIN x 100/BIN grid


def main():
    by_season: dict[str, list[pd.DataFrame]] = {}
    for path in sorted(glob.glob(str(HEAT_DIR / "heatmaps-*-????-????.csv.gz"))):
        m = re.search(r"heatmaps-([a-z0-9-]+)-(\d{4}-\d{4})\.csv\.gz$", path)
        if not m or m.group(1) not in BIG5:
            continue
        df = pd.read_csv(path, usecols=["player_id", "x", "y", "count"])
        by_season.setdefault(m.group(2), []).append(df)

    for season, frames in sorted(by_season.items()):
        df = pd.concat(frames, ignore_index=True)
        df["BX"] = (df["x"] // BIN).clip(0, 100 // BIN - 1).astype(int)
        df["BY"] = (df["y"] // BIN).clip(0, 100 // BIN - 1).astype(int)
        binned = (
            df.groupby(["player_id", "BX", "BY"], as_index=False)["count"].sum()
            .rename(columns={"player_id": "PlayerId", "count": "C"})
        )
        out = OUT_DIR / f"player_heat-{season}.csv"
        binned.to_csv(out, index=False)
        print(f"{season}: {df.player_id.nunique()} players, {len(binned)} cells "
              f"-> {out.name} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
