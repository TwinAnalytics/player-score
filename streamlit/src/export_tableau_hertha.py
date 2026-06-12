# src/export_tableau_hertha.py
"""
Builds a Tableau-ready data package for the Hertha BSC dashboard.

Output: Data/Exports/Tableau_Hertha/
  player_seasons.csv   one row per player-season: full 2. Bundesliga (context
                       cohort) + all Hertha Bundesliga seasons; all Sofascore
                       metrics + per-90 convenience columns + IsHertha flag
  shots.csv            one row per shot: all 2. Bundesliga shots (2022-23+)
                       + every shot in Hertha's Bundesliga matches (2022-23)
  matches.csv          one row per match (from the shot files' event metadata)
  heatmaps_hertha.csv  heatmap grid points of every Hertha player-season
                       (from whatever league heatmap files exist so far)
  player_profiles.csv  master data (DOB, positions, height, market value)
                       for all players appearing in player_seasons
  README.md            join keys / data notes for Tableau

Idempotent — re-run any time as more raw data lands (heatmaps, profiles).

Usage:
    python -m src.export_tableau_hertha
"""
from __future__ import annotations

import glob
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERTHA = "Hertha BSC"

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "Data" / "Raw" / "Sofascore"
OUT = ROOT / "Data" / "Exports" / "Tableau_Hertha"

LEAGUE_LABELS = {
    "2-bundesliga": "2. Bundesliga",
    "bundesliga": "Bundesliga",
    "premier-league": "Premier League",
    "laliga": "La Liga",
    "serie-a": "Serie A",
    "ligue-1": "Ligue 1",
}

PER90_COLS = ["goals", "assists", "expectedGoals", "expectedAssists", "keyPasses",
              "tackles", "interceptions", "ballRecovery", "touches", "totalDuelsWon"]


def _season_files(pattern: str) -> list[tuple[str, str, str]]:
    out = []
    for path in sorted(glob.glob(str(RAW / pattern))):
        m = re.search(r"-([a-z0-9-]+)-(\d{4}-\d{4})\.csv(\.gz)?$", path)
        if m:
            out.append((m.group(1), m.group(2), path))
    return out


def build_player_seasons() -> pd.DataFrame:
    frames = []
    for league, season, path in _season_files("sofascore_player_stats-*-????-????.csv"):
        df = pd.read_csv(path)
        df = df[(df["league"] == "2-bundesliga") | (df["team_name"] == HERTHA)]
        if not df.empty:
            frames.append(df)
    ps = pd.concat(frames, ignore_index=True)
    ps["League"] = ps["league"].map(LEAGUE_LABELS).fillna(ps["league"])
    ps["IsHertha"] = ps["team_name"] == HERTHA
    minutes = ps["minutesPlayed"].where(ps["minutesPlayed"] > 0)
    for col in PER90_COLS:
        ps[f"{col}_per90"] = (ps[col] / minutes * 90).round(3)
    return ps


def build_shots_and_matches() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for league, season, path in _season_files("Shotmaps/shotmaps-*-????-????.csv"):
        df = pd.read_csv(path)
        if league != "2-bundesliga":
            df = df[(df["home_team"] == HERTHA) | (df["away_team"] == HERTHA)]
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    shots = pd.concat(frames, ignore_index=True)
    shots["League"] = shots["league"].map(LEAGUE_LABELS).fillna(shots["league"])
    shots["MatchDate"] = pd.to_datetime(shots["start_timestamp"], unit="s", utc=True).dt.date
    shots["ShotTeam"] = shots["isHome"].map({True: None, False: None})
    shots.loc[shots["isHome"] == True, "ShotTeam"] = shots["home_team"]
    shots.loc[shots["isHome"] == False, "ShotTeam"] = shots["away_team"]
    shots["Opponent"] = shots["home_team"].where(shots["ShotTeam"] != shots["home_team"], shots["away_team"])
    shots["IsHerthaShot"] = shots["ShotTeam"] == HERTHA
    shots["IsGoal"] = shots["shotType"] == "goal"

    match_cols = ["league", "League", "season", "event_id", "MatchDate", "round",
                  "home_team", "away_team", "home_score", "away_score"]
    matches = shots[match_cols].drop_duplicates("event_id").reset_index(drop=True)
    matches["IsHerthaMatch"] = (matches["home_team"] == HERTHA) | (matches["away_team"] == HERTHA)
    return shots, matches


def build_hertha_heatmaps(player_seasons: pd.DataFrame) -> pd.DataFrame:
    hertha_keys = set(zip(
        player_seasons.loc[player_seasons["IsHertha"], "player_id"],
        player_seasons.loc[player_seasons["IsHertha"], "league"],
        player_seasons.loc[player_seasons["IsHertha"], "season"],
    ))
    frames = []
    for league, season, path in _season_files("Heatmaps/heatmaps-*-????-????.csv.gz"):
        df = pd.read_csv(path)
        df = df[[(pid, league, season) in hertha_keys for pid in df["player_id"]]]
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_profiles(player_seasons: pd.DataFrame) -> pd.DataFrame:
    path = RAW / "player_profiles.csv"
    if not path.exists():
        return pd.DataFrame()
    profiles = pd.read_csv(path).drop_duplicates("player_id", keep="last")
    return profiles[profiles["player_id"].isin(set(player_seasons["player_id"]))]


README = """# Tableau-Datenpaket: Hertha BSC Dashboard

Erzeugt von streamlit/src/export_tableau_hertha.py — bei neuen Rohdaten einfach
neu laufen lassen (python -m src.export_tableau_hertha).

## Dateien & Join-Keys

| Datei | Korn | Schlüssel |
|---|---|---|
| player_seasons.csv | Spieler × Saison × Liga | player_id, season, league |
| player_profiles.csv | Spieler (Stammdaten) | player_id |
| shots.csv | einzelner Schuss | event_id (→ matches), player_id |
| matches.csv | Spiel | event_id |
| heatmaps_hertha.csv | Rasterpunkt (x, y, count) | player_id, season, league |

## Tableau-Hinweise
- Beziehungen: player_seasons ↔ player_profiles über player_id;
  shots ↔ matches über event_id; shots ↔ player_profiles über player_id.
- player_seasons enthält die KOMPLETTE 2. Bundesliga (Vergleichskohorte) plus
  alle Hertha-Bundesliga-Saisons; Filter IsHertha für den Verein.
- Schusskoordinaten: x/y in Prozent der Spielfeldlänge/-breite, Angriffsrichtung
  normalisiert. goal_mouth_x/y/z = Position im Tormund. Kein Schuss-xG in der
  2. Bundesliga (Sofascore-Coverage), IsGoal als Ergebnis-Flag.
- Heatmap: count pro (x, y)-Rasterzelle, für Dichte-Visualisierung binnen.
- xG/xA in player_seasons erst ab Saison 2025-2026 (2. Bundesliga).
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    ps = build_player_seasons()
    ps.to_csv(OUT / "player_seasons.csv", index=False)
    print(f"player_seasons.csv: {len(ps)} rows ({ps.IsHertha.sum()} Hertha)")

    shots, matches = build_shots_and_matches()
    if not shots.empty:
        shots.to_csv(OUT / "shots.csv", index=False)
        matches.to_csv(OUT / "matches.csv", index=False)
        print(f"shots.csv: {len(shots)} rows ({shots.IsHerthaShot.sum()} Hertha) | matches.csv: {len(matches)}")

    hm = build_hertha_heatmaps(ps)
    if not hm.empty:
        hm.to_csv(OUT / "heatmaps_hertha.csv", index=False)
        print(f"heatmaps_hertha.csv: {len(hm)} rows, {hm.player_id.nunique()} Spieler")

    profiles = build_profiles(ps)
    if not profiles.empty:
        profiles.to_csv(OUT / "player_profiles.csv", index=False)
        print(f"player_profiles.csv: {len(profiles)} rows")

    (OUT / "README.md").write_text(README, encoding="utf-8")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
