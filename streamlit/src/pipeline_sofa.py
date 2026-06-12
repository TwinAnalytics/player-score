# src/pipeline_sofa.py
"""
Sofascore-based season pipeline — produces player_scores-{season}.csv for
seasons from 2025-26 onward in the exact schema of the FBref era, so all
downstream exports and the website keep working unchanged. Historical
FBref-based season files stay frozen.

Name continuity: player and squad names are written in the FBref spelling
wherever the player/team ever appeared in the FBref era (via the matching in
Data/Processed/player_sofascore_stats.csv). Players new to the Big-5 keep
their Sofascore name — they have no history to break.

New vs. the FBref era: goalkeepers are included (Pos == "GK") and carry a
GKScore_abs/GKBand. Outfield score columns stay NaN for them and vice versa.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .processing_sofa import build_season_table
from .scoring_sofa import compute_all_scores_sofa

OUTPUT_COLUMNS = [
    "Player", "Squad", "Comp", "Pos", "Age", "Min", "90s",
    "OffScore_abs", "OffBand", "MidScore_abs", "MidBand",
    "DefScore_abs", "DefBand", "IntensityScore_abs", "IntensityBand",
    "GKScore_abs", "GKBand",
]


def build_name_registry(processed_dir: Path) -> tuple[dict[int, str], dict[int, str]]:
    """
    {sofa_player_id: fbref_name}, {sofa_team_id: fbref_squad} from the
    FBref<->Sofascore matching table. Most recent season wins for players;
    majority vote for teams (player-level matching makes single rows
    unreliable for mid-season transfers).
    """
    path = Path(processed_dir) / "player_sofascore_stats.csv"
    if not path.exists():
        print("[SOFA PIPE WARN] no matching table found - using Sofascore names everywhere")
        return {}, {}

    m = pd.read_csv(path, usecols=["season", "Player", "Squad",
                                   "sofa_player_id", "sofa_team_id", "sofa_team_name"])

    players = (
        m.sort_values("season")
        .groupby("sofa_player_id")["Player"].last()
        .to_dict()
    )
    team_votes = m.groupby(["sofa_team_id", "Squad"]).size().reset_index(name="n")
    teams = (
        team_votes.sort_values("n")
        .groupby("sofa_team_id")["Squad"].last()
        .to_dict()
    )
    return players, teams


def run_sofa_pipeline(season: str, sofascore_dir: Path, processed_dir: Path) -> Path:
    sofascore_dir, processed_dir = Path(sofascore_dir), Path(processed_dir)

    table = build_season_table(season, sofascore_dir)
    if table.empty:
        raise FileNotFoundError(f"No Sofascore season stats for {season} in {sofascore_dir}")

    player_names, team_names = build_name_registry(processed_dir)
    table["Player"] = table["player_id"].map(player_names).fillna(table["Player"])
    table["Squad"] = table["team_id"].map(team_names).fillna(table["Squad"])

    scored = compute_all_scores_sofa(table)

    out = scored[OUTPUT_COLUMNS].copy()
    out_path = processed_dir / f"player_scores-{season}.csv"
    out.to_csv(out_path, index=False)

    n_gk = (out["Pos"] == "GK").sum()
    print(f"[SOFA PIPE] {season}: {len(out)} players ({n_gk} GK) -> {out_path}")
    return out_path


if __name__ == "__main__":
    import sys

    root = Path(__file__).resolve().parents[2]
    season = sys.argv[1] if len(sys.argv) > 1 else "2025-2026"
    run_sofa_pipeline(season, root / "Data" / "Raw" / "Sofascore", root / "Data" / "Processed")
