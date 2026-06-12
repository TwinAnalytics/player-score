# src/export_sofascore_frontend.py
"""
Builds the website-facing Sofascore metrics CSV from the full matched table.

Input:  Data/Processed/player_sofascore_stats.csv  (~115 columns, 10 MB)
Output: Data/Processed/player_sofa_metrics.csv     (lean selection, ~2 MB)

Selects the metrics shown in the frontend (duels, big chances, errors,
goalkeeping) and derives per-90 values from Sofascore minutes. The Sofascore
rating is intentionally NOT exported — it is only used internally for score
validation.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# (output name, sofa column, as_per90)
METRICS = [
    ("GroundDuelsWonPct", "sofa_groundDuelsWonPercentage", False),
    ("AerialDuelsWonPct", "sofa_aerialDuelsWonPercentage", False),
    ("TotalDuelsWonPct", "sofa_totalDuelsWonPercentage", False),
    ("GroundDuelsWon90", "sofa_groundDuelsWon", True),
    ("AerialDuelsWon90", "sofa_aerialDuelsWon", True),
    ("BigChancesCreated", "sofa_bigChancesCreated", False),
    ("BigChancesCreated90", "sofa_bigChancesCreated", True),
    ("BigChancesMissed", "sofa_bigChancesMissed", False),
    ("PossWonAttThird90", "sofa_possessionWonAttThird", True),
    ("Recoveries90", "sofa_ballRecovery", True),
    ("Dispossessed90", "sofa_dispossessed", True),
    ("Touches90", "sofa_touches", True),
    ("ErrorsLeadToShot", "sofa_errorLeadToShot", False),
    ("ErrorsLeadToGoal", "sofa_errorLeadToGoal", False),
    # Goalkeeping
    ("Saves90", "sofa_saves", True),
    ("GoalsPrevented", "sofa_goalsPrevented", False),
    ("GoalsConceded90", "sofa_goalsConceded", True),
    ("HighClaims90", "sofa_highClaims", True),
    ("CleanSheets", "sofa_cleanSheet", False),
    ("PenaltiesFaced", "sofa_penaltyFaced", False),
    ("PenaltiesSaved", "sofa_penaltySave", False),
]


def export_sofa_metrics(processed_dir: Path) -> None:
    processed_dir = Path(processed_dir)
    src = processed_dir / "player_sofascore_stats.csv"
    if not src.exists():
        print("[SOFA EXPORT] player_sofascore_stats.csv not found, skipping.")
        return

    df = pd.read_csv(src)
    minutes = df["sofa_minutesPlayed"].where(df["sofa_minutesPlayed"] > 0)

    out = pd.DataFrame({
        "Player": df["Player"],
        "Squad": df["Squad"],
        "Season": df["season"],
        "Comp": df["Comp"],
        "PlayerId": df["sofa_player_id"],
        "PosGroup": df["sofa_position_group"],
        "SofaMinutes": df["sofa_minutesPlayed"],
    })
    for name, col, per90 in METRICS:
        vals = df[col]
        if per90:
            vals = vals / minutes * 90
        out[name] = vals.round(2)

    dest = processed_dir / "player_sofa_metrics.csv"
    out.to_csv(dest, index=False)
    print(f"[SOFA EXPORT] {len(out)} rows -> {dest}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    export_sofa_metrics(root / "Data" / "Processed")
