"""
L1 + L3 normalization for PlayerScore.

L3: League strength multiplier — adjusts raw benchmark score by league quality.
     Applied first so weaker-league players rank lower in the percentile step.
     All Big-5 leagues currently set to 1.0 (no effect on current dataset).
     Ready for expansion to non-Big-5 leagues.

L1: Within-role percentile normalization — converts L3-adjusted raw score to
     a percentile rank within the same Season + Pos group, scaled 0–1000.
     Makes MainScore cross-position comparable:
     "Score 700" = 70th percentile of your role in that season.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Exact Comp values as they appear in the FBref-scraped CSVs
LEAGUE_WEIGHTS: dict[str, float] = {
    "eng Premier League": 1.00,
    "es La Liga": 1.00,
    "de Bundesliga": 0.97,
    "it Serie A": 0.97,
    "fr Ligue 1": 0.93,
    # Future non-Big-5 leagues:
    # "nl Eredivisie": 0.85,
    # "pt Primeira Liga": 0.83,
    # "tr Süper Lig": 0.80,
    # "us Major League Soccer": 0.72,
}


def _raw_score_for_pos(row: pd.Series) -> float:
    """Pick the role-relevant benchmark score for a player row."""
    pos = row.get("Pos", "")
    if pos in ("FW", "Off_MF"):
        val = row.get("OffScore_abs", np.nan)
    elif pos == "MF":
        val = row.get("MidScore_abs", np.nan)
    elif pos in ("DF", "Def_MF"):
        val = row.get("DefScore_abs", np.nan)
    else:
        return np.nan
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def _main_band(score: float) -> str:
    if pd.isna(score):
        return "Below Big-5 Level"
    if score >= 900:
        return "Exceptional"        # Top 10% in role
    if score >= 750:
        return "World Class"        # Top 25% in role
    if score >= 400:
        return "Top Starter"        # 25th–60th percentile
    if score >= 200:
        return "Solid Squad Player" # 10th–30th percentile
    return "Below Big-5 Level"      # Bottom 20%


def add_main_score(
    df: pd.DataFrame,
    weights: dict[str, float] = LEAGUE_WEIGHTS,
) -> pd.DataFrame:
    """
    Add RawScore, MainScore, MainBand columns to a player scores DataFrame.

    RawScore  — the role-relevant raw benchmark score (OffScore_abs / MidScore_abs /
                DefScore_abs), preserved for transparency and sub-score display.
    MainScore — L3-adjusted, L1-normalized percentile score (0–1000).
                Cross-position comparable within each Season+Pos group.
    MainBand  — qualitative tier for MainScore.

    Parameters
    ----------
    df      : DataFrame with columns Player, Pos, Comp, Season (or derived from
              filename), OffScore_abs, MidScore_abs, DefScore_abs.
    weights : Dict mapping Comp values to league strength multipliers.
              Unknown leagues default to 1.0 (neutral).
    """
    df = df.copy()

    # Step 1 — RawScore: pick the role-relevant benchmark score
    df["RawScore"] = df.apply(_raw_score_for_pos, axis=1)

    # Step 2 — L3: multiply by league weight (unknown leagues → 1.0)
    league_mult = df["Comp"].map(weights).fillna(1.0)
    df["_AdjScore"] = df["RawScore"] * league_mult

    # Step 3 — L1: within-role percentile per Season+Pos group → 0–1000
    # Players without a valid role score (NaN) are excluded from ranking.
    def _pct_to_1000(x: pd.Series) -> pd.Series:
        return x.rank(pct=True, na_option="keep") * 1000

    group_cols = [c for c in ("Season", "Pos") if c in df.columns]
    if group_cols:
        df["MainScore"] = (
            df.groupby(group_cols)["_AdjScore"]
            .transform(_pct_to_1000)
            .round(1)
        )
    else:
        # Fallback: no season grouping (single-season DataFrames from pipeline)
        df["MainScore"] = _pct_to_1000(df["_AdjScore"]).round(1)

    df.drop(columns=["_AdjScore"], inplace=True)

    # Step 4 — MainBand
    df["MainBand"] = df["MainScore"].apply(_main_band)

    return df
