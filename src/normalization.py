"""
L2 + L3 normalization for PlayerScore.

L3: League strength multiplier — adjusts raw benchmark score by league quality.
     Applied first so weaker-league players score lower in the normalization step.
     All Big-5 leagues currently set to 1.0 (no effect on current dataset).
     Ready for expansion to non-Big-5 leagues.

L2: Within-role Z-score normalization — converts L3-adjusted raw score to a
     z-score within the same Season + Pos group, then scales to 0–1000 centered
     at 500 (average player = 500). Scale factor K=200 means 2 standard deviations
     above the role average → score 900 (Exceptional threshold).

     "Score 700" = roughly 1 standard deviation above average for your role.
     "Score 900" = roughly 2 standard deviations above average (top ~2-3%).

     Advantage over L1 (percentile): preserves the spread of the distribution.
     True outliers are separated from the cluster — no artificial compression at
     the top where many players would otherwise share near-1000 scores.
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

# Scale factor: number of MainScore points per standard deviation above the mean.
# K=200 → 2 std above average = score 900 (Exceptional threshold).
# Roughly 2-3% of players per role reach Exceptional per season.
_K: float = 200.0


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
        return "Exceptional"        # ~2+ std above role average (~2-3% of players)
    if score >= 750:
        return "World Class"        # ~1-2 std above average (~10% of players)
    if score >= 400:
        return "Top Starter"        # around or above the role median
    if score >= 200:
        return "Solid Squad Player" # below average but not bottom tier
    return "Below Big-5 Level"      # significantly below role average


def _zscore_to_1000(x: pd.Series) -> pd.Series:
    """
    Convert a series of raw scores to MainScores via z-score normalization.
    Mean → 500, each standard deviation → ±K points, clipped to [0, 1000].
    NaN values remain NaN.
    """
    valid = x.dropna()
    if len(valid) < 2:
        return pd.Series(500.0, index=x.index)
    mean = valid.mean()
    std = valid.std()
    if std == 0:
        return pd.Series(500.0, index=x.index)
    z = (x - mean) / std
    return (500.0 + z * _K).clip(0.0, 1000.0).round(1)


def add_main_score(
    df: pd.DataFrame,
    weights: dict[str, float] = LEAGUE_WEIGHTS,
) -> pd.DataFrame:
    """
    Add RawScore, MainScore, MainBand columns to a player scores DataFrame.

    RawScore  — the role-relevant raw benchmark score (OffScore_abs / MidScore_abs /
                DefScore_abs), preserved for transparency and sub-score display.
    MainScore — L3-adjusted, L2-normalized z-score (0–1000, centered at 500).
                Cross-position comparable within each Season+Pos group.
                Average player of any role scores ~500. Exceptional (~2 std above) ≥900.
    MainBand  — qualitative tier for MainScore.

    Parameters
    ----------
    df      : DataFrame with columns Player, Pos, Comp, Season (or inferred),
              OffScore_abs, MidScore_abs, DefScore_abs.
    weights : Dict mapping Comp values to league strength multipliers.
              Unknown leagues default to 1.0 (neutral).
    """
    df = df.copy()

    # Step 1 — RawScore: pick the role-relevant benchmark score
    df["RawScore"] = df.apply(_raw_score_for_pos, axis=1)

    # Step 2 — L3: multiply by league weight (unknown leagues → 1.0)
    league_mult = df["Comp"].map(weights).fillna(1.0)
    df["_AdjScore"] = df["RawScore"] * league_mult

    # Step 3 — L2: within-role z-score per Season+Pos group → 0–1000
    group_cols = [c for c in ("Season", "Pos") if c in df.columns]
    if group_cols:
        df["MainScore"] = (
            df.groupby(group_cols)["_AdjScore"]
            .transform(_zscore_to_1000)
        )
    else:
        # Fallback: no season grouping (single-season DataFrames from pipeline)
        df["MainScore"] = _zscore_to_1000(df["_AdjScore"])

    df.drop(columns=["_AdjScore"], inplace=True)

    # Step 4 — MainBand
    df["MainBand"] = df["MainScore"].apply(_main_band)

    return df
