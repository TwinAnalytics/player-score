"""
Style-based similar player search using pizza (per-90) metrics.
Similarity is computed via percentile-normalised Euclidean distance
on the 14 pizza chart dimensions.
"""

import numpy as np
import pandas as pd

PIZZA_COLS = [
    "Succ_Per90",   "Cmp%",          "PrgC_Per90",  "PrgR_Per90",
    "Gls_Per90",    "Ast_Per90",     "xG_Per90",    "xAG_Per90",
    "SCA_Per90",    "KP_Per90",
    "TklW_Per90",   "Int_Per90",     "Blocks_Per90", "Clr_Per90",
]


def find_similar_players(
    df_features: pd.DataFrame,
    df_scores: pd.DataFrame,
    player: str,
    role: str,
    n: int = 5,
) -> pd.DataFrame:
    """Return the *n* most stylistically similar players to *player*.

    Parameters
    ----------
    df_features:
        Feature table for a single season (output of
        ``load_feature_table_for_season()``).  Must contain a ``Player``
        column, a positional column (``Pos_raw`` or ``Pos``), and the
        pizza columns that are available.
    df_scores:
        Season slice of ``df_all`` that already has ``MainScore`` and
        ``MainBand`` columns.
    player:
        Name of the target player.
    role:
        Positional role string (e.g. ``"FW"``, ``"MF"``, ``"DF"``).
    n:
        Number of similar players to return (excluding the player themselves).

    Returns
    -------
    pd.DataFrame with columns [Player, Squad, Comp, MainScore, MainBand].
    Empty DataFrame if the player is not found or too few peers exist.
    """
    pos_col = "Pos_raw" if "Pos_raw" in df_features.columns else "Pos"

    peers = df_features[df_features[pos_col] == role].copy()

    # Require at least 450 minutes played
    if "Min" in peers.columns:
        min_col = pd.to_numeric(peers["Min"], errors="coerce").fillna(0)
    elif "90s" in peers.columns:
        min_col = pd.to_numeric(peers["90s"], errors="coerce").fillna(0) * 90
    else:
        min_col = pd.Series(9999, index=peers.index)  # no filter possible

    peers = peers[min_col >= 450]

    cols = [c for c in PIZZA_COLS if c in peers.columns]
    if not cols:
        return pd.DataFrame()

    X = peers[cols].copy().fillna(0)

    # Percentile-normalise each column to 0–100
    for c in cols:
        X[c] = X[c].rank(pct=True) * 100

    player_mask = peers["Player"] == player
    if not player_mask.any():
        return pd.DataFrame()

    vec = X[player_mask].values[0]
    dists = np.sqrt(((X.values - vec) ** 2).sum(axis=1))

    peers = peers.copy()
    peers["_dist"] = dists
    others = peers[peers["Player"] != player]
    top = others.nsmallest(n, "_dist")[["Player", "Squad", "Comp"]]

    score_cols = [c for c in ["MainScore", "MainBand"] if c in df_scores.columns]
    if score_cols:
        score_lookup = (
            df_scores[["Player"] + score_cols]
            .drop_duplicates("Player")
        )
        top = top.merge(score_lookup, on="Player", how="left")

    return top.reset_index(drop=True)
