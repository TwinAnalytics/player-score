# src/squad.py
from __future__ import annotations

from typing import List

import pandas as pd

import numpy as np


# Welche Score-Spalten wir pro Squad aggregieren
SQUAD_SCORE_COLS: List[str] = [
    "OffScore_abs",
    "MidScore_abs",
    "DefScore_abs",
]


def _weighted_mean(group: pd.DataFrame, value_col: str, weight_col: str = "Min") -> float | None:
    """
    Minuten-gewichteter Durchschnitt einer Score-Spalte.
    Fallback: einfacher Mittelwert, wenn Gewichtssumme 0 ist.
    """
    if value_col not in group.columns:
        return None

    s = pd.to_numeric(group[value_col], errors="coerce")
    w = pd.to_numeric(group.get(weight_col, 0), errors="coerce").fillna(0)

    mask = s.notna()
    if not mask.any():
        return None

    s = s[mask]
    w = w[mask]

    if w.sum() <= 0:
        return float(s.mean())

    return float((s * w).sum() / w.sum())


def compute_squad_scores(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregiert Spieler-Scores zu Squad-Scores je Season + Squad.

    Erwartet ein long-Format-DataFrame wie df_all aus load_all_seasons, mit:
    - Season
    - Squad
    - Player
    - Min
    - OffScore_abs / MidScore_abs / DefScore_abs (sofern vorhanden)
    """

    def _agg(group: pd.DataFrame) -> pd.Series:
        result: dict[str, float | int | str] = {}

        if isinstance(group.name, tuple):
            season_key, squad_key = group.name
        else:
            # Falls du irgendwann nur nach "Squad" gruppierst o.ä.
            season_key, squad_key = (group.get("Season", np.nan), group.name)

        result["Season"] = season_key
        result["Squad"] = squad_key


        # Minuten gesamt
        mins = pd.to_numeric(group.get("Min", 0), errors="coerce").fillna(0)
        result["Min_squad"] = float(mins.sum())

        # 90s gesamt (falls vorhanden)
        if "90s" in group.columns:
            nineties = pd.to_numeric(group["90s"], errors="coerce").fillna(0)
            result["90s_squad"] = float(nineties.sum())

        # Durchschnittsalter (minuten-gewichtet)
        if "Age" in group.columns:
            result["Age_squad_mean"] = _weighted_mean(group, "Age")

        # Anzahl unterschiedlicher Spieler
        result["NumPlayers_squad"] = int(group["Player"].nunique())

        # --- Score-Aggregate (minuten-gewichtet) ---
        for col in SQUAD_SCORE_COLS:
            if col in group.columns:
                squad_col = col.replace("_abs", "_squad")
                result[squad_col] = _weighted_mean(group, col)

        # --- Overall-Squad-Score (Mittelwert aus Off/Mid/Def) ---
        comp_scores = [
            result.get("OffScore_squad"),
            result.get("MidScore_squad"),
            result.get("DefScore_squad"),
        ]
        comp_scores = [v for v in comp_scores if v is not None]
        if comp_scores:
            result["OverallScore_squad"] = float(sum(comp_scores) / len(comp_scores))

        return pd.Series(result)

    df_squad = (
        df_all
        .groupby(["Season", "Squad"])
        .apply(_agg, include_groups=False)
        .reset_index(drop=True)
    )

    # -------------------------------------------------------
    # Add Comp (league) to squad scores (Season+Squad mapping)
    # -------------------------------------------------------
    if "Comp" in df_all.columns:
        comp_map = (
            df_all.dropna(subset=["Season", "Squad", "Comp"])
                .drop_duplicates(subset=["Season", "Squad"])[["Season", "Squad", "Comp"]]
        )
        df_squad = df_squad.merge(comp_map, on=["Season", "Squad"], how="left")

    return df_squad
