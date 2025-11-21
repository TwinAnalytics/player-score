# src/squad.py
from __future__ import annotations

from typing import List

import pandas as pd


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

    required_cols = ["Season", "Squad", "Player", "Min"]
    missing = [c for c in required_cols if c not in df_all.columns]
    if missing:
        raise ValueError(f"compute_squad_scores: missing required columns: {missing}")

    def _agg(group: pd.DataFrame) -> pd.Series:
        result: dict[str, float | int | None] = {
            "n_players": group["Player"].nunique(),
            "minutes_total": float(
                pd.to_numeric(group["Min"], errors="coerce").fillna(0).sum()
            ),
        }

        for col in SQUAD_SCORE_COLS:
            if col in group.columns:
                squad_col = col.replace("_abs", "_squad")
                result[squad_col] = _weighted_mean(group, col)

        return pd.Series(result)

    df_squad = (
        df_all
        .groupby(["Season", "Squad"])
        .apply(_agg)
        .reset_index()
    )

    return df_squad
