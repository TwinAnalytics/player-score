from __future__ import annotations

from pathlib import Path
from functools import reduce
from typing import List

import pandas as pd

from src.processing import prepare_positions, add_standard_per90, filter_by_90s
from src.scoring import (
    compute_off_scores,
    compute_mid_scores,
    compute_def_scores,
)

# ---------------------------------------------------------------------------
# Spaltenauswahl aus der players_data_light-CSV
# ---------------------------------------------------------------------------

# SCORE_INPUT_COLS: List[str] = [
#     "Rk",
#     "Player",
#     "Pos",
#     "Squad",
#     "Comp",  # falls vorhanden
#     "Age",
#     "MP",
#     "Min",
#     "90s",

#     # Offensiv-Stats
#     "Gls",
#     "npxG",
#     "G-PK",
#     "SoT",
#     "TB",
#     "Ast",
#     "xG",
#     "xAG",
#     "Sh/90",
#     "SoT/90",
#     "KP",
#     "SCA90",
#     "1/3",
#     "PPA",
#     "PrgC",
#     "PrgP",
#     "Mis",
#     "Succ",

#     # Defensiv-Stats
#     "Tkl",
#     "TklW",
#     "Int",
#     "Clr",
#     "Blocks",
#     "Tkl%",
#     "Err",
#     "Fls",
#     "Won%",

#     # Zonen-Touches
#     "Def Pen",
#     "Def 3rd_stats_possession",
#     "Mid 3rd_stats_possession",
#     "Att 3rd_stats_possession",
#     "Att Pen",
# ]


# ---------------------------------------------------------------------------
# Rohdaten laden & Feature-Tabelle bauen
# ---------------------------------------------------------------------------

def load_raw_for_scoring(season: str, raw_dir: Path) -> pd.DataFrame:
    """
    Lädt die komplette players_data-Season.csv aus Data/Raw,
    ohne Spalten auf SCORE_INPUT_COLS zu reduzieren.
    """
    season_safe = season.replace("/", "-")
    csv_path = raw_dir / f"players_data-{season_safe}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def build_feature_table(season: str, raw_dir: Path) -> pd.DataFrame:
    """
    - lädt players_data_light-Season.csv
    - reduziert auf Score-relevante Spalten
    - wendet Positionslogik an
    - filtert nach Einsatzzeit (90s >= 5)
    - rechnet per90-Spalten (Standardliste aus processing.DEFAULT_PER90_COLS)
    """
    df = load_raw_for_scoring(season, raw_dir)

    # Blocks-Alias für die Per90-Logik:
    if "Blocks" not in df.columns and "Blocks_stats_defense" in df.columns:
        df["Blocks"] = df["Blocks_stats_defense"]

    # Positionslogik aus processing.py (FW / MF / DF / Off_MF / Def_MF, GK raus)
    df = prepare_positions(df)

    # Minutenfilter (für alle Rollen gleich – Defensiv kannst du später separat strenger machen)
    df = filter_by_90s(df, min_90s=0.0)

    # per90-Spalten (nutzt DEFAULT_PER90_COLS aus processing.add_standard_per90)
    df = add_standard_per90(df)

    return df


# ---------------------------------------------------------------------------
# Scores pro Rolle zusammenführen
# ---------------------------------------------------------------------------

def compute_all_scores(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Erwartet eine Feature-Tabelle (nach build_feature_table) und
    gibt eine Gesamttabelle mit allen Scores & Bändern zurück.
    """
    df = df_features.copy()

    # Scores pro Rolle (Logik & Gewichte liegen in scoring.py)
    df_off = compute_off_scores(df)
    df_mf = compute_mid_scores(df)
    df_def = compute_def_scores(df)

    # Basis-Infos pro Spieler/Rolle
    base_cols = [c for c in ["Player", "Squad", "Comp", "Pos", "Age", "Min", "90s"] if c in df.columns]
    base = df[base_cols].drop_duplicates(
        subset=[c for c in ["Player", "Squad", "Pos"] if c in base_cols]
    )

    dfs_to_merge = [base]

    if not df_off.empty:
        dfs_to_merge.append(
            df_off[["Player", "Squad", "Pos", "OffScore_abs", "OffBand"]]
        )
    if not df_mf.empty:
        dfs_to_merge.append(
            df_mf[["Player", "Squad", "Pos", "MidScore_abs", "MidBand"]]
        )
    if not df_def.empty:
        dfs_to_merge.append(
            df_def[["Player", "Squad", "Pos", "DefScore_abs", "DefBand"]]
        )

    df_all = reduce(
        lambda left, right: left.merge(right, on=["Player", "Squad", "Pos"], how="left"),
        dfs_to_merge,
    )

    return df_all


# ---------------------------------------------------------------------------
# Top-Level-Pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(
    season: str,
    raw_dir: Path,
    processed_dir: Path,
) -> Path:
    """
    Full Run:
    - lädt players_data_light-Season.csv aus raw_dir
    - baut Feature-Tabelle (Spalten, Positionen, per90, Filter)
    - berechnet Off/Mid/Def-Scores + Bänder
    - speichert finale player_scores-Season.csv in processed_dir
    """
    df_features = build_feature_table(season, raw_dir)
    df_scores = compute_all_scores(df_features)

    processed_dir.mkdir(parents=True, exist_ok=True)
    season_safe = season.replace("/", "-")
    out_path = processed_dir / f"player_scores-{season_safe}.csv"
    df_scores.to_csv(out_path, index=False)

    return out_path
