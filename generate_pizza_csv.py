"""
Generate player_pizza_all_seasons.csv from raw FBref player data.

Extracts 14 per-90 pizza chart metrics from Data/Raw/players_data-*.csv
and saves a slim CSV to Data/Processed/player_pizza_all_seasons.csv.
"""

import glob
import os
import re

import pandas as pd

RAW_GLOB = "Data/Raw/players_data-*.csv"
OUT_PATH = "Data/Processed/player_pizza_all_seasons.csv"

# Columns to extract from raw data (raw names before per-90 conversion)
RAW_METRICS = {
    # Possession
    "Succ": "Succ_Per90",
    "Cmp%": "Cmp%",           # already a rate — kept as-is
    "PrgC": "PrgC_Per90",
    "PrgR": "PrgR_Per90",
    "TB": "TB_Per90",
    # Attacking
    "Gls": "Gls_Per90",
    "Ast": "Ast_Per90",
    "xG": "xG_Per90",
    "xAG": "xAG_Per90",
    "SoT": "SoT_Per90",
    "SCA": "SCA_Per90",
    "KP": "KP_Per90",
    # Defending
    "TklW": "TklW_Per90",
    "Int": "Int_Per90",
    "Blocks_stats_defense": "Blocks_Per90",
    "Clr": "Clr_Per90",
}

ID_COLS = ["Player", "Squad", "Comp", "Pos", "90s"]


def season_from_path(path: str) -> str:
    match = re.search(r"(\d{4}-\d{4})", path)
    return match.group(1) if match else "unknown"


def process_file(path: str) -> pd.DataFrame:
    season = season_from_path(path)
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"  ERROR reading {path}: {e}")
        return pd.DataFrame()

    df["Season"] = season

    # Ensure 90s column is numeric
    ninety_col = "90s"
    if ninety_col not in df.columns:
        print(f"  SKIP {path}: no '90s' column")
        return pd.DataFrame()

    df[ninety_col] = pd.to_numeric(df[ninety_col], errors="coerce")

    # Collect available metric columns
    out_rows = []
    keep_cols = ["Player", "Squad", "Comp", "Pos", "Season", "90s"]
    metric_out_cols = []

    for raw_col, out_col in RAW_METRICS.items():
        if raw_col not in df.columns:
            # Try fallback: raw_col without suffix
            found = False
            for c in df.columns:
                if c.startswith(raw_col):
                    raw_col = c
                    found = True
                    break
            if not found:
                # Add NaN column so schema is consistent
                df[out_col] = float("nan")
                metric_out_cols.append(out_col)
                continue

        if raw_col == "Cmp%":
            # Already a percentage rate — just copy
            df[out_col] = pd.to_numeric(df[raw_col], errors="coerce")
        else:
            # Divide by 90s to get per-90 rate
            numeric = pd.to_numeric(df[raw_col], errors="coerce")
            df[out_col] = numeric / df[ninety_col]

        metric_out_cols.append(out_col)

    result = df[keep_cols + metric_out_cols].copy()

    # Filter: only outfield players with data
    result = result[result["Pos"].notna() & ~result["Pos"].str.contains("GK", na=False)]
    result = result[result["90s"] >= 0.5]

    print(f"  {season}: {len(result)} rows")
    return result


def main():
    paths = sorted(glob.glob(RAW_GLOB))
    if not paths:
        print(f"No files found matching {RAW_GLOB}")
        return

    print(f"Processing {len(paths)} season files...")
    frames = []
    for path in paths:
        print(f"  {os.path.basename(path)}")
        df = process_file(path)
        if not df.empty:
            frames.append(df)

    if not frames:
        print("No data to save.")
        return

    combined = pd.concat(frames, ignore_index=True)

    # Round per-90 values to 3 decimal places
    per90_cols = [c for c in combined.columns if c.endswith("_Per90")]
    combined[per90_cols] = combined[per90_cols].round(3)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(combined)} rows to {OUT_PATH}")
    print(f"Columns: {list(combined.columns)}")


if __name__ == "__main__":
    main()
