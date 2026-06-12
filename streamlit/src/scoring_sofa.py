# src/scoring_sofa.py
"""
Role scoring on Sofascore season statistics — used for seasons from 2025-26
onward (the FBref era 2017/18–2024/25 stays frozen as computed).

Design goals:
- Same formula and bands as the FBref scoring (compute_score_absolute,
  score_band_5 from src.scoring are reused directly).
- Same role weights as src/scoring.py wherever a Sofascore equivalent of the
  FBref feature exists; documented proxies where it does not:
    npxG        ≈ expectedGoals − 0.79 × penaltiesTaken
    PrgP        → accurateFinalThirdPasses (progressive passing proxy)
    PrgC        → successfulDribbles      (progressive carrying proxy)
    Carries     → touches                 (volume proxy, intensity only)
    PrgDist     → totalDuelsWon           (intensity engagement proxy)
    Def Pen/3rd → ballRecovery / aerialDuelsWon (defensive presence proxies)
- New: goalkeeper score (GKScore_abs/GKBand) — goalkeepers were excluded in
  the FBref era entirely.

Benchmarks are p95 values of qualified players (>= 1350 min) over the
completed xG-era seasons 2022-23 … 2024-25, per feature, stored in
src/sofa_benchmarks.json. Regenerate with:
    python -m src.scoring_sofa  (requires scraped stats + player profiles)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .scoring import compute_score_absolute, score_band_5

BENCHMARK_PATH = Path(__file__).resolve().parent / "sofa_benchmarks.json"
BENCHMARK_SEASONS = ["2022-2023", "2023-2024", "2024-2025"]
BENCHMARK_MIN_MINUTES = 1350
MIN_90S_FOR_SCORING = 5.0

# ------------------------------------------------------------------
# Per-90 feature construction
# ------------------------------------------------------------------

# feature -> sofa stat column (value / minutes * 90)
PER90_SOURCES = {
    "sofa_ast_p90": "assists",
    "sofa_xa_p90": "expectedAssists",
    "sofa_kp_p90": "keyPasses",
    "sofa_fthird_p90": "accurateFinalThirdPasses",
    "sofa_drib_p90": "successfulDribbles",
    "sofa_tklw_p90": "tacklesWon",
    "sofa_int_p90": "interceptions",
    "sofa_blk_p90": "blockedShots",
    "sofa_clr_p90": "clearances",
    "sofa_aerw_p90": "aerialDuelsWon",
    "sofa_gdw_p90": "groundDuelsWon",
    "sofa_recov_p90": "ballRecovery",
    "sofa_pwa3_p90": "possessionWonAttThird",
    "sofa_lball_p90": "accurateLongBalls",
    "sofa_duelw_p90": "totalDuelsWon",
    "sofa_touch_p90": "touches",
    "sofa_saves_p90": "saves",
    "sofa_gprev_p90": "goalsPrevented",
    "sofa_claims_p90": "highClaims",
    "sofa_runsout_p90": "successfulRunsOut",
}


def add_per90_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive all scoring features from raw Sofascore season totals."""
    df = df.copy()
    minutes = df["minutesPlayed"].where(df["minutesPlayed"] > 0)

    for feat, col in PER90_SOURCES.items():
        df[feat] = df[col] / minutes * 90

    df["sofa_npg_p90"] = (df["goals"] - df["penaltyGoals"].fillna(0)) / minutes * 90
    df["sofa_npxg_p90"] = (
        (df["expectedGoals"] - 0.79 * df["penaltiesTaken"].fillna(0)).clip(lower=0)
        / minutes * 90
    )
    df["sofa_tklint_p90"] = (df["tackles"].fillna(0) + df["interceptions"].fillna(0)) / minutes * 90

    # GK ratios (not per-90)
    saves = df["saves"].fillna(0)
    conceded = df["goalsConceded"].fillna(0)
    df["sofa_save_pct"] = (saves / (saves + conceded).replace(0, pd.NA)) * 100
    df["sofa_pass_pct"] = df["accuratePassesPercentage"]
    return df


# ------------------------------------------------------------------
# Role weights (mirroring src/scoring.py)
# ------------------------------------------------------------------

OFF_WEIGHTS = {
    # FBref FW:     G-PK .40  Ast .15  npxG .20  xAG .05  KP .10  PrgP .05  PrgC .05
    "FW": {
        "sofa_npg_p90": 0.40, "sofa_ast_p90": 0.15, "sofa_npxg_p90": 0.20,
        "sofa_xa_p90": 0.05, "sofa_kp_p90": 0.10, "sofa_fthird_p90": 0.05,
        "sofa_drib_p90": 0.05,
    },
    # FBref Off_MF: G-PK .25  Ast .20  npxG .15  xAG .10  KP .12  PrgP .08  PrgC .05
    "Off_MF": {
        "sofa_npg_p90": 0.25, "sofa_ast_p90": 0.20, "sofa_npxg_p90": 0.15,
        "sofa_xa_p90": 0.10, "sofa_kp_p90": 0.12, "sofa_fthird_p90": 0.08,
        "sofa_drib_p90": 0.05,
    },
}

MID_WEIGHTS = {
    # FBref MF: Ast .17 xAG .14 G-PK .07 KP .16 PrgP .11 TklW .08 Int .06 Mid3rd .06 Att3rd .04
    "MF": {
        "sofa_ast_p90": 0.17, "sofa_xa_p90": 0.14, "sofa_npg_p90": 0.07,
        "sofa_kp_p90": 0.16, "sofa_fthird_p90": 0.11, "sofa_tklw_p90": 0.08,
        "sofa_int_p90": 0.06, "sofa_recov_p90": 0.06, "sofa_pwa3_p90": 0.04,
    },
}

DEF_WEIGHTS = {
    # FBref DF:     TklW .26  Int .22  Blocks .15  Clr .11  DefPen .08  Def3rd .06  PrgP .05
    "DF": {
        "sofa_tklw_p90": 0.26, "sofa_int_p90": 0.22, "sofa_blk_p90": 0.15,
        "sofa_clr_p90": 0.11, "sofa_aerw_p90": 0.08, "sofa_recov_p90": 0.06,
        "sofa_lball_p90": 0.05,
    },
    # FBref Def_MF: TklW .32  Int .28  Blocks .12  Clr .08  DefPen .08  Def3rd .07
    "Def_MF": {
        "sofa_tklw_p90": 0.32, "sofa_int_p90": 0.28, "sofa_blk_p90": 0.12,
        "sofa_clr_p90": 0.08, "sofa_recov_p90": 0.08, "sofa_gdw_p90": 0.07,
    },
}

INTENSITY_WEIGHTS = {
    # FBref:        Carries   PrgDist   Recov   Won(aer)  Tkl+Int
    "FW":     {"sofa_touch_p90": 0.30, "sofa_duelw_p90": 0.30, "sofa_recov_p90": 0.20,
               "sofa_aerw_p90": 0.10, "sofa_tklint_p90": 0.10},
    "Off_MF": {"sofa_touch_p90": 0.27, "sofa_duelw_p90": 0.27, "sofa_recov_p90": 0.25,
               "sofa_tklint_p90": 0.14, "sofa_aerw_p90": 0.07},
    "MF":     {"sofa_recov_p90": 0.25, "sofa_tklint_p90": 0.25, "sofa_touch_p90": 0.20,
               "sofa_duelw_p90": 0.20, "sofa_aerw_p90": 0.10},
    "Def_MF": {"sofa_tklint_p90": 0.35, "sofa_recov_p90": 0.30, "sofa_aerw_p90": 0.15,
               "sofa_touch_p90": 0.12, "sofa_duelw_p90": 0.08},
    "DF":     {"sofa_tklint_p90": 0.30, "sofa_aerw_p90": 0.25, "sofa_recov_p90": 0.25,
               "sofa_touch_p90": 0.10, "sofa_duelw_p90": 0.10},
}

GK_WEIGHTS = {
    "GK": {
        "sofa_gprev_p90": 0.35,   # goals prevented: save quality vs xGOT faced
        "sofa_saves_p90": 0.25,   # save volume
        "sofa_save_pct": 0.15,    # save percentage
        "sofa_claims_p90": 0.10,  # command of the box
        "sofa_pass_pct": 0.10,    # distribution
        "sofa_runsout_p90": 0.05, # sweeping
    },
}

ALL_FEATURES = sorted({f for ws in (
    *OFF_WEIGHTS.values(), *MID_WEIGHTS.values(), *DEF_WEIGHTS.values(),
    *INTENSITY_WEIGHTS.values(), *GK_WEIGHTS.values(),
) for f in ws})


# ------------------------------------------------------------------
# Benchmarks
# ------------------------------------------------------------------

def load_benchmarks() -> dict[str, dict[str, float]]:
    """{score_block: {feature: p95}} from sofa_benchmarks.json."""
    with open(BENCHMARK_PATH) as fh:
        return json.load(fh)


def compute_benchmarks(sofascore_dir: Path) -> dict[str, dict[str, float]]:
    """
    p95 per feature over qualified players of BENCHMARK_SEASONS.
    Per score block, the p95 pool is restricted to the roles the block scores
    (e.g. OFF benchmarks come from FW + Off_MF players), mirroring the
    role-relative spirit of the FBref benchmarks.
    """
    from .processing_sofa import build_season_table

    frames = [build_season_table(s, sofascore_dir) for s in BENCHMARK_SEASONS]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    df = df[df["minutesPlayed"] >= BENCHMARK_MIN_MINUTES]
    df = add_per90_features(df)

    pools = {
        "OFF": df[df["Pos"].isin(["FW", "Off_MF"])],
        "MID": df[df["Pos"] == "MF"],
        "DEF": df[df["Pos"].isin(["DF", "Def_MF"])],
        "INTENSITY": df[df["Pos"] != "GK"],
        "GK": df[df["Pos"] == "GK"],
    }
    block_weights = {
        "OFF": OFF_WEIGHTS, "MID": MID_WEIGHTS, "DEF": DEF_WEIGHTS,
        "INTENSITY": INTENSITY_WEIGHTS, "GK": GK_WEIGHTS,
    }

    out: dict[str, dict[str, float]] = {}
    for block, pool in pools.items():
        feats = sorted({f for ws in block_weights[block].values() for f in ws})
        out[block] = {f: round(float(pool[f].quantile(0.95)), 3) for f in feats}
    return out


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------

def _apply_block(df: pd.DataFrame, weights_by_pos: dict, benchmarks: dict,
                 score_col: str, band_col: str) -> pd.DataFrame:
    frames = []
    for pos, weights in weights_by_pos.items():
        df_pos = df[df["Pos"] == pos]
        if df_pos.empty:
            continue
        scored = compute_score_absolute(
            df_pos, feature_weights=weights, feature_benchmarks=benchmarks,
            score_name=score_col, max_score=1000.0,
        )
        scored[band_col] = scored[score_col].apply(score_band_5)
        frames.append(scored)
    return pd.concat(frames) if frames else df.iloc[0:0]


def compute_all_scores_sofa(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: season table from processing_sofa.build_season_table.
    Returns df with Off/Mid/Def/Intensity/GK scores + bands; players below
    MIN_90S_FOR_SCORING keep NaN scores (same convention as the FBref pipeline).
    """
    bm = load_benchmarks()
    df = add_per90_features(df)

    eligible = df[df["90s"] >= MIN_90S_FOR_SCORING]
    blocks = [
        (OFF_WEIGHTS, bm["OFF"], "OffScore_abs", "OffBand"),
        (MID_WEIGHTS, bm["MID"], "MidScore_abs", "MidBand"),
        (DEF_WEIGHTS, bm["DEF"], "DefScore_abs", "DefBand"),
        (INTENSITY_WEIGHTS, bm["INTENSITY"], "IntensityScore_abs", "IntensityBand"),
        (GK_WEIGHTS, bm["GK"], "GKScore_abs", "GKBand"),
    ]
    for weights_by_pos, benchmarks, score_col, band_col in blocks:
        scored = _apply_block(eligible, weights_by_pos, benchmarks, score_col, band_col)
        df[score_col] = scored[score_col].reindex(df.index)
        df[band_col] = scored[band_col].reindex(df.index)
    return df


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    bm = compute_benchmarks(root / "Data" / "Raw" / "Sofascore")
    with open(BENCHMARK_PATH, "w") as fh:
        json.dump(bm, fh, indent=2)
    print(f"Benchmarks written to {BENCHMARK_PATH}:")
    print(json.dumps(bm, indent=2))
