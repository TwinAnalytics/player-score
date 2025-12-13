from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

from src.scraping_fbref_player_stats import run_pipeline_for_season as scrape_player_stats
from src.scraping_fbref_squad_stats import run_pipeline_for_season as scrape_squad_stats
from src.pipeline import run_full_pipeline
from src.multi_season import load_all_seasons, aggregate_player_scores
from src.squad import compute_squad_scores



PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = (PROJECT_ROOT / "Data" / "Raw").resolve()
PROCESSED_DIR = (PROJECT_ROOT / "Data" / "Processed").resolve()

DEFAULT_SEASONS = [
    "2017-2018", "2018-2019", "2019-2020", "2020-2021", "2021-2022",
    "2022-2023", "2023-2024", "2024-2025", "2025-2026",
]


def _env_flag(name: str, default: bool = True) -> bool:
    v = os.getenv(name, "")
    if not v.strip():
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _get_seasons() -> list[str]:
    env_seasons = os.getenv("PIPELINE_SEASONS", "").strip()
    if env_seasons:
        return [s.strip() for s in env_seasons.split(",") if s.strip()]
    return DEFAULT_SEASONS


def run_scraping_block(seasons: list[str], *, scrape_players: bool, scrape_squads: bool) -> None:
    print("=" * 80)
    print("STEP 1: FBref scraping for all seasons")
    print("=" * 80)
    print(f"Seasons: {', '.join(seasons)}")
    print(f"Scrape players: {scrape_players}")
    print(f"Scrape squads:  {scrape_squads}")
    print(f"Raw output dir: {RAW_DIR}")
    print("-" * 80)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for season in seasons:
        if scrape_players:
            scrape_player_stats(season, output_folder=RAW_DIR)
        if scrape_squads:
            scrape_squad_stats(season, output_folder=RAW_DIR)

    print("=" * 80)
    print("Scraping finished for all configured seasons.")
    print("=" * 80)


def run_processing_block(seasons: list[str]) -> None:
    print("=" * 80)
    print("STEP 2: Processing pipeline (Raw -> Processed -> Scores)")
    print("=" * 80)
    print(f"Processed output dir: {PROCESSED_DIR}")
    print("-" * 80)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for season in seasons:
        run_full_pipeline(season, RAW_DIR, PROCESSED_DIR)

    export_multi_season_tables()

def export_multi_season_tables() -> None:
    df_all = load_all_seasons(PROCESSED_DIR)
    if df_all.empty:
        print("[WARN] df_all is empty, skipping exports.")
        return

    # 1) LONG: alle Spieler, alle Saisons
    out_long = PROCESSED_DIR / "player_scores_all_seasons_long.csv"
    df_all.to_csv(out_long, index=False)
    print(f"[SAVE] {out_long}")

    # 2) AGG: je Spieler aggregiert
    df_agg = aggregate_player_scores(df_all)
    out_agg = PROCESSED_DIR / "player_scores_agg_by_player.csv"
    df_agg.to_csv(out_agg, index=False)
    print(f"[SAVE] {out_agg}")

    # 3) SQUAD SCORES: alle Teams, alle Saisons
    df_squad = compute_squad_scores(df_all)
    out_squad = PROCESSED_DIR / "squad_scores_all_seasons.csv"
    df_squad.to_csv(out_squad, index=False)
    print(f"[SAVE] {out_squad}")

def main() -> None:
    seasons = _get_seasons()
    scrape_players = _env_flag("SCRAPE_PLAYERS", True)
    scrape_squads = _env_flag("SCRAPE_SQUADS", True)
    do_scrape = _env_flag("DO_SCRAPE", True)
    do_process = _env_flag("DO_PROCESS", True)

    if do_scrape:
        run_scraping_block(seasons, scrape_players=scrape_players, scrape_squads=scrape_squads)

    if do_process:
        run_processing_block(seasons)

    print("=" * 80)
    print("DONE.")
    print("=" * 80)


if __name__ == "__main__":
    main()