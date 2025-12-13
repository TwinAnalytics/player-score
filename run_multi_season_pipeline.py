from __future__ import annotations

import os
from pathlib import Path

from src.scraping_fbref_player_stats import run_pipeline_for_season as scrape_player_stats
from src.scraping_fbref_squad_stats import run_pipeline_for_season as scrape_squad_stats
from src.pipeline import run_full_pipeline


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
        # Erwartet Raw: Data/Raw/players_data-<season>.csv
        run_full_pipeline(season)


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