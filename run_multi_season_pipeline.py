"""
Multi-season pipeline for PlayerScore.

What it does:
1) For each season in SEASONS:
    - scrape FBref *player* stats  -> players_data-<season>.csv
    - scrape FBref *squad*  stats  -> squads_data-<season>.csv
2) Afterwards you can plug in your existing processing:
    - Process Raw -> Processed -> Scores
"""

from __future__ import annotations

from pathlib import Path

from src.scraping_fbref_player_stats import run_pipeline_for_season as scrape_player_stats
from src.scraping_fbref_squad_stats import run_pipeline_for_season as scrape_squad_stats

import os

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = (PROJECT_ROOT / "Data" / "Raw").resolve()
PROCESSED_DIR = (PROJECT_ROOT / "Data" / "Processed").resolve()

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

SEASONS = [
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
    "2025-2026",  # ggf. rausnehmen, falls FBref noch keine Daten hat
]

DEFAULT_SEASONS = [
    "2017-2018","2018-2019","2019-2020","2020-2021","2021-2022",
    "2022-2023","2023-2024","2024-2025","2025-2026"
]

env_seasons = os.getenv("PIPELINE_SEASONS", "").strip()
if env_seasons:
    SEASONS = [s.strip() for s in env_seasons.split(",") if s.strip()]
else:
    SEASONS = DEFAULT_SEASONS
SCRAPE_PLAYERS = True
SCRAPE_SQUADS = True


# -------------------------------------------------------------------
# Main orchestration
# -------------------------------------------------------------------

def run_scraping_block() -> None:
    """
    Loop over all seasons and scrape player + squad stats.
    """
    print("=" * 80)
    print("STEP 1: FBref scraping for all seasons")
    print("=" * 80)
    print(f"Seasons: {', '.join(SEASONS)}")
    print(f"Scrape players: {SCRAPE_PLAYERS}")
    print(f"Scrape squads:  {SCRAPE_SQUADS}")
    print(f"Raw output dir: {RAW_DIR}")
    print("-" * 80)

    for season in SEASONS:
        if SCRAPE_PLAYERS:
            scrape_player_stats(season, output_folder=RAW_DIR)

        if SCRAPE_SQUADS:
            scrape_squad_stats(season, output_folder=RAW_DIR)

    print("=" * 80)
    print("Scraping finished for all configured seasons.")
    print("=" * 80)


def run_processing_block() -> None:
    """
    Hook for your existing multi-season processing pipeline.

    Here you plug in:
    - load players_data-*.csv / squads_data-*.csv from RAW_DIR
    - compute scores, roles, per90, aggregates
    - save to PROCESSED_DIR

    For now, this is just a placeholder.
    """
    print("=" * 80)
    print("STEP 2: Processing pipeline (Raw -> Processed -> Scores)")
    print("=" * 80)
    print(
        "TODO: Insert your processing logic here (from Raw -> Processed -> Scores). "
        "Scraping is already handled above."
    )


def main() -> None:
    run_scraping_block()
    run_processing_block()


if __name__ == "__main__":
    main()