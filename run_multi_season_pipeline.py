from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd  # optional, falls du es irgendwo noch brauchst

from src.scraping_fbref_player_stats import run_pipeline_for_season as scrape_player_stats
from src.scraping_fbref_squad_stats import run_pipeline_for_season as scrape_squad_stats
from src.scraping_fbref_big5_table import run_pipeline_for_season as scrape_big5_table

from src.pipeline import run_full_pipeline
from src.multi_season import load_all_seasons, aggregate_player_scores
from src.squad import compute_squad_scores


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = (PROJECT_ROOT / "Data" / "Raw").resolve()
PROCESSED_DIR = (PROJECT_ROOT / "Data" / "Processed").resolve()

DEFAULT_START_SEASON = "2017-2018"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _env_flag(name: str, default: bool = True) -> bool:
    v = os.getenv(name, "")
    if not v.strip():
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _season_str(start_year: int) -> str:
    """e.g. 2017 -> '2017-2018'"""
    return f"{start_year}-{start_year + 1}"


def _current_season_utc(season_start_month: int = 8) -> str:
    """
    Compute current season based on UTC date.
    Football season assumed to start in August by default.
    """
    now = datetime.now(timezone.utc)
    year = now.year
    month = now.month

    if month >= season_start_month:
        return _season_str(year)
    return _season_str(year - 1)


def _generate_seasons_from(start_season: str, end_season: str) -> list[str]:
    """
    start_season/end_season format: '2017-2018'
    returns inclusive list.
    """
    try:
        start_year = int(start_season.split("-")[0])
        end_year = int(end_season.split("-")[0])
    except Exception as e:
        raise ValueError(f"Invalid season format. Expected 'YYYY-YYYY'. Got: {start_season=} {end_season=}") from e

    if end_year < start_year:
        raise ValueError(f"end_season must be >= start_season. Got: {start_season=} {end_season=}")

    return [_season_str(y) for y in range(start_year, end_year + 1)]


def _get_seasons() -> list[str]:
    """
    Priority:
      1) PIPELINE_SEASONS env var (comma-separated)
      2) AUTO seasons from AUTO_SEASON_START (default 2017-2018) up to current season (UTC)
         (enabled by USE_AUTO_SEASONS=1/true)
      3) fallback: only current season
    """
    env_seasons = os.getenv("PIPELINE_SEASONS", "").strip()
    if env_seasons:
        return [s.strip() for s in env_seasons.split(",") if s.strip()]

    use_auto = _env_flag("USE_AUTO_SEASONS", True)
    if use_auto:
        start_season = os.getenv("AUTO_SEASON_START", DEFAULT_START_SEASON).strip() or DEFAULT_START_SEASON
        end_season = _current_season_utc(season_start_month=8)
        return _generate_seasons_from(start_season, end_season)

    # fallback: only current season
    return [_current_season_utc(season_start_month=8)]


# -------------------------------------------------------------------
# Exports
# -------------------------------------------------------------------
def export_multi_season_tables() -> None:
    df_all = load_all_seasons(PROCESSED_DIR)
    if df_all.empty:
        print("[WARN] df_all is empty, skipping exports.")
        return

    # 1) LONG: all players / all seasons
    out_long = PROCESSED_DIR / "player_scores_all_seasons_long.csv"
    df_all.to_csv(out_long, index=False)
    print(f"[SAVE] {out_long}")

    # 2) AGG: aggregated by player
    df_agg = aggregate_player_scores(df_all)
    out_agg = PROCESSED_DIR / "player_scores_agg_by_player.csv"
    df_agg.to_csv(out_agg, index=False)
    print(f"[SAVE] {out_agg}")

    # 3) SQUAD SCORES: all teams / all seasons
    df_squad = compute_squad_scores(df_all)
    out_squad = PROCESSED_DIR / "squad_scores_all_seasons.csv"
    df_squad.to_csv(out_squad, index=False)
    print(f"[SAVE] {out_squad}")

def export_big5_all_seasons() -> None:
    raw_dir = RAW_DIR
    files = sorted(raw_dir.glob("big5_table-*.csv"))
    if not files:
        print("[WARN] No big5_table-*.csv found, skipping.")
        return

    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    # Falls Season nicht drin wäre: df.insert(0, "Season", ...)
    # (hast du ja schon)

    out = PROCESSED_DIR / "big5_table_all_seasons.csv"
    df.to_csv(out, index=False)
    print(f"[SAVE] {out}")


# -------------------------------------------------------------------
# STEP 1: Scraping
# -------------------------------------------------------------------
def run_scraping_block(
    seasons: list[str],
    *,
    scrape_players: bool,
    scrape_squads: bool,
    scrape_big5: bool,
) -> None:
    print("=" * 80)
    print("STEP 1: FBref scraping for all seasons")
    print("=" * 80)
    print(f"Seasons: {', '.join(seasons)}")
    print(f"Scrape players:    {scrape_players}")
    print(f"Scrape squads:     {scrape_squads}")
    print(f"Scrape big5 table: {scrape_big5}")
    print(f"Raw output dir:    {RAW_DIR}")
    print("-" * 80)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for season in seasons:
        print("-" * 80)
        print(f"[SEASON] {season}")
        try:
            if scrape_players:
                scrape_player_stats(season, output_folder=RAW_DIR)

            if scrape_squads:
                scrape_squad_stats(season, output_folder=RAW_DIR)

            if scrape_big5:
                scrape_big5_table(season, output_folder=RAW_DIR)

        except Exception as e:
            # Don't kill the whole run – just skip this season
            print(f"[ERROR] scraping failed for {season}: {e}")
            continue

    print("=" * 80)
    print("Scraping finished for all configured seasons.")
    print("=" * 80)


# -------------------------------------------------------------------
# STEP 2: Processing
# -------------------------------------------------------------------
def run_processing_block(seasons: list[str]) -> None:
    print("=" * 80)
    print("STEP 2: Processing pipeline (Raw -> Processed -> Scores)")
    print("=" * 80)
    print(f"Processed output dir: {PROCESSED_DIR}")
    print("-" * 80)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for season in seasons:
        print("-" * 80)
        print(f"[SEASON] {season}")
        try:
            run_full_pipeline(season, RAW_DIR, PROCESSED_DIR)
        except Exception as e:
            print(f"[ERROR] processing failed for {season}: {e}")
            continue

    export_multi_season_tables()
    export_big5_all_seasons()   


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main() -> None:
    seasons = _get_seasons()

    # scraping flags
    scrape_players = _env_flag("SCRAPE_PLAYERS", True)
    scrape_squads = _env_flag("SCRAPE_SQUADS", True)
    scrape_big5 = _env_flag("SCRAPE_BIG5_TABLE", True)

    # master flags
    do_scrape = _env_flag("DO_SCRAPE", True)
    do_process = _env_flag("DO_PROCESS", True)

    if do_scrape:
        run_scraping_block(
            seasons,
            scrape_players=scrape_players,
            scrape_squads=scrape_squads,
            scrape_big5=scrape_big5,
        )

    if do_process:
        run_processing_block(seasons)

    print("=" * 80)
    print("DONE.")
    print("=" * 80)


if __name__ == "__main__":
    main()