"""
Scrapes FBref Big-5 *squad/team* stats for a given season and saves ONE full CSV:

    squads_data-<season>.csv

Usage (standalone):
    python scraping_fbref_squad_stats.py

Or import in other scripts:
    from scraping_fbref_squad_stats import run_pipeline_for_season
    run_pipeline_for_season("2024-2025")
"""

from __future__ import annotations

import random
import time
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from playwright.sync_api import (
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_FOLDER = (PROJECT_ROOT / "Data" / "Raw").resolve()

# This scraper is fixed to squad-level stats
STATS_LEVEL = "squads"


# -------------------------------------------------------------------
# URL construction
# -------------------------------------------------------------------

def build_urls_for_season(season: str) -> Dict[str, str]:
    """
    Build all FBref URLs (Big-5) for a given season for SQUAD stats.
    """
    base_url = f"https://fbref.com/en/comps/Big5/{season}"
    season_tag = f"{season}-Big-5-European-Leagues-Stats"

    urls = {
        f"{base_url}/stats/squads/{season_tag}": "stats_teams_standard_for",
        f"{base_url}/shooting/squads/{season_tag}": "stats_teams_shooting_for",
        f"{base_url}/passing/squads/{season_tag}": "stats_teams_passing_for",
        f"{base_url}/passing_types/squads/{season_tag}": "stats_teams_passing_types_for",
        f"{base_url}/gca/squads/{season_tag}": "stats_teams_gca_for",
        f"{base_url}/defense/squads/{season_tag}": "stats_teams_defense_for",
        f"{base_url}/possession/squads/{season_tag}": "stats_teams_possession_for",
        f"{base_url}/playingtime/squads/{season_tag}": "stats_teams_playing_time_for",
        f"{base_url}/misc/squads/{season_tag}": "stats_teams_misc_for",
        f"{base_url}/keepers/squads/{season_tag}": "stats_teams_keeper_for",
        f"{base_url}/keepersadv/squads/{season_tag}": "stats_teams_keeper_adv_for",
    }
    return urls


# -------------------------------------------------------------------
# Scraping helpers
# -------------------------------------------------------------------

def _candidate_table_ids(table_id: str) -> List[str]:
    """
    For squads FBref sometimes flips 'teams'/'squads' and adds/removes '_for'.
    We generate a small set of candidates and try them.
    """
    candidates = [table_id]

    if STATS_LEVEL == "squads":
        if table_id.startswith("stats_teams_"):
            candidates.append(table_id.replace("teams", "squads", 1))
        elif table_id.startswith("stats_squads_"):
            candidates.append(table_id.replace("squads", "teams", 1))

        if table_id.endswith("_for"):
            base = table_id[:-4]
            if base not in candidates:
                candidates.append(base)

    # Deduplicate
    seen = set()
    unique: List[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def scrape_table(page, url: str, table_id: str) -> Optional[pd.DataFrame]:
    """
    Load an FBref page and parse a single squad table.
    """
    try:
        page.goto(url, timeout=0, wait_until="load")

        candidate_ids = _candidate_table_ids(table_id)
        effective_id = None

        for cid in candidate_ids:
            try:
                page.wait_for_selector(f"table#{cid}", timeout=20000)
                effective_id = cid
                break
            except TimeoutError:
                continue
            except PlaywrightTimeoutError:
                continue

        if effective_id is None:
            print(f"[WARN] Squad table with any of ids {candidate_ids} not found on {url}")
            return None

        html = page.content()
        df = pd.read_html(StringIO(html), attrs={"id": effective_id})[0]

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)

        df = df.loc[:, ~df.columns.duplicated()]

        if "Squad" in df.columns:
            df = df[df["Squad"] != "Squad"]

        print(f"[OK] Retrieved squad table '{effective_id}'")
        return df

    except Exception as e:
        print(f"[ERROR] retrieving squad table '{table_id}' from {url}: {e}")
        return None


def scrape_all_tables(season: str) -> Dict[str, pd.DataFrame]:
    """
    Scrapes all FBref squad tables for the given season.
    """
    urls = build_urls_for_season(season)
    dfs: Dict[str, pd.DataFrame] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--no-sandbox",
                "--no-zygote",
                "--disable-infobars",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
                "--window-size=1920,1080",
            ],
        )

        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/117.0 Safari/537.36"
            ),
            viewport={"width": 1920, "height": 1080},
        )

        page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
        )

        for url, table_id in urls.items():
            print(f"Scraping squad table {table_id} from {url}")
            df = scrape_table(page, url, table_id)
            if df is not None:
                dfs[table_id] = df
            time.sleep(random.uniform(1, 2))

        browser.close()

    return dfs


# -------------------------------------------------------------------
# Merge / Cleaning
# -------------------------------------------------------------------

def merge_dataframes(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all squad tables. For squads we merge on 'Squad' only.
    """
    main_key = "stats_teams_standard_for"
    if main_key not in dfs:
        # Fallback if FBref changed naming
        if "stats_teams_standard" in dfs:
            main_key = "stats_teams_standard"
        else:
            raise ValueError(f"Missing main squad table '{main_key}' in scraped data!")

    merged_df = dfs[main_key].copy()

    for name, df in dfs.items():
        if name == main_key:
            continue

        possible_keys = ["Squad"]
        join_keys = [
            col
            for col in possible_keys
            if col in merged_df.columns and col in df.columns
        ]

        if not join_keys:
            print(
                f"[WARN] No common join keys between main squad table and '{name}'. "
                "Skipping merge for this table."
            )
            continue

        merged_df = merged_df.merge(
            df,
            on=join_keys,
            how="left",
            suffixes=("", f"_{name}"),
        )

    return merged_df


def remove_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(
        columns=[c for c in df.columns if "matches" in c.lower()],
        errors="ignore",
    )


def fix_age_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    For squads there is often no 'Age', but we keep this to be robust
    if FBref ever adds such a column.
    """
    if "Age" in df.columns:
        df["Age"] = df["Age"].astype(str).str.split("-").str[0]
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    return df


# -------------------------------------------------------------------
# Saving
# -------------------------------------------------------------------

def save_full_csv(
    df_full: pd.DataFrame,
    season: str,
    output_folder: Path | str = OUTPUT_FOLDER,
) -> None:
    """
    Save ONE full CSV per season:
        squads_data-<season>.csv
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    season_safe = season.replace("/", "-")
    filename = f"squads_data-{season_safe}.csv"
    full_path = output_folder / filename

    df_full.to_csv(full_path, index=False)
    print(f"[SAVE] Squad full file: {full_path}")


# -------------------------------------------------------------------
# Pipeline entry point
# -------------------------------------------------------------------

def run_pipeline_for_season(
    season: str,
    output_folder: Path | str = OUTPUT_FOLDER,
) -> None:
    """
    High-level pipeline for SQUAD stats:

    1. Scrape all Big-5 FBref squad tables for this season
    2. Merge & clean
    3. Save ONE full CSV (no light version)
    """
    print("=" * 80)
    print(f"Starting SQUAD scraping pipeline for season {season}")
    print("=" * 80)
    print(f"Output folder: {output_folder}")

    dfs = scrape_all_tables(season)
    if not dfs:
        raise RuntimeError(
            f"No squad tables could be loaded for season {season}. "
            "Check URLs or if FBref has squad data for this season."
        )

    merged_df = merge_dataframes(dfs)
    df_cleaned = remove_unwanted_columns(merged_df)
    df_cleaned = fix_age_format(df_cleaned)

    save_full_csv(df_cleaned, season, output_folder=output_folder)


if __name__ == "__main__":
    DEFAULT_SEASON = "2024-2025"
    run_pipeline_for_season(DEFAULT_SEASON)