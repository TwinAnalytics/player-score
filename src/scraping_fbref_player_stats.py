"""
Scrapes FBref Big-5 *player* stats for a given season and saves ONE full CSV:

    players_data-<season>.csv

Usage (standalone):
    python scraping_fbref_player_stats.py

Or import in other scripts:
    from scraping_fbref_player_stats import run_pipeline_for_season
    run_pipeline_for_season("2024-2025")
"""

from __future__ import annotations

import random
import time
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional
from bs4 import BeautifulSoup, Comment

import pandas as pd
from playwright.sync_api import (
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

# -------------------------------------------------------------------
# Paths / Project structure
# -------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Where to save raw CSVs
OUTPUT_FOLDER = (PROJECT_ROOT / "Data" / "Raw").resolve()

# Fixed: this script is for PLAYER stats
STATS_LEVEL = "players"

def is_current_season(season: str) -> bool:
    year = time.localtime().tm_year
    month = time.localtime().tm_mon

    if month >= 7:
        current = f"{year}-{year+1}"
    else:
        current = f"{year-1}-{year}"

    return season == current
# -------------------------------------------------------------------
# URL construction
# -------------------------------------------------------------------

def build_urls_for_season(season: str) -> Dict[str, str]:
    """
    Build all FBref URLs (Big-5) for PLAYER stats.
    Handles historical vs current season correctly.
    """

    if is_current_season(season):
        # LIVE season → no season in URL
        base = "https://fbref.com/en/comps/Big5"
        season_tag = "Big-5-European-Leagues-Stats"
    else:
        # Historical season
        base = f"https://fbref.com/en/comps/Big5/{season}"
        season_tag = f"{season}-Big-5-European-Leagues-Stats"

    urls = {
        f"{base}/stats/players/{season_tag}": "stats_standard",
        f"{base}/shooting/players/{season_tag}": "stats_shooting",
        f"{base}/passing/players/{season_tag}": "stats_passing",
        f"{base}/passing_types/players/{season_tag}": "stats_passing_types",
        f"{base}/gca/players/{season_tag}": "stats_gca",
        f"{base}/defense/players/{season_tag}": "stats_defense",
        f"{base}/possession/players/{season_tag}": "stats_possession",
        f"{base}/playingtime/players/{season_tag}": "stats_playing_time",
        f"{base}/misc/players/{season_tag}": "stats_misc",
        f"{base}/keepers/players/{season_tag}": "stats_keeper",
        f"{base}/keepersadv/players/{season_tag}": "stats_keeper_adv",
    }

    return urls


# -------------------------------------------------------------------
# Scraping helpers
# -------------------------------------------------------------------

def _candidate_table_ids(table_id: str) -> List[str]:
    """
    For player tables we mostly keep the original id.
    Kept compatible with squad logic if FBref ever changes something.
    """
    candidates = [table_id]

    # Only special handling for squads; here just future-proofing.
    if STATS_LEVEL == "squads":
        if table_id.startswith("stats_teams_"):
            candidates.append(table_id.replace("teams", "squads", 1))
        elif table_id.startswith("stats_squads_"):
            candidates.append(table_id.replace("squads", "teams", 1))

        if table_id.endswith("_for"):
            base = table_id[:-4]
            if base not in candidates:
                candidates.append(base)

    seen = set()
    unique: List[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def scrape_table(page, url: str, table_id: str) -> Optional[pd.DataFrame]:
    """
    Load an FBref page and parse a single table for the given id.
    Works even if FBref hides tables in HTML comments.
    """
    try:
        page.goto(url, timeout=60000, wait_until="domcontentloaded")
        # Warten bis irgendwas "stabil" ist (statt table#id, weil oft hidden)
        page.wait_for_selector("body", timeout=30000)
        page.wait_for_timeout(1500)

        html = page.content()

        # Try candidate IDs (future-proofing)
        candidate_ids = _candidate_table_ids(table_id)

        df = None
        effective_id = None
        for cid in candidate_ids:
            df_try = extract_fbref_table_from_html(html, cid)
            if df_try is not None and not df_try.empty:
                df = df_try
                effective_id = cid
                break

        if df is None:
            print(f"[WARN] Table with any of ids {candidate_ids} not found (html/comments) on {url}")
            return None

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)

        # Remove duplicated columns
        df = df.loc[:, ~df.columns.duplicated()]

        # Remove header rows that got read as data
        if "Player" in df.columns:
            df = df[df["Player"] != "Player"]
        if "Squad" in df.columns:
            df = df[df["Squad"] != "Squad"]

        print(f"[OK] Retrieved table '{effective_id}'")
        return df

    except Exception as e:
        print(f"[ERROR] retrieving table '{table_id}' from {url}: {e}")
        return None


def scrape_all_tables(season: str) -> Dict[str, pd.DataFrame]:
    """
    Scrapes all FBref tables for the given season and returns
    a dict {table_id: DataFrame}.
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

        # Hide webdriver flag
        page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
        )

        for url, table_id in urls.items():
            print(f"Scraping {table_id} from {url}")
            df = scrape_table(page, url, table_id)
            if df is not None:
                dfs[table_id] = df
            time.sleep(random.uniform(1, 2))  # be nice to FBref

        browser.close()

    return dfs

def extract_fbref_table_from_html(html: str, table_id: str) -> Optional[pd.DataFrame]:
    """
    FBref tables are often hidden inside HTML comments.
    This tries:
      1) direct <table id=...>
      2) comment block inside <div id="all_<table_id>">
    """
    # 1) direct
    try:
        out = pd.read_html(StringIO(html), attrs={"id": table_id})
        if out:
            return out[0]
    except Exception:
        pass

    # 2) inside comments under div#all_<table_id>
    soup = BeautifulSoup(html, "lxml")
    container = soup.find("div", id=f"all_{table_id}")
    if container is None:
        return None

    comments = container.find_all(string=lambda t: isinstance(t, Comment))
    for c in comments:
        if table_id in c:
            try:
                out = pd.read_html(StringIO(c), attrs={"id": table_id})
                if out:
                    return out[0]
            except Exception:
                continue

    return None
# -------------------------------------------------------------------
# Merge / Cleaning
# -------------------------------------------------------------------

def merge_dataframes(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all downloaded player tables on ['Player', 'Squad'] when possible.
    """
    main_key = "stats_standard"
    if main_key not in dfs:
        raise ValueError(f"Missing main table '{main_key}' in scraped data!")

    merged_df = dfs[main_key].copy()

    for name, df in dfs.items():
        if name == main_key:
            continue

        possible_keys = ["Player", "Squad"]
        join_keys = [
            col
            for col in possible_keys
            if col in merged_df.columns and col in df.columns
        ]

        if not join_keys:
            print(
                f"[WARN] No common join keys between main and '{name}'. "
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
    """
    Remove columns that contain 'matches' in their name (FBref artefacts).
    """
    return df.drop(
        columns=[c for c in df.columns if "matches" in c.lower()],
        errors="ignore",
    )


def fix_age_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert FBref 'Age' from 'yy-ddd' to an integer 'yy'.
    Example: '22-150' -> 22
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
        players_data-<season>.csv
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    season_safe = season.replace("/", "-")
    filename = f"players_data-{season_safe}.csv"
    full_path = output_folder / filename

    df_full.to_csv(full_path, index=False)
    print(f"[SAVE] Player full file: {full_path}")


# -------------------------------------------------------------------
# Pipeline entry point
# -------------------------------------------------------------------

def run_pipeline_for_season(
    season: str,
    output_folder: Path | str = OUTPUT_FOLDER,
) -> None:
    """
    High-level pipeline for one season:

    1. Scrape all Big-5 FBref *player* tables for this season
    2. Merge & clean (Age fix, 'matches'-Spalten entfernen)
    3. Save ONE full CSV (no light version)
    """
    print("=" * 80)
    print(f"Starting PLAYER scraping pipeline for season {season}")
    print("=" * 80)
    print(f"Output folder: {output_folder}")

    dfs = scrape_all_tables(season)

    if not dfs:
        raise RuntimeError(
            f"No tables could be loaded for season {season}. "
            "Check URLs or if FBref has data for this season."
        )

    merged_df = merge_dataframes(dfs)
    df_cleaned = remove_unwanted_columns(merged_df)
    df_cleaned = fix_age_format(df_cleaned)

    save_full_csv(df_cleaned, season, output_folder=output_folder)


if __name__ == "__main__":
    # Default standalone run – adjust season here if you like
    DEFAULT_SEASON = "2024-2025"
    run_pipeline_for_season(DEFAULT_SEASON)