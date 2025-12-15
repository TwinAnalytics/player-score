from __future__ import annotations

import random
import time
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup, Comment
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_FOLDER = (PROJECT_ROOT / "Data" / "Raw").resolve()

TABLE_ID = "big5_table"


def _extract_table_html_from_comments(html: str, table_id: str) -> Optional[str]:
    """
    FBref packt Tabellen manchmal in HTML-Comments <!-- ... -->.
    Diese Funktion sucht dort nach <table id="..."> und gibt den HTML-Block zurück.
    """
    soup = BeautifulSoup(html, "lxml")

    # 1) Direkt im DOM?
    t = soup.find("table", {"id": table_id})
    if t is not None:
        return str(t)

    # 2) In comments?
    for c in soup.find_all(string=lambda text: isinstance(text, Comment)):
        if table_id in c:
            try:
                cs = BeautifulSoup(c, "lxml")
                t2 = cs.find("table", {"id": table_id})
                if t2 is not None:
                    return str(t2)
            except Exception:
                continue

    return None


def scrape_big5_table(season: str) -> pd.DataFrame:
    """
    Scraped die Big-5 League Table pro Saison:
    https://fbref.com/en/comps/Big5/<season>/<season>-Big-5-European-Leagues-Stats
    Table id: big5_table
    """
    url = f"https://fbref.com/en/comps/Big5/{season}/{season}-Big-5-European-Leagues-Stats"

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-gpu",
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

        page.goto(url, timeout=0, wait_until="load")
        # kleine Wartezeit, damit FBref alles rendert
        time.sleep(random.uniform(1.0, 2.0))

        html = page.content()
        browser.close()

    table_html = _extract_table_html_from_comments(html, TABLE_ID)
    if table_html is None:
        raise RuntimeError(f"[Big5] Table id '{TABLE_ID}' not found on {url}")

    df = pd.read_html(StringIO(table_html), attrs={"id": TABLE_ID})[0]

    df.insert(0, "Season", season)

    # MultiIndex flatten, falls vorhanden
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)

    # Headerzeilen raus (falls doppelt eingelesen)
    if "Squad" in df.columns:
        df = df[df["Squad"] != "Squad"]

    df = df.loc[:, ~df.columns.duplicated()]
    return df


def save_big5_table(df: pd.DataFrame, season: str, output_folder: Path | str = OUTPUT_FOLDER) -> Path:
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    season_safe = season.replace("/", "-")
    out = output_folder / f"big5_table-{season_safe}.csv"
    df.to_csv(out, index=False)
    print(f"[SAVE] Big5 table: {out}")
    return out


def run_pipeline_for_season(season: str, output_folder: Path | str = OUTPUT_FOLDER) -> None:
    print("=" * 80)
    print(f"Starting BIG5 TABLE scraping for season {season}")
    print("=" * 80)

    df = scrape_big5_table(season)
    save_big5_table(df, season, output_folder=output_folder)


if __name__ == "__main__":
    import sys
    season = sys.argv[1] if len(sys.argv) > 1 else "2024-2025"
    run_pipeline_for_season(season)