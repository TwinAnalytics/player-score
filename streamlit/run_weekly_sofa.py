"""
Weekly pipeline, Sofascore era (replaces the FBref-based run_multi_season_pipeline
for the recurring job; the FBref history 2017-18 to 2024-25 stays frozen).

Steps, all for the CURRENT season only:
  1. Season stats scrape, Big-5 + 2. Bundesliga (forced refresh)
  2. Shotmaps rescrape (delete current-season files first)
  3. Heatmaps rescrape (delete current-season files first; also fills any
     older league-seasons that are still missing)
  4. Player profiles delta (new players only, resume logic)
  5. Transfermarkt market values (unchanged from the FBref era)
  6. Birth-date fallback, scoring pipeline, all exports

Env flags: DO_SCRAPE / DO_PROCESS (default true), SOFA_DELAY for throttling.
The workflow then commits Data/Processed and syncs the website CSVs into the
playerscore-web repo.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

STREAMLIT_DIR = Path(__file__).resolve().parent
ROOT = STREAMLIT_DIR.parent
RAW = ROOT / "Data" / "Raw" / "Sofascore"
PROCESSED = ROOT / "Data" / "Processed"

BIG5_SLUGS = ["premier-league", "laliga", "bundesliga", "serie-a", "ligue-1"]
EXTRA_LEAGUES = "2-bundesliga:44"  # Tableau raw data, not part of PlayerScore


def _flag(name: str, default: bool = True) -> bool:
    v = os.getenv(name, "").strip().lower()
    return default if not v else v in ("1", "true", "yes", "on")


def current_season() -> str:
    now = datetime.now(timezone.utc)
    year = now.year if now.month >= 8 else now.year - 1
    return f"{year}-{year + 1}"


def run_module(module: str, *args: str, **extra_env) -> None:
    env = {**os.environ, **{k: str(v) for k, v in extra_env.items()}}
    print(f"\n=== {module} {extra_env or ''}", flush=True)
    subprocess.run([sys.executable, "-m", module, *args],
                   cwd=STREAMLIT_DIR, env=env, check=True)


def delete_current_season_files(season: str) -> None:
    patterns = [
        RAW / "Shotmaps" / f"shotmaps-*-{season}.csv",
        RAW / "Heatmaps" / f"heatmaps-*-{season}.csv.gz",
    ]
    import glob
    for pattern in patterns:
        for path in glob.glob(str(pattern)):
            os.remove(path)
            print(f"refresh: removed {Path(path).name}")


def main() -> None:
    season = os.getenv("PIPELINE_SEASON", current_season())
    print(f"Weekly Sofascore run for {season}")

    if _flag("DO_SCRAPE"):
        # 1. Season stats (forced, current season, Big-5 + 2. Bundesliga)
        run_module("src.scraping_sofascore",
                   FORCE_RESCRAPE="true", SOFA_FIRST_SEASON=season)
        run_module("src.scraping_sofascore",
                   FORCE_RESCRAPE="true", SOFA_FIRST_SEASON=season,
                   SOFA_LEAGUES=EXTRA_LEAGUES)

        # 2./3. Shotmaps + heatmaps for the current season
        delete_current_season_files(season)
        run_module("src.scraping_sofascore_shotmaps", SOFA_FIRST_SEASON=season)
        run_module("src.scraping_sofascore_shotmaps", SOFA_FIRST_SEASON=season,
                   SOFA_LEAGUES=EXTRA_LEAGUES)
        run_module("src.scraping_sofascore_heatmaps")

        # 4. Profile delta (new players only)
        run_module("src.scraping_sofascore_profiles")

        # 5. Transfermarkt market values (works as in the FBref era)
        if _flag("SCRAPE_TRANSFERMARKT"):
            sys.path.insert(0, str(STREAMLIT_DIR))
            from run_multi_season_pipeline import run_transfermarkt_block
            run_transfermarkt_block()

    if _flag("DO_PROCESS"):
        run_module("src.build_dob_fallback")
        run_module("src.pipeline_sofa", season)

        sys.path.insert(0, str(STREAMLIT_DIR))
        from run_multi_season_pipeline import export_multi_season_tables
        export_multi_season_tables()

        run_module("src.export_sofascore_frontend")
        run_module("src.export_shots_frontend")
        run_module("src.export_heatmaps_frontend")
        run_module("src.export_gk_shots_frontend")
        run_module("src.export_tableau_hertha")

    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
