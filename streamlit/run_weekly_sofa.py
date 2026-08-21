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

# Extra scoring leagues (Tier 1 + Tier 2), scored from 2026-27 on. Built from
# the single source of truth in scraping_sofascore.
from src.scraping_sofascore import EXTRA_SCORING_LEAGUES  # noqa: E402
EXTRA_LEAGUES = ",".join(f"{slug}:{tid}" for slug, (tid, _c) in EXTRA_SCORING_LEAGUES.items())


def _flag(name: str, default: bool = True) -> bool:
    v = os.getenv(name, "").strip().lower()
    return default if not v else v in ("1", "true", "yes", "on")


def current_season() -> str:
    now = datetime.now(timezone.utc)
    year = now.year if now.month >= 8 else now.year - 1
    return f"{year}-{year + 1}"


def prev_season(season: str) -> str:
    start = int(season.split("-")[0])
    return f"{start - 1}-{start}"


def qualified_count(season: str, min_minutes: int = 450) -> int:
    """Players (across all leagues) in `season` with enough minutes to be
    scored (>= 5x90). Calendar-year leagues (MLS, Brazil, Norway) are mature
    mid-year and make the season worth scoring even while the Big-5 are thin."""
    import glob
    import pandas as pd
    total = 0
    for f in glob.glob(str(RAW / f"sofascore_player_stats-*-{season}.csv")):
        try:
            df = pd.read_csv(f, usecols=["minutesPlayed"])
            total += int((df["minutesPlayed"] >= min_minutes).sum())
        except Exception:
            continue
    return total


def season_to_score(season: str) -> str:
    """Score the current season once any league in it has enough data;
    otherwise keep the previous season so the app never shows a near-empty
    just-started season. The previous season's file persists regardless."""
    n = qualified_count(season)
    if n < 80:
        p = prev_season(season)
        print(f"[INFO] {season} has only {n} qualified players (season just started); "
              f"scoring {p} instead", flush=True)
        return p
    return season


def run_module(module: str, *args: str, **extra_env) -> None:
    env = {**os.environ, **{k: str(v) for k, v in extra_env.items()}}
    print(f"\n=== {module} {extra_env or ''}", flush=True)
    subprocess.run([sys.executable, "-m", module, *args],
                   cwd=STREAMLIT_DIR, env=env, check=True)


def run_optional(module: str, *args: str, **extra_env) -> None:
    """Like run_module, but a failure is logged and does not abort the run.

    Used for non-essential steps (shotmaps, heatmaps, profile delta, extra
    leagues). The core chain (season stats -> scores -> app exports) must
    always complete so the website still gets refreshed.
    """
    try:
        run_module(module, *args, **extra_env)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] optional step {module} failed, continuing: {exc}", flush=True)


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
        # 1. Season stats for the Big-5 (essential for the scores)
        run_module("src.scraping_sofascore",
                   FORCE_RESCRAPE="true", SOFA_FIRST_SEASON=season)

        # Everything below is non-essential: a failure must not stop the run
        # from producing fresh scores and publishing the site.

        # 2. Extra leagues for the Tableau package (2. Bundesliga)
        run_optional("src.scraping_sofascore",
                     FORCE_RESCRAPE="true", SOFA_FIRST_SEASON=season,
                     SOFA_LEAGUES=EXTRA_LEAGUES)

        # 3. Shotmaps + heatmaps for the current season (chart data)
        delete_current_season_files(season)
        run_optional("src.scraping_sofascore_shotmaps", SOFA_FIRST_SEASON=season)
        run_optional("src.scraping_sofascore_shotmaps", SOFA_FIRST_SEASON=season,
                     SOFA_LEAGUES=EXTRA_LEAGUES)
        run_optional("src.scraping_sofascore_heatmaps", SOFA_FIRST_SEASON=season)

        # 4. Profile delta (fills ages/roles for new players). Capped per run
        # so a large historical backlog never blocks the pipeline for hours.
        run_optional("src.scraping_sofascore_profiles",
                     PROFILE_MAX=os.getenv("PROFILE_MAX", "400"))

        # 5. Transfermarkt market values
        if _flag("SCRAPE_TRANSFERMARKT"):
            try:
                sys.path.insert(0, str(STREAMLIT_DIR))
                from run_multi_season_pipeline import run_transfermarkt_block
                run_transfermarkt_block()
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Transfermarkt step failed, continuing: {exc}", flush=True)

    if _flag("DO_PROCESS"):
        # Score the current season only once it has enough data; early in a
        # new season keep the previous one as the app's latest.
        score_season = season_to_score(season)

        # Core chain: must succeed so the app gets updated
        run_module("src.build_dob_fallback")
        run_module("src.pipeline_sofa", score_season)

        sys.path.insert(0, str(STREAMLIT_DIR))
        from run_multi_season_pipeline import export_multi_season_tables
        export_multi_season_tables()
        run_module("src.export_sofascore_frontend")

        # Chart exports: non-essential
        run_optional("src.export_shots_frontend")
        run_optional("src.export_heatmaps_frontend")
        run_optional("src.export_gk_shots_frontend")
        run_optional("src.export_tableau_hertha")

    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
