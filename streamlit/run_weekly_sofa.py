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
import time
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
EXTRA_SLUGS = list(EXTRA_SCORING_LEAGUES.keys())

# Wall-clock budget for the whole scraping phase. The essential Big-5 stats
# always run; every optional step below is skipped once the budget is spent, so
# scoring + exports + the website publish ALWAYS finish inside the job timeout.
# Before this guard existed, a slow optional step (extra-league shotmaps, ~4 h)
# could eat the entire 5 h cap and the run was cancelled before it ever
# committed — throwing away even the ~30 min of good stats work.
_START = time.monotonic()
SCRAPE_BUDGET_MIN = int(os.getenv("SOFA_SCRAPE_BUDGET_MIN", "180"))


def over_budget(label: str) -> bool:
    used = (time.monotonic() - _START) / 60
    if used >= SCRAPE_BUDGET_MIN:
        print(f"[BUDGET] {used:.0f} min used (cap {SCRAPE_BUDGET_MIN}) — skipping "
              f"{label} so scoring + publish still finish; last run's files are kept",
              flush=True)
        return True
    return False


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


def delete_season_files(season: str, slugs) -> None:
    """Delete current-season chart files for the given leagues so the scraper
    re-fetches them (it skips any league-season whose file already exists).

    Scoped per league on purpose: leagues we are NOT about to rescrape (because
    the budget ran out) keep last run's files, so their charts stay stale rather
    than disappearing."""
    for slug in slugs:
        for path in (
            RAW / "Shotmaps" / f"shotmaps-{slug}-{season}.csv",
            RAW / "Heatmaps" / f"heatmaps-{slug}-{season}.csv.gz",
        ):
            if path.exists():
                path.unlink()
                print(f"refresh: removed {path.name}")


def main() -> None:
    season = os.getenv("PIPELINE_SEASON", current_season())
    print(f"Weekly Sofascore run for {season}")

    if _flag("DO_SCRAPE"):
        # 1. Season stats for the Big-5 (ESSENTIAL for the scores). ~30 min.
        #    Always runs — never gated by the budget.
        run_module("src.scraping_sofascore",
                   FORCE_RESCRAPE="true", SOFA_FIRST_SEASON=season)

        # Everything below is non-essential and wall-clock-budgeted: a failure
        # must not stop the run from producing fresh scores and publishing the
        # site, and neither must a slow step. Ordered most to least important so
        # the budget, if it runs out, drops only the least essential work. The
        # slow extra-league shotmaps (~4 h) deliberately run LAST.

        # 2. Extra-league season stats (2. Bundesliga + Tier-2 scores)
        if not over_budget("extra-league stats"):
            run_optional("src.scraping_sofascore",
                         FORCE_RESCRAPE="true", SOFA_FIRST_SEASON=season,
                         SOFA_LEAGUES=EXTRA_LEAGUES)

        # 3. Big-5 shotmaps + heatmaps (chart data, ~45 min)
        if not over_budget("Big-5 shotmaps/heatmaps"):
            delete_season_files(season, BIG5_SLUGS)
            run_optional("src.scraping_sofascore_shotmaps", SOFA_FIRST_SEASON=season)
            run_optional("src.scraping_sofascore_heatmaps", SOFA_FIRST_SEASON=season)

        # 4. Profile delta (fills ages/roles for new players). Capped per run
        # so a large historical backlog never blocks the pipeline for hours.
        if not over_budget("player profiles"):
            run_optional("src.scraping_sofascore_profiles",
                         PROFILE_MAX=os.getenv("PROFILE_MAX", "400"))

        # 5. Transfermarkt market values
        if _flag("SCRAPE_TRANSFERMARKT") and not over_budget("Transfermarkt"):
            try:
                sys.path.insert(0, str(STREAMLIT_DIR))
                from run_multi_season_pipeline import run_transfermarkt_block
                run_transfermarkt_block()
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Transfermarkt step failed, continuing: {exc}", flush=True)

        # 6. Extra-league shotmaps (Tableau package; SLOW, ~4 h). Last on
        # purpose: when the budget runs out here, last run's files stay in place
        # (the scraper skips existing files) so those charts are stale, never
        # empty — and scoring + publish below still run.
        if not over_budget("extra-league shotmaps"):
            delete_season_files(season, EXTRA_SLUGS)
            run_optional("src.scraping_sofascore_shotmaps", SOFA_FIRST_SEASON=season,
                         SOFA_LEAGUES=EXTRA_LEAGUES)

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
