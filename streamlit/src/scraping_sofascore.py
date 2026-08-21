"""Sofascore player season statistics scraper.

Pulls the full detailed player statistics (~106 metrics) for the Big-5
leagues from Sofascore's internal JSON API. Direct HTTP requests are
blocked (403), so all calls run through a Playwright browser context via
page.evaluate(fetch) — same approach as the FBref scrapers.

Output: one CSV per league-season in Data/Raw/Sofascore/
        sofascore_player_stats-{league}-{season}.csv
plus a combined Data/Raw/Sofascore/sofascore_player_stats_all_seasons_long.csv

Usage:
    python -m src.scraping_sofascore            # scrape all missing league-seasons
    FORCE_RESCRAPE=true python -m src.scraping_sofascore
"""

import os
import random
import time

import pandas as pd
from playwright.sync_api import sync_playwright

BASE = "https://www.sofascore.com/api/v1"

LEAGUES = {
    "premier-league": 17,
    "laliga": 8,
    "serie-a": 23,
    "bundesliga": 35,
    "ligue-1": 34,
}

FIRST_SEASON = "2015-2016"

# Extra scoring leagues beyond the Big-5, added from 2026-27 onward.
# slug -> (unique-tournament id, FBref-style Comp label). All are scored
# against the same Big-5 p95 benchmarks. Calendar-year leagues (MLS, Brazil,
# Norway) have their single-year season mapped to 'YYYY-YYYY' by _season_label.
EXTRA_SCORING_LEAGUES = {
    "eredivisie":          (37,  "nl Eredivisie"),
    "primeira-liga":       (238, "pt Primeira Liga"),
    "scottish-prem":       (36,  "sco Premiership"),
    "swiss-super-league":  (215, "ch Super League"),
    "saudi-pro-league":    (955, "sa Pro League"),
    "championship":        (18,  "eng Championship"),
    "2-bundesliga":        (44,  "de 2. Bundesliga"),
    "super-lig":           (52,  "tr Süper Lig"),
    "belgian-pro-league":  (38,  "be Pro League"),
    "austrian-bundesliga": (45,  "at Bundesliga"),
    "mls":                 (242, "us MLS"),
    "eliteserien":         (20,  "no Eliteserien"),
    "brasileirao":         (325, "br Série A"),
}
EXTRA_LEAGUES_FIRST_SEASON = "2026-2027"

# All stat fields exposed by the season-statistics leaderboard endpoint.
STAT_FIELDS = [
    "accurateChippedPasses", "accurateCrosses", "accurateCrossesPercentage",
    "accurateFinalThirdPasses", "accurateLongBalls", "accurateLongBallsPercentage",
    "accurateOppositionHalfPasses", "accurateOwnHalfPasses", "accuratePasses",
    "accuratePassesPercentage", "aerialDuelsWon", "aerialDuelsWonPercentage",
    "aerialLost", "appearances", "assists", "attemptPenaltyMiss",
    "attemptPenaltyPost", "attemptPenaltyTarget", "ballRecovery",
    "bigChancesCreated", "bigChancesMissed", "blockedShots", "cleanSheet",
    "clearances", "crossesNotClaimed", "directRedCards", "dispossessed",
    "dribbledPast", "duelLost", "errorLeadToGoal", "errorLeadToShot",
    "expectedAssists", "expectedGoals", "fouls", "freeKickGoal",
    "goalConversionPercentage", "goalKicks", "goals", "goalsAssistsSum",
    "goalsConceded", "goalsConcededInsideTheBox", "goalsConcededOutsideTheBox",
    "goalsFromInsideTheBox", "goalsFromOutsideTheBox", "goalsPrevented",
    "groundDuelsWon", "groundDuelsWonPercentage", "headedGoals", "highClaims",
    "hitWoodwork", "inaccuratePasses", "interceptions", "keyPasses",
    "leftFootGoals", "matchesStarted", "minutesPlayed", "offsides", "ownGoals",
    "passToAssist", "penaltiesTaken", "penaltyConceded", "penaltyConversion",
    "penaltyFaced", "penaltyGoals", "penaltySave", "penaltyWon",
    "possessionLost", "possessionWonAttThird", "punches", "rating", "redCards",
    "rightFootGoals", "runsOut", "savedShotsFromInsideTheBox",
    "savedShotsFromOutsideTheBox", "saves", "savesCaught", "savesParried",
    "scoringFrequency", "setPieceConversion", "shotFromSetPiece",
    "shotsFromInsideTheBox", "shotsFromOutsideTheBox", "shotsOffTarget",
    "shotsOnTarget", "successfulDribbles", "successfulDribblesPercentage",
    "successfulRunsOut", "tackles", "tacklesWon", "tacklesWonPercentage",
    "totalAttemptAssist", "totalChippedPasses", "totalContest", "totalCross",
    "totalDuelsWon", "totalDuelsWonPercentage", "totalLongBalls",
    "totalOppositionHalfPasses", "totalOwnHalfPasses", "totalPasses",
    "totalShots", "totwAppearances", "touches", "wasFouled", "yellowCards",
    "yellowRedCards",
]

POSITIONS = ["G", "D", "M", "F"]

# uniform seconds between API calls; override e.g. SOFA_DELAY="2.0,4.0" after rate limiting
REQUEST_DELAY = tuple(float(x) for x in os.getenv("SOFA_DELAY", "0.6,1.4").split(","))

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Data", "Raw", "Sofascore",
)


def _season_label(year_str: str) -> str | None:
    """Convert Sofascore year strings to a 'YYYY-YYYY' season label.

    Split-calendar leagues use '15/16' or '2015/2016'. Calendar-year leagues
    (MLS, Brazil, Norway ...) use a single year like '2026'; that year Y is
    mapped to 'Y-(Y+1)' so it co-locates with the current winter season and
    matches the 'YYYY-YYYY' file-name convention used everywhere.
    """
    if "/" not in year_str:
        if year_str.isdigit() and len(year_str) == 4:
            y = int(year_str)
            return f"{y}-{y + 1}"
        return None
    a, b = year_str.split("/")
    if len(a) == 2:
        a = ("19" if int(a) >= 50 else "20") + a
    if len(b) == 2:
        b = ("19" if int(b) >= 50 else "20") + b
    return f"{a}-{b}"


def _wanted_seasons() -> set[str]:
    first_season = os.getenv("SOFA_FIRST_SEASON", FIRST_SEASON)
    first = int(first_season.split("-")[0])
    return {f"{y}-{y + 1}" for y in range(first, 2100)}


def leagues_from_env() -> dict[str, int]:
    """Override target leagues via SOFA_LEAGUES='slug:id,slug:id' (default: Big-5).

    Used for one-off pulls of extra leagues (e.g. '2-bundesliga:44') without
    touching the Big-5 defaults that the weekly pipeline relies on.
    """
    spec = os.getenv("SOFA_LEAGUES", "").strip()
    if not spec:
        return LEAGUES
    out = {}
    for part in spec.split(","):
        slug, tid = part.split(":")
        out[slug.strip()] = int(tid)
    return out


class SofascoreClient:
    """Playwright-backed fetch client for Sofascore's internal API."""

    def __init__(self):
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._page = None
        self._open_page()

    def _open_page(self):
        ctx = self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        )
        self._page = ctx.new_page()
        self._page.goto("https://www.sofascore.com/", wait_until="domcontentloaded", timeout=60000)
        self._page.wait_for_timeout(2500)

    def get_json(self, url: str, retries: int = 4) -> dict | None:
        # Recycle the browser page periodically on very long runs
        self._calls = getattr(self, "_calls", 0) + 1
        if self._calls % 2500 == 0:
            self._page.context.close()
            self._open_page()
        for attempt in range(retries):
            time.sleep(random.uniform(*REQUEST_DELAY))
            try:
                resp = self._page.evaluate(
                    """async (url) => {
                        const r = await fetch(url);
                        try { return {status: r.status, body: await r.json()}; }
                        catch(e) { return {status: r.status, body: null}; }
                    }""",
                    url,
                )
            except Exception as exc:  # page crashed / navigation issue
                print(f"    fetch error ({exc}); reopening page")
                self._open_page()
                continue
            if resp["status"] == 200:
                return resp["body"]
            if resp["status"] == 404:
                return None
            wait = 5 * (attempt + 1)
            print(f"    HTTP {resp['status']} on {url} — retry in {wait}s")
            time.sleep(wait)
        raise RuntimeError(f"Giving up on {url}")

    def close(self):
        self._browser.close()
        self._pw.stop()


def fetch_seasons(client: SofascoreClient, tournament_id: int) -> dict[str, int]:
    """Return {season_label: season_id} for one league, newest first."""
    body = client.get_json(f"{BASE}/unique-tournament/{tournament_id}/seasons")
    out = {}
    for s in body["seasons"]:
        label = _season_label(s["year"])
        if label:
            out[label] = s["id"]
    return out


def fetch_league_season(client: SofascoreClient, tournament_id: int, season_id: int) -> pd.DataFrame:
    """Fetch all players of one league-season with all stat fields, tagged by position group."""
    fields = "%2C".join(STAT_FIELDS)
    rows = {}
    for pos in POSITIONS:
        page_no = 1
        while True:
            offset = (page_no - 1) * 100
            url = (
                f"{BASE}/unique-tournament/{tournament_id}/season/{season_id}/statistics"
                f"?limit=100&offset={offset}&order=-rating&accumulation=total"
                f"&fields={fields}&filters=position.in.{pos}"
            )
            body = client.get_json(url)
            if body is None:
                break
            for res in body.get("results", []):
                player = res.pop("player")
                team = res.pop("team")
                pid = player["id"]
                if pid in rows:
                    continue  # already captured under another position group
                row = {
                    "player_id": pid,
                    "player_name": player.get("name"),
                    "player_slug": player.get("slug"),
                    "team_id": team.get("id"),
                    "team_name": team.get("name"),
                    "position_group": pos,
                }
                row.update({k: v for k, v in res.items() if k in STAT_FIELDS})
                rows[pid] = row
            if page_no >= body.get("pages", 1):
                break
            page_no += 1
    return pd.DataFrame(list(rows.values()))


def main():
    force = os.getenv("FORCE_RESCRAPE", "false").lower() == "true"
    os.makedirs(OUT_DIR, exist_ok=True)
    wanted = _wanted_seasons()

    client = SofascoreClient()
    all_frames = []
    try:
        for league, tid in leagues_from_env().items():
            seasons = fetch_seasons(client, tid)
            targets = sorted(s for s in seasons if s in wanted)
            print(f"== {league}: {len(targets)} seasons ({targets[0]} … {targets[-1]})")
            for season in targets:
                out_path = os.path.join(OUT_DIR, f"sofascore_player_stats-{league}-{season}.csv")
                if os.path.exists(out_path) and not force:
                    print(f"   {season}: exists, skipping")
                    all_frames.append(pd.read_csv(out_path))
                    continue
                df = fetch_league_season(client, tid, seasons[season])
                if df.empty:
                    print(f"   {season}: NO DATA")
                    continue
                df.insert(0, "league", league)
                df.insert(1, "season", season)
                df.to_csv(out_path, index=False)
                print(f"   {season}: {len(df)} players -> {os.path.basename(out_path)}")
                all_frames.append(df)
    finally:
        client.close()

    # Rebuild the combined file from ALL season files on disk (not just this
    # run's leagues), so a partial run never shrinks the combined CSV.
    import glob as _glob
    season_files = sorted(_glob.glob(os.path.join(OUT_DIR, "sofascore_player_stats-*-????-????.csv")))
    if season_files:
        combined = pd.concat((pd.read_csv(f) for f in season_files), ignore_index=True)
        combined_path = os.path.join(OUT_DIR, "sofascore_player_stats_all_seasons_long.csv")
        combined.to_csv(combined_path, index=False)
        print(f"Combined: {len(combined)} rows from {len(season_files)} files -> {combined_path}")


if __name__ == "__main__":
    main()
