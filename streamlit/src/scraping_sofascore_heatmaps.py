"""Sofascore player season-heatmap scraper.

For every player in the already-scraped season statistics (Data/Raw/Sofascore/
sofascore_player_stats-*.csv), pulls the season heatmap: ~1400 grid points
(x, y, count) describing where the player touched the ball. One API request
per player per league-season; players without heatmap coverage (404) are
skipped silently and counted.

Output: Data/Raw/Sofascore/Heatmaps/heatmaps-{league}-{season}.csv.gz
(long format: player_id, player_name, x, y, count). Gzipped because a full
league-season is ~600k rows. Resumable per league-season.

Usage:
    python -m src.scraping_sofascore_heatmaps
"""

import glob
import os
import re

import pandas as pd

from .scraping_sofascore import BASE, LEAGUES, OUT_DIR, SofascoreClient, fetch_seasons

FIRST_SEASON = "2017-2018"
HEAT_DIR = os.path.join(OUT_DIR, "Heatmaps")


def stats_files() -> list[tuple[str, str, str]]:
    """[(league, season, path)] of scraped season-stat CSVs >= FIRST_SEASON.

    Only Big-5 leagues (the ones the app uses). Extra leagues such as the
    2. Bundesliga are scraped for the Tableau package but have no tournament
    id in LEAGUES, so heatmaps skip them.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(OUT_DIR, "sofascore_player_stats-*-????-????.csv"))):
        m = re.match(r"sofascore_player_stats-(.+)-(\d{4}-\d{4})\.csv", os.path.basename(path))
        if m and m.group(2) >= FIRST_SEASON and m.group(1) in LEAGUES:
            out.append((m.group(1), m.group(2), path))
    return out


def fetch_player_heatmap(client: SofascoreClient, player_id: int,
                         tournament_id: int, season_id: int) -> list[dict] | None:
    body = client.get_json(
        f"{BASE}/player/{player_id}/unique-tournament/{tournament_id}/season/{season_id}/heatmap/overall"
    )
    if body is None:
        return None
    return body.get("points", [])


def main():
    os.makedirs(HEAT_DIR, exist_ok=True)

    client = SofascoreClient()
    season_ids: dict[str, dict[str, int]] = {}
    try:
        for league, season, stats_path in stats_files():
            out_path = os.path.join(HEAT_DIR, f"heatmaps-{league}-{season}.csv.gz")
            if os.path.exists(out_path):
                print(f"== {league} {season}: exists, skipping", flush=True)
                continue
            if league not in season_ids:
                season_ids[league] = fetch_seasons(client, LEAGUES[league])
            sid = season_ids[league].get(season)
            if sid is None:
                print(f"== {league} {season}: no season id, skipping", flush=True)
                continue

            players = pd.read_csv(stats_path, usecols=["player_id", "player_name"])
            rows, missing, failed = [], 0, 0
            for i, p in enumerate(players.itertuples(index=False), 1):
                try:
                    points = fetch_player_heatmap(client, p.player_id, LEAGUES[league], sid)
                except RuntimeError:
                    failed += 1
                    continue
                if not points:
                    missing += 1
                    continue
                rows.extend(
                    {"player_id": p.player_id, "player_name": p.player_name, **pt}
                    for pt in points
                )
                if i % 100 == 0:
                    print(f"== {league} {season}: {i}/{len(players)} players", flush=True)

            if not rows:
                print(f"== {league} {season}: NO HEATMAP DATA", flush=True)
                continue
            df = pd.DataFrame(rows)
            df.insert(0, "league", league)
            df.insert(1, "season", season)
            df.to_csv(out_path, index=False, compression="gzip")
            print(f"== {league} {season}: {len(players) - missing - failed}/{len(players)} players "
                  f"with heatmap ({missing} missing, {failed} failed) -> {os.path.basename(out_path)}",
                  flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
