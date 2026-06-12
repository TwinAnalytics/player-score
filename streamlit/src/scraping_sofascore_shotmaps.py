"""Sofascore match shotmap scraper.

For every finished Big-5 match since FIRST_SEASON, pulls the full shotmap:
each shot with pitch coordinates, goal-mouth coordinates, xG, xGOT, body
part, situation and minute. Player-/team-level shotmaps can be aggregated
from this. xG/xGOT only exist from the 2022-23 season onward, which is why
earlier seasons are skipped by default.

Output: Data/Raw/Sofascore/Shotmaps/shotmaps-{league}-{season}.csv
(one row per shot, ~25 shots per match). Resumable per league-season.

Usage:
    python -m src.scraping_sofascore_shotmaps
"""

import os

import pandas as pd

from .scraping_sofascore import BASE, OUT_DIR, SofascoreClient, _wanted_seasons, fetch_seasons, leagues_from_env

FIRST_SEASON = "2022-2023"
SHOT_DIR = os.path.join(OUT_DIR, "Shotmaps")

SHOT_FIELDS = ["isHome", "shotType", "situation", "bodyPart", "goalType",
               "time", "addedTime", "timeSeconds", "xg", "xgot"]


def fetch_event_ids(client: SofascoreClient, tournament_id: int, season_id: int) -> list[dict]:
    """All finished events of one league-season, oldest first."""
    events, page = [], 0
    while True:
        body = client.get_json(f"{BASE}/unique-tournament/{tournament_id}/season/{season_id}/events/last/{page}")
        if body is None:
            break
        for e in body.get("events", []):
            if e.get("status", {}).get("type") != "finished":
                continue
            events.append({
                "event_id": e["id"],
                "start_timestamp": e.get("startTimestamp"),
                "round": e.get("roundInfo", {}).get("round"),
                "home_team": e.get("homeTeam", {}).get("name"),
                "away_team": e.get("awayTeam", {}).get("name"),
                "home_score": e.get("homeScore", {}).get("current"),
                "away_score": e.get("awayScore", {}).get("current"),
            })
        if not body.get("hasNextPage"):
            break
        page += 1
    return events


def fetch_shotmap(client: SofascoreClient, event: dict) -> list[dict]:
    body = client.get_json(f"{BASE}/event/{event['event_id']}/shotmap")
    if body is None:
        return []
    rows = []
    for s in body.get("shotmap", []):
        player = s.get("player", {})
        pc = s.get("playerCoordinates", {})
        gm = s.get("goalMouthCoordinates", {})
        row = dict(event)
        row.update({
            "player_id": player.get("id"),
            "player_name": player.get("name"),
            "x": pc.get("x"),
            "y": pc.get("y"),
            "goal_mouth_x": gm.get("x"),
            "goal_mouth_y": gm.get("y"),
            "goal_mouth_z": gm.get("z"),
            "goal_mouth_location": s.get("goalMouthLocation"),
        })
        row.update({k: s.get(k) for k in SHOT_FIELDS})
        rows.append(row)
    return rows


def main():
    os.makedirs(SHOT_DIR, exist_ok=True)
    wanted = {s for s in _wanted_seasons() if s >= FIRST_SEASON}

    client = SofascoreClient()
    try:
        for league, tid in leagues_from_env().items():
            seasons = fetch_seasons(client, tid)
            targets = sorted(s for s in seasons if s in wanted)
            print(f"== {league}: shotmaps for {targets}", flush=True)
            for season in targets:
                out_path = os.path.join(SHOT_DIR, f"shotmaps-{league}-{season}.csv")
                if os.path.exists(out_path):
                    print(f"   {season}: exists, skipping", flush=True)
                    continue
                events = fetch_event_ids(client, tid, seasons[season])
                rows, failed = [], 0
                for i, ev in enumerate(events, 1):
                    try:
                        rows.extend(fetch_shotmap(client, ev))
                    except RuntimeError:
                        failed += 1
                    if i % 50 == 0:
                        print(f"   {season}: {i}/{len(events)} events", flush=True)
                if not rows:
                    print(f"   {season}: NO SHOT DATA", flush=True)
                    continue
                df = pd.DataFrame(rows)
                df.insert(0, "league", league)
                df.insert(1, "season", season)
                df.to_csv(out_path, index=False)
                print(f"   {season}: {len(df)} shots from {len(events)} events "
                      f"({failed} failed) -> {os.path.basename(out_path)}", flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
