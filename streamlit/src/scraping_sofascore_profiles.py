"""Sofascore player profile scraper.

Fetches master data for every unique player in the scraped season stats:
date of birth, detailed positions (DM/MC/AM/ST/...), height, preferred foot,
nationality and proposed market value. One request per player.

The detailed positions drive the role classification (FW / Off_MF / MF /
Def_MF / DF / GK) of the Sofascore-based scoring pipeline; date of birth
replaces the FBref age column.

Output: Data/Raw/Sofascore/player_profiles.csv — appended incrementally,
so an interrupted run resumes where it stopped.

Usage:
    python -m src.scraping_sofascore_profiles
"""

import csv
import glob
import os
from datetime import datetime, timezone

import pandas as pd

from .scraping_sofascore import BASE, OUT_DIR, SofascoreClient

PROFILE_PATH = os.path.join(OUT_DIR, "player_profiles.csv")

FIELDS = ["player_id", "name", "position", "positions_detailed", "date_of_birth",
          "height", "preferred_foot", "country", "market_value_eur",
          "current_team", "retired_or_unknown"]


def all_player_ids() -> pd.DataFrame:
    frames = []
    for path in glob.glob(os.path.join(OUT_DIR, "sofascore_player_stats-*-????-????.csv")):
        frames.append(pd.read_csv(path, usecols=["player_id", "player_name"]))
    df = pd.concat(frames, ignore_index=True).drop_duplicates("player_id")
    return df.sort_values("player_id")


def already_done() -> set[int]:
    if not os.path.exists(PROFILE_PATH):
        return set()
    return set(pd.read_csv(PROFILE_PATH, usecols=["player_id"])["player_id"])


def fetch_profile(client: SofascoreClient, player_id: int) -> dict | None:
    body = client.get_json(f"{BASE}/player/{player_id}")
    if body is None:
        return None
    p = body.get("player", {})
    dob = p.get("dateOfBirthTimestamp")
    return {
        "player_id": player_id,
        "name": p.get("name"),
        "position": p.get("position"),
        "positions_detailed": "|".join(p.get("positionsDetailed") or []),
        "date_of_birth": (
            datetime.fromtimestamp(dob, tz=timezone.utc).date().isoformat() if dob else None
        ),
        "height": p.get("height"),
        "preferred_foot": p.get("preferredFoot"),
        "country": (p.get("country") or {}).get("name"),
        "market_value_eur": p.get("proposedMarketValue"),
        "current_team": (p.get("team") or {}).get("name"),
        "retired_or_unknown": p.get("deceased", False) or p.get("retired", False),
    }


def main():
    players = all_player_ids()
    done = already_done()
    todo = players[~players["player_id"].isin(done)]
    # Optional cap (players per run). The weekly job sets this so a large
    # backlog does not block the pipeline for hours; each run chips away at it.
    cap = os.getenv("PROFILE_MAX")
    if cap and cap.isdigit():
        todo = todo.head(int(cap))
    print(f"{len(players)} players total, {len(done)} done, {len(todo)} to fetch", flush=True)

    write_header = not os.path.exists(PROFILE_PATH)
    client = SofascoreClient()
    fetched = failed = 0
    try:
        with open(PROFILE_PATH, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            if write_header:
                writer.writeheader()
            for i, p in enumerate(todo.itertuples(index=False), 1):
                try:
                    row = fetch_profile(client, p.player_id)
                except RuntimeError:
                    failed += 1
                    continue
                if row is None:
                    # 404 — keep a stub so we don't refetch forever
                    row = {"player_id": p.player_id, "name": p.player_name,
                           "retired_or_unknown": True}
                writer.writerow(row)
                fetched += 1
                if i % 50 == 0:
                    fh.flush()
                if i % 250 == 0:
                    print(f"progress: {i}/{len(todo)} ({failed} failed)", flush=True)
    finally:
        client.close()
    print(f"DONE: {fetched} profiles written, {failed} failed -> {PROFILE_PATH}", flush=True)


if __name__ == "__main__":
    main()
