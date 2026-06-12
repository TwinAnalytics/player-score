"""
API-Football Explorer – Freie Daten abrufen und erkunden
=========================================================
Free Plan: 100 Requests/Tag, alle Endpoints, alle Ligen.
Doku: https://www.api-football.com/documentation-v3
"""

import requests
import json
from datetime import date
from pathlib import Path

# ============================================================
# HIER DEINEN API-KEY EINSETZEN
# ============================================================
API_KEY = "a7ae7b0eb7664e2e01f4b55365a07326"
# ============================================================

BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def api_call(endpoint: str, params: dict = None) -> dict:
    url = f"{BASE_URL}/{endpoint}"
    response = requests.get(url, headers=HEADERS, params=params or {})
    data = response.json()
    remaining = response.headers.get("x-ratelimit-requests-remaining", "?")
    print(f"  → Remaining requests today: {remaining}")
    if data.get("errors"):
        print(f"  ⚠ Fehler: {data['errors']}")
    return data


def save_json(data: dict, filename: str):
    path = DATA_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  → Gespeichert: data/{filename}")


def show_json(data: dict, max_items: int = 3):
    results = data.get("response", [])
    print(f"  → {len(results)} Ergebnisse gesamt\n")
    for item in results[:max_items]:
        print(json.dumps(item, indent=2, ensure_ascii=False))
        print("  ---")


def check_status():
    print("\n" + "=" * 60)
    print("1. ACCOUNT STATUS")
    print("=" * 60)
    data = api_call("status")
    save_json(data, "01_status.json")
    account = data.get("response", {}).get("account", {})
    requests_info = data.get("response", {}).get("requests", {})
    sub = data.get("response", {}).get("subscription", {})
    print(f"  Plan:       {sub.get('plan', '?')}")
    print(f"  Endet:      {sub.get('end', '?')}")
    print(f"  Requests:   {requests_info.get('current', '?')} / {requests_info.get('limit_day', '?')} heute")


def get_leagues():
    print("\n" + "=" * 60)
    print("2. BUNDESLIGA META")
    print("=" * 60)
    data = api_call("leagues", {"id": 78})
    save_json(data, "02_bundesliga_league.json")
    show_json(data, max_items=1)


def get_standings():
    print("\n" + "=" * 60)
    print("3. BUNDESLIGA TABELLE 2024/25")
    print("=" * 60)
    data = api_call("standings", {"league": 78, "season": 2024})
    save_json(data, "03_bundesliga_standings.json")
    standings = data.get("response", [])
    if standings:
        table = standings[0].get("league", {}).get("standings", [[]])[0]
        print(f"  {'#':<4} {'Team':<30} {'Sp':>3} {'S':>3} {'U':>3} {'N':>3} {'Tore':>7} {'Pkt':>4}")
        print("  " + "-" * 60)
        for row in table:
            team = row["team"]["name"]
            stats = row["all"]
            goals = f"{stats['goals']['for']}:{stats['goals']['against']}"
            print(f"  {row['rank']:<4} {team:<30} {stats['played']:>3} {stats['win']:>3} {stats['draw']:>3} {stats['lose']:>3} {goals:>7} {row['points']:>4}")


def get_fixtures():
    print("\n" + "=" * 60)
    print("4. HEUTIGE SPIELE")
    print("=" * 60)
    today = date.today().isoformat()
    data = api_call("fixtures", {"date": today})
    save_json(data, "04_fixtures_today.json")
    results = data.get("response", [])
    print(f"  → {len(results)} Spiele heute\n")
    for fix in results[:10]:
        league = fix["league"]["name"]
        home = fix["teams"]["home"]["name"]
        away = fix["teams"]["away"]["name"]
        status = fix["fixture"]["status"]["short"]
        goals_h = fix["goals"]["home"]
        goals_a = fix["goals"]["away"]
        score = f"{goals_h}:{goals_a}" if goals_h is not None else "–:–"
        print(f"  [{league:<25}] {home:<25} {score:^5} {away:<25} ({status})")


def get_top_scorers():
    print("\n" + "=" * 60)
    print("5. TOP-TORSCHÜTZEN BUNDESLIGA 2024/25")
    print("=" * 60)
    data = api_call("players/topscorers", {"league": 78, "season": 2024})
    save_json(data, "05_bundesliga_topscorers.json")
    results = data.get("response", [])
    print(f"  {'#':<4} {'Spieler':<25} {'Team':<20} {'Tore':>5} {'Assists':>8}")
    print("  " + "-" * 65)
    for i, item in enumerate(results[:15], 1):
        player = item["player"]["name"]
        stats = item["statistics"][0]
        team = stats["team"]["name"]
        goals = stats["goals"]["total"] or 0
        assists = stats["goals"]["assists"] or 0
        print(f"  {i:<4} {player:<25} {team:<20} {goals:>5} {assists:>8}")


def get_player_stats():
    print("\n" + "=" * 60)
    print("6. SPIELER-STATISTIKEN (Lewandowski ID=1100)")
    print("=" * 60)
    data = api_call("players", {"id": 1100, "season": 2024})
    save_json(data, "06_player_lewandowski.json")
    show_json(data, max_items=1)


def get_team_stats():
    print("\n" + "=" * 60)
    print("7. TEAM-STATISTIKEN (FC Bayern, Bundesliga 2024/25)")
    print("=" * 60)
    data = api_call("teams/statistics", {"league": 78, "season": 2024, "team": 157})
    save_json(data, "07_fcbayern_stats.json")
    stats = data.get("response", {})
    if stats:
        print(f"  Team:     {stats.get('team', {}).get('name', '?')}")
        fixtures = stats.get("fixtures", {})
        print(f"  Spiele:   {fixtures.get('played', {}).get('total', '?')}")
        print(f"  Siege:    {fixtures.get('wins', {}).get('total', '?')}")
        goals_for = stats.get("goals", {}).get("for", {}).get("total", {}).get("total", "?")
        goals_ag = stats.get("goals", {}).get("against", {}).get("total", {}).get("total", "?")
        print(f"  Tore:     {goals_for}:{goals_ag}")


def get_last_fixtures():
    print("\n" + "=" * 60)
    print("8. LETZTE 5 BUNDESLIGA-SPIELE (inkl. Events/Stats)")
    print("=" * 60)
    data = api_call("fixtures", {"league": 78, "season": 2024, "last": 5})
    save_json(data, "08_bundesliga_last5.json")
    results = data.get("response", [])
    for fix in results:
        home = fix["teams"]["home"]["name"]
        away = fix["teams"]["away"]["name"]
        score = f"{fix['goals']['home']}:{fix['goals']['away']}"
        print(f"  {home} {score} {away}")


# ============================================================
# HAUPTPROGRAMM
# ============================================================
if __name__ == "__main__":
    if API_KEY == "DEIN_API_KEY_HIER":
        print("❌ Bitte setze deinen API-Key oben ein!")
        print("   Dashboard: https://dashboard.api-football.com → Account → My Access")
        exit(1)

    print("🔍 API-Football Explorer")
    print(f"   Daten werden in: {DATA_DIR}")
    print("   Free Plan: 100 Requests/Tag\n")

    check_status()        # 1 Request
    get_leagues()         # 1 Request
    get_standings()       # 1 Request
    get_fixtures()        # 1 Request
    get_top_scorers()     # 1 Request
    get_player_stats()    # 1 Request
    get_team_stats()      # 1 Request
    get_last_fixtures()   # 1 Request

    print("\n" + "=" * 60)
    print("FERTIG! 8 von 100 Requests verbraucht.")
    print(f"JSON-Rohdaten findest du in: {DATA_DIR}")
    print("=" * 60)
    print("\nWichtige IDs:")
    print("  Bundesliga: 78  |  Premier League: 39  |  La Liga: 140")
    print("  Serie A: 135    |  Ligue 1: 61         |  Champions League: 2")
    print("  2. Bundesliga: 79  |  DFB Pokal: 81")
    print("  FC Bayern: 157  |  Borussia Dortmund: 165  |  Hertha BSC: 168")
