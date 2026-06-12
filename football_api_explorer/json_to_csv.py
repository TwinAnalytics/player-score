"""
JSON → CSV Konverter für API-Football Daten
"""

import json
import csv
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def load(filename):
    with open(DATA_DIR / filename, encoding="utf-8") as f:
        return json.load(f).get("response", {})


# ── 1. Bundesliga Tabelle ────────────────────────────────────────────────────
def convert_standings():
    response = load("03_bundesliga_standings.json")
    table = response[0]["league"]["standings"][0]
    rows = []
    for r in table:
        rows.append({
            "rank":           r["rank"],
            "team_id":        r["team"]["id"],
            "team":           r["team"]["name"],
            "played":         r["all"]["played"],
            "wins":           r["all"]["win"],
            "draws":          r["all"]["draw"],
            "losses":         r["all"]["lose"],
            "goals_for":      r["all"]["goals"]["for"],
            "goals_against":  r["all"]["goals"]["against"],
            "goal_diff":      r["goalsDiff"],
            "points":         r["points"],
            "form":           r["form"],
            "description":    r.get("description", ""),
        })
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "03_bundesliga_standings.csv", index=False)
    print(f"✓ Standings: {len(df)} Zeilen")


# ── 2. Heutige Spiele ────────────────────────────────────────────────────────
def convert_fixtures():
    response = load("04_fixtures_today.json")
    rows = []
    for f in response:
        fix = f["fixture"]
        rows.append({
            "fixture_id":    fix["id"],
            "date":          fix["date"],
            "status":        fix["status"]["short"],
            "status_long":   fix["status"]["long"],
            "elapsed":       fix["status"].get("elapsed"),
            "league_id":     f["league"]["id"],
            "league":        f["league"]["name"],
            "country":       f["league"]["country"],
            "season":        f["league"]["season"],
            "round":         f["league"]["round"],
            "home_team_id":  f["teams"]["home"]["id"],
            "home_team":     f["teams"]["home"]["name"],
            "away_team_id":  f["teams"]["away"]["id"],
            "away_team":     f["teams"]["away"]["name"],
            "goals_home":    f["goals"]["home"],
            "goals_away":    f["goals"]["away"],
            "ht_home":       f["score"]["halftime"]["home"],
            "ht_away":       f["score"]["halftime"]["away"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "04_fixtures_today.csv", index=False)
    print(f"✓ Fixtures today: {len(df)} Zeilen")


# ── 3. Top-Torschützen ───────────────────────────────────────────────────────
def convert_topscorers():
    response = load("05_bundesliga_topscorers.json")
    rows = []
    for item in response:
        p = item["player"]
        s = item["statistics"][0]
        rows.append({
            "player_id":        p["id"],
            "name":             p["name"],
            "firstname":        p["firstname"],
            "lastname":         p["lastname"],
            "age":              p["age"],
            "nationality":      p["nationality"],
            "height_cm":        p["height"],
            "weight_kg":        p["weight"],
            "team_id":          s["team"]["id"],
            "team":             s["team"]["name"],
            "appearances":      s["games"]["appearences"],
            "minutes":          s["games"]["minutes"],
            "rating":           s["games"]["rating"],
            "goals":            s["goals"]["total"],
            "assists":          s["goals"]["assists"],
            "shots_total":      s["shots"]["total"],
            "shots_on":         s["shots"]["on"],
            "passes_key":       s["passes"]["key"],
            "passes_accuracy":  s["passes"]["accuracy"],
            "dribbles_success": s["dribbles"]["success"],
            "dribbles_attempts":s["dribbles"]["attempts"],
            "fouls_drawn":      s["fouls"]["drawn"],
            "fouls_committed":  s["fouls"]["committed"],
            "yellow_cards":     s["cards"]["yellow"],
            "red_cards":        s["cards"]["red"],
            "penalty_scored":   s["penalty"]["scored"],
            "penalty_missed":   s["penalty"]["missed"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "05_bundesliga_topscorers.csv", index=False)
    print(f"✓ Top Scorers: {len(df)} Zeilen")


# ── 4. Spieler-Detail (Haaland) ──────────────────────────────────────────────
def convert_player():
    response = load("06_player_lewandowski.json")
    p = response[0]["player"]
    rows = []
    for s in response[0]["statistics"]:
        rows.append({
            "player_id":         p["id"],
            "name":              p["name"],
            "age":               p["age"],
            "nationality":       p["nationality"],
            "height_cm":         p["height"],
            "weight_kg":         p["weight"],
            "team":              s["team"]["name"],
            "league":            s["league"]["name"],
            "league_country":    s["league"]["country"],
            "season":            s["league"]["season"],
            "appearances":       s["games"]["appearences"],
            "lineups":           s["games"]["lineups"],
            "minutes":           s["games"]["minutes"],
            "position":          s["games"]["position"],
            "rating":            s["games"]["rating"],
            "goals":             s["goals"]["total"],
            "assists":           s["goals"]["assists"],
            "shots_total":       s["shots"]["total"],
            "shots_on":          s["shots"]["on"],
            "passes_total":      s["passes"]["total"],
            "passes_key":        s["passes"]["key"],
            "passes_accuracy":   s["passes"]["accuracy"],
            "tackles":           s["tackles"]["total"],
            "interceptions":     s["tackles"]["interceptions"],
            "duels_total":       s["duels"]["total"],
            "duels_won":         s["duels"]["won"],
            "dribbles_attempts": s["dribbles"]["attempts"],
            "dribbles_success":  s["dribbles"]["success"],
            "fouls_drawn":       s["fouls"]["drawn"],
            "fouls_committed":   s["fouls"]["committed"],
            "yellow_cards":      s["cards"]["yellow"],
            "red_cards":         s["cards"]["red"],
            "penalty_scored":    s["penalty"]["scored"],
            "penalty_missed":    s["penalty"]["missed"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "06_player_haaland.csv", index=False)
    print(f"✓ Player (Haaland): {len(df)} Zeilen (eine pro Liga/Wettbewerb)")


# ── 5. Team-Statistiken FC Bayern ────────────────────────────────────────────
def convert_team_stats():
    response = load("07_fcbayern_stats.json")
    s = response

    def goals_row(direction, period, value):
        return {"direction": direction, "period": period, "value": value}

    # Flache Hauptstatistiken
    fixtures = s["fixtures"]
    goals = s["goals"]
    biggest = s["biggest"]

    main = [{
        "team":                     s["team"]["name"],
        "league":                   s["league"]["name"],
        "season":                   s["league"]["season"],
        "played_home":              fixtures["played"]["home"],
        "played_away":              fixtures["played"]["away"],
        "played_total":             fixtures["played"]["total"],
        "wins_home":                fixtures["wins"]["home"],
        "wins_away":                fixtures["wins"]["away"],
        "wins_total":               fixtures["wins"]["total"],
        "draws_home":               fixtures["draws"]["home"],
        "draws_away":               fixtures["draws"]["away"],
        "draws_total":              fixtures["draws"]["total"],
        "loses_home":               fixtures["loses"]["home"],
        "loses_away":               fixtures["loses"]["away"],
        "loses_total":              fixtures["loses"]["total"],
        "goals_for_total":          goals["for"]["total"]["total"],
        "goals_for_home":           goals["for"]["total"]["home"],
        "goals_for_away":           goals["for"]["total"]["away"],
        "goals_for_avg_total":      goals["for"]["average"]["total"],
        "goals_against_total":      goals["against"]["total"]["total"],
        "goals_against_home":       goals["against"]["total"]["home"],
        "goals_against_away":       goals["against"]["total"]["away"],
        "goals_against_avg_total":  goals["against"]["average"]["total"],
        "biggest_win_home":         biggest["wins"]["home"],
        "biggest_win_away":         biggest["wins"]["away"],
        "biggest_loss_home":        biggest["loses"]["home"],
        "biggest_loss_away":        biggest["loses"]["away"],
        "clean_sheet_total":        s["clean_sheet"]["total"],
        "failed_to_score_total":    s["failed_to_score"]["total"],
        "form":                     s["form"],
    }]
    pd.DataFrame(main).to_csv(DATA_DIR / "07_fcbayern_stats.csv", index=False)

    # Goals per minute (Verteilung)
    minute_rows = []
    for minute_range, val in goals["for"]["minute"].items():
        if val and val.get("total") is not None:
            minute_rows.append({
                "minute_range": minute_range,
                "goals_for":    val["total"],
                "pct_for":      val["percentage"],
            })
    ag = goals["against"]["minute"]
    for i, (minute_range, val) in enumerate(ag.items()):
        if val and val.get("total") is not None and i < len(minute_rows):
            minute_rows[i]["goals_against"] = val["total"]
            minute_rows[i]["pct_against"]   = val["percentage"]

    pd.DataFrame(minute_rows).to_csv(DATA_DIR / "07_fcbayern_goals_per_minute.csv", index=False)
    print(f"✓ Team Stats (Bayern): Haupttabelle + Goals-per-Minute")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Konvertiere JSON → CSV ...\n")
    convert_standings()
    convert_fixtures()
    convert_topscorers()
    convert_player()
    convert_team_stats()
    print(f"\nAlle CSVs in: {DATA_DIR}")
