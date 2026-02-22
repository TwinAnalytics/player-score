"""
One-time script: download club crest PNGs from the ESPN public API.

Usage:
    python download_club_crests.py

Downloads to: Data/Assets/club_crests/{slug}.png
Skips files that already exist.
"""
import re
import time
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_FILE = Path("Data/Processed/player_scores_all_seasons_long.csv")
CREST_DIR = Path("Data/Assets/club_crests")

# ESPN league codes covering Big-5 + common promoted/relegated clubs
ESPN_LEAGUES = [
    "eng.1",  # Premier League
    "esp.1",  # La Liga
    "ger.1",  # Bundesliga
    "ita.1",  # Serie A
    "fra.1",  # Ligue 1
    "eng.2",  # Championship
    "ger.2",  # 2. Bundesliga
    "ita.2",  # Serie B
    "fra.2",  # Ligue 2
    "esp.2",  # Segunda División
]

# FBref display name → ESPN displayName (for fuzzy matching)
ALIAS_MAP: dict[str, str] = {
    "Athletic Club":        "Athletic Club",
    "Betis":                "Real Betis",
    "Celta Vigo":           "Celta Vigo",
    "Atlético Madrid":      "Atlético Madrid",
    "Leverkusen":           "Bayer Leverkusen",
    "Schalke 04":           "Schalke 04",
    "Köln":                 "Cologne",
    "Wolves":               "Wolverhampton",
    "Nott'ham Forest":      "Nottingham Forest",
    "Sheffield Utd":        "Sheffield Utd",
    "Tottenham":            "Tottenham Hotspur",
    "Man United":           "Manchester United",
    "Man City":             "Manchester City",
    "Paris S-G":            "Paris Saint-Germain",
    "Internazionale":       "Internazionale",
    "Hellas Verona":        "Hellas Verona",
    "West Brom":            "West Brom",
    "West Ham":             "West Ham",
    "Newcastle Utd":        "Newcastle United",
    "Nantes":               "FC Nantes",
    "Rennes":               "Stade Rennais",
    "Lens":                 "Lens",
    "Strasbourg":           "Strasbourg",
    "Reims":                "Reims",
    "Brest":                "Brest",
    "Lorient":              "Lorient",
    "Dijon":                "Dijon",
    "Angers":               "Angers",
    "Metz":                 "Metz",
    "Amiens":               "Amiens",
    "Nîmes":                "Nîmes",
    "Caen":                 "Caen",
    "Guingamp":             "Guingamp",
    "Troyes":               "Troyes",
    "Clermont Foot":        "Clermont Foot",
    "Ajaccio":              "AC Ajaccio",
    "Paderborn":            "Paderborn",
    "Mainz 05":             "Mainz",
    "Greuther Fürth":       "Greuther Fürth",
    "Hertha BSC":           "Hertha Berlin",
    "Augsburg":             "Augsburg",
    "Gladbach":             "Borussia Mönchengladbach",
    "Hamburger SV":         "Hamburger SV",
    "Darmstadt 98":         "Darmstadt",
    "Heidenheim":           "Heidenheim",
    "Hoffenheim":           "Hoffenheim",
    "Hannover 96":          "Hannover 96",
    "Nuremberg":            "1. FC Nürnberg",
    "Fortuna Düsseldorf":   "Fortuna Düsseldorf",
    "Düsseldorf":           "Fortuna Düsseldorf",
    "Huddersfield":         "Huddersfield",
    "Luton Town":           "Luton Town",
    "Cardiff City":         "Cardiff",
    "Stoke City":           "Stoke City",
    "Hull City":            "Hull City",
    "Ipswich Town":         "Ipswich",
    "Espanyol":             "Espanyol",
    "Rayo Vallecano":       "Rayo Vallecano",
    "Mallorca":             "Real Mallorca",
    "Girona":               "Girona",
    "Getafe":               "Getafe",
    "Villarreal":           "Villarreal",
    "Osasuna":              "Osasuna",
    "Almería":              "Almería",
    "Eibar":                "Eibar",
    "Leganés":              "Leganés",
    "Huesca":               "Huesca",
    "Cádiz":                "Cádiz",
    "Elche":                "Elche",
    "Granada":              "Granada",
    "Levante":              "Levante",
    "Valladolid":           "Valladolid",
    "Alavés":               "Alavés",
    "Udinese":              "Udinese",
    "Sassuolo":             "Sassuolo",
    "Salernitana":          "Salernitana",
    "Empoli":               "Empoli",
    "Spezia":               "Spezia",
    "Cremonese":            "Cremonese",
    "Lecce":                "Lecce",
    "Monza":                "Monza",
    "Venezia":              "Venezia",
    "Benevento":            "Benevento",
    "Crotone":              "Crotone",
    "Brescia":              "Brescia",
    "Chievo":               "Chievo",
    "Frosinone":            "Frosinone",
    "Parma":                "Parma",
    "SPAL":                 "SPAL",
    "Eint Frankfurt":       "Eintracht Frankfurt",
    "Arminia":              "Arminia Bielefeld",
    "Bournemouth":          "AFC Bournemouth",
    "Brentford":            "Brentford",
    "Brighton":             "Brighton & Hove Albion",
    "Burnley":              "Burnley",
    "Leicester City":       "Leicester City",
    "Leeds United":         "Leeds United",
    "Norwich City":         "Norwich City",
    "Watford":              "Watford",
    "Sunderland":           "Sunderland",
    "Swansea City":         "Swansea City",
    "Dortmund":             "Borussia Dortmund",
    "Como":                 "Como",
    "Laval":                "Laval",
    "Rodez":                "Rodez",
    "Auxerre":              "Auxerre",
    "Bordeaux":             "Bordeaux",
    "Bochum":               "Bochum",
    "Norwich City":         "Norwich City",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", name.lower().replace(" ", "_").replace("-", "_").replace(".", ""))


def _normalize(name: str) -> str:
    """Lowercase + remove common suffixes/articles for fuzzy matching."""
    name = name.lower()
    for suffix in ["fc", "cf", "sc", "ac", "as", "ud", "sd", "rcd", "1.", "sv"]:
        name = re.sub(rf"\b{suffix}\b", "", name)
    return re.sub(r"\s+", " ", name).strip()


def fetch_espn_team_logos() -> dict[str, str]:
    """Fetch displayName → logo URL from ESPN for all configured leagues."""
    logo_map: dict[str, str] = {}
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (PlayerScore/1.0)"

    for league_code in ESPN_LEAGUES:
        url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league_code}/teams"
        try:
            r = session.get(url, timeout=15)
            if r.status_code != 200:
                print(f"  ESPN {league_code}: HTTP {r.status_code}")
                continue
            sports = r.json().get("sports", [{}])
            teams = sports[0].get("leagues", [{}])[0].get("teams", [])
            for t in teams:
                team = t.get("team", {})
                dname = team.get("displayName", "")
                logos = team.get("logos", [])
                logo = logos[0].get("href", "") if logos else ""
                if dname and logo:
                    logo_map[dname] = logo
            print(f"  ESPN {league_code}: {len(teams)} teams")
        except Exception as exc:
            print(f"  ESPN {league_code} error: {exc}")
        time.sleep(0.3)

    return logo_map


def _download_png(url: str, dest: Path) -> bool:
    """Download PNG from url to dest. Returns True on success."""
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception as exc:
        print(f"    Download error: {exc}")
        return False


def _find_logo(club: str, logo_map: dict[str, str]) -> str | None:
    """Match an FBref club name to an ESPN logo URL."""
    # 1) Direct match
    if club in logo_map:
        return logo_map[club]

    # 2) Alias match → lookup in logo_map
    alias = ALIAS_MAP.get(club)
    if alias and alias in logo_map:
        return logo_map[alias]

    # 3) Fuzzy normalised match
    norm_club = _normalize(ALIAS_MAP.get(club, club))
    for espn_name, url in logo_map.items():
        if _normalize(espn_name) == norm_club:
            return url

    # 4) Partial contains match (last resort)
    norm_short = norm_club[:8]
    for espn_name, url in logo_map.items():
        if norm_short and norm_short in _normalize(espn_name):
            return url

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    CREST_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        print(f"ERROR: Data file not found: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE, usecols=["Squad"])
    clubs = sorted(df["Squad"].dropna().unique().tolist())
    print(f"Found {len(clubs)} unique clubs.\n")

    print("Fetching ESPN team logo index…")
    logo_map = fetch_espn_team_logos()
    print(f"Total ESPN teams indexed: {len(logo_map)}\n")

    n_downloaded = 0
    n_existing = 0
    n_not_found: list[str] = []

    for club in clubs:
        slug = _slug(club)
        dest = CREST_DIR / f"{slug}.png"

        if dest.exists():
            n_existing += 1
            print(f"  [skip]  {club}")
            continue

        logo_url = _find_logo(club, logo_map)

        if not logo_url:
            print(f"  [miss]  {club}")
            n_not_found.append(club)
            continue

        ok = _download_png(logo_url, dest)
        if ok:
            print(f"  [  ok]  {club}  →  {dest.name}")
            n_downloaded += 1
        else:
            print(f"  [fail]  {club}")
            n_not_found.append(club)

        time.sleep(0.1)  # be gentle with the CDN

    print(f"\n{'─'*50}")
    print(f"Downloaded      : {n_downloaded}")
    print(f"Already existed : {n_existing}")
    print(f"Not found       : {len(n_not_found)}")
    if n_not_found:
        print("  Missing clubs:")
        for c in n_not_found:
            print(f"    - {c}")


if __name__ == "__main__":
    main()
