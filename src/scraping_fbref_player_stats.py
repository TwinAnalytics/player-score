import os
import json
import time
import random
from io import StringIO
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# Ordner, in dem dieses Script liegt (z.B. .../player_score/src)
SCRIPT_DIR = Path(__file__).resolve().parent
# Projektwurzel (z.B. .../player_score)
PROJECT_ROOT = SCRIPT_DIR.parent

# Optional: Kaggle-Import (nur nötig, wenn du den auskommentierten Upload-Teil
# wirklich wieder aktivieren willst)
try:
    from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
except ImportError:
    KaggleApi = None

# ============================================================
# KONFIGURATION
# ============================================================

# Saison, die du scrapen willst (z.B. "2023-2024", "2024-2025", "2025-2026", ...)
SEASON = [
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
    "2025-2026",
]

# Dein Kaggle-Username (aktuell nicht genutzt, aber zur Vollständigkeit gelassen)
KAGGLE_USERNAME = "fancho"

# EIN gemeinsames Kaggle-Dataset für alle Saisons (nur für den auskommentierten Teil relevant)
DATASET_NAME = "fancho/football-player-stats"

# Ordner, in dem die finalen CSV-Dateien gespeichert werden
# -> .../Portfolio/player_score/Data/Raw
OUTPUT_FOLDER = (PROJECT_ROOT / "Data" / "Raw").resolve()

# "players" oder "squads"
# - "players": individuelle Spieler-Statistiken
# - "squads": Team-/Squad-Statistiken
STATS_LEVEL = "players"  

# FBref-Basis-URLs für die gewählte Saison
BASE_URL = f"https://fbref.com/en/comps/Big5/{SEASON}"
SEASON_TAG = f"{SEASON}-Big-5-European-Leagues-Stats"

# URLS + Tabellen-IDs hängen vom Level ab
if STATS_LEVEL == "players":
    URLS = {
        f"{BASE_URL}/stats/players/{SEASON_TAG}": "stats_standard",
        f"{BASE_URL}/shooting/players/{SEASON_TAG}": "stats_shooting",
        f"{BASE_URL}/passing/players/{SEASON_TAG}": "stats_passing",
        f"{BASE_URL}/passing_types/players/{SEASON_TAG}": "stats_passing_types",
        f"{BASE_URL}/gca/players/{SEASON_TAG}": "stats_gca",
        f"{BASE_URL}/defense/players/{SEASON_TAG}": "stats_defense",
        f"{BASE_URL}/possession/players/{SEASON_TAG}": "stats_possession",
        f"{BASE_URL}/playingtime/players/{SEASON_TAG}": "stats_playing_time",
        f"{BASE_URL}/misc/players/{SEASON_TAG}": "stats_misc",
        f"{BASE_URL}/keepers/players/{SEASON_TAG}": "stats_keeper",
        f"{BASE_URL}/keepersadv/players/{SEASON_TAG}": "stats_keeper_adv",
    }
elif STATS_LEVEL == "squads":
    # Squad-/Team-Tabellen: FBref nutzt IDs mit "teams"
    URLS = {
        f"{BASE_URL}/stats/squads/{SEASON_TAG}": "stats_teams_standard_for",
        f"{BASE_URL}/shooting/squads/{SEASON_TAG}": "stats_teams_shooting_for",
        f"{BASE_URL}/passing/squads/{SEASON_TAG}": "stats_teams_passing_for",
        f"{BASE_URL}/passing_types/squads/{SEASON_TAG}": "stats_teams_passing_types_for",
        f"{BASE_URL}/gca/squads/{SEASON_TAG}": "stats_teams_gca_for",
        f"{BASE_URL}/defense/squads/{SEASON_TAG}": "stats_teams_defense_for",
        f"{BASE_URL}/possession/squads/{SEASON_TAG}": "stats_teams_possession_for",
        f"{BASE_URL}/playingtime/squads/{SEASON_TAG}": "stats_teams_playing_time_for",
        f"{BASE_URL}/misc/squads/{SEASON_TAG}": "stats_teams_misc_for",
        f"{BASE_URL}/keepers/squads/{SEASON_TAG}": "stats_teams_keeper_for",
        f"{BASE_URL}/keepersadv/squads/{SEASON_TAG}": "stats_teams_keeper_adv_for",
    }
else:
    raise ValueError("STATS_LEVEL must be 'players' or 'squads'")


# ============================================================
# KAGGLE AUTH (aktuell unbenutzt, kann bleiben für später)
# ============================================================

def authenticate_kaggle():
    """
    Meldet sich mit ~/.kaggle/kaggle.json bei Kaggle an.
    Wird aktuell nicht verwendet, da der Upload-Teil deaktiviert ist.
    """
    if KaggleApi is None:
        raise ImportError(
            "KaggleApi ist nicht installiert. "
            "Installiere das Paket 'kaggle', falls du den Upload wieder aktivieren möchtest."
        )
    api = KaggleApi()
    api.authenticate()
    print("Kaggle API authentication successful!")
    return api


# ============================================================
# SCRAPING-FUNKTIONEN
# ============================================================

def _candidate_table_ids(table_id: str):
    """
    Liefert mögliche Varianten der table-id für Squad-Tabellen,
    z.B. stats_teams_shooting_for -> auch stats_squads_shooting_for, stats_teams_shooting, ...
    """
    candidates = [table_id]

    if STATS_LEVEL == "squads":
        if table_id.startswith("stats_teams_"):
            candidates.append(table_id.replace("teams", "squads", 1))
        elif table_id.startswith("stats_squads_"):
            candidates.append(table_id.replace("squads", "teams", 1))

        if table_id.endswith("_for"):
            base = table_id[:-4]
            if base not in candidates:
                candidates.append(base)

    # Duplikate entfernen, Reihenfolge beibehalten
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def scrape_table(page, url, table_id):
    """
    Lädt eine FBref-Seite und liest eine Tabelle mit bestimmter ID aus.
    Für Squad-Tabellen werden ggf. mehrere mögliche IDs ausprobiert.
    """
    try:
        page.goto(url, timeout=0, wait_until="load")

        candidate_ids = _candidate_table_ids(table_id)
        effective_id = None

        for cid in candidate_ids:
            try:
                page.wait_for_selector(f"table#{cid}", timeout=20000)
                effective_id = cid
                break
            except PlaywrightTimeoutError:
                continue

        if effective_id is None:
            print(f"Table with any of ids {candidate_ids} not found. Skipping this table.")
            return None

        html = page.content()

        # Tabelle mit pandas aus dem HTML ziehen
        df = pd.read_html(StringIO(html), attrs={"id": effective_id})[0]

        # MultiIndex-Spalten (mehrere Kopfzeilen) vereinfachen
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)

        # doppelte Spaltennamen entfernen
        df = df.loc[:, ~df.columns.duplicated()]

        # Kopfzeilen-Zeilen entfernen (falls Player- oder Squad-Spalte sich selbst enthält)
        if "Player" in df.columns:
            df = df[df["Player"] != "Player"]
        if "Squad" in df.columns:
            df = df[df["Squad"] != "Squad"]

        print(f"Retrieved: {effective_id}")
        return df

    except Exception as e:
        print(f"Error retrieving {table_id}: {e}")
        return None


def scrape_all_tables():
    """
    Ruft alle FBref-Tabellen für die Saison SEASON ab
    und gibt ein Dict {table_id: DataFrame} zurück.
    """
    dfs = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--no-sandbox",
                "--no-zygote",
                "--disable-infobars",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
                "--window-size=1920,1080",
            ]
        )

        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/117.0 Safari/537.36"
            ),
            viewport={"width": 1920, "height": 1080}
        )

        # "Versteckt", dass es ein automatisierter Browser ist
        page.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        """)

        for url, table_id in URLS.items():
            print(f"Scraping {table_id} from {url}")
            df = scrape_table(page, url, table_id)
            if df is not None:
                dfs[table_id] = df
            # kleine Pause, um FBref nicht zu stressen
            time.sleep(random.uniform(1, 2))

        browser.close()

    return dfs


# ============================================================
# DATEN MERGEN / BEREINIGEN
# ============================================================

def merge_dataframes(dfs):
    """
    Merged alle Tabellen.

    - Für Player-Stats wird auf ["Player", "Squad"] gemerged (wenn vorhanden).
    - Für Squad-Stats (ohne Player-Spalte) wird z.B. nur auf ["Squad"] gemerged.
    """
    # Haupttabelle je nach STATS_LEVEL
    if STATS_LEVEL == "players":
        main_key = "stats_standard"
    else:  # squads
        main_key = "stats_teams_standard_for"

    if main_key not in dfs:
        # Falls es doch ohne _for heißen sollte:
        if STATS_LEVEL == "squads" and "stats_teams_standard" in dfs:
            main_key = "stats_teams_standard"
        else:
            raise ValueError(f"Missing main table '{main_key}'!")

    merged_df = dfs[main_key].copy()

    for name, df in dfs.items():
        if name == main_key:
            continue

        # gemeinsame Join-Keys bestimmen
        possible_keys = ["Player", "Squad"]
        join_keys = [col for col in possible_keys if col in merged_df.columns and col in df.columns]

        if not join_keys:
            print(f"Warning: no common join keys for table '{name}'. Skipping merge for this table.")
            continue

        merged_df = merged_df.merge(
            df,
            on=join_keys,
            how="left",
            suffixes=("", f"_{name}"),
        )

    return merged_df


def remove_unwanted_columns(df):
    """Entfernt Spalten, deren Name 'matches' enthält."""
    return df.drop(
        columns=[col for col in df.columns if "matches" in col.lower()],
        errors="ignore",
    )


def fix_age_format(df):
    """
    Konvertiert 'Age' von 'yy-ddd' nach nur 'yy' (nur Jahre).
    Beispiel: '22-150' -> 22
    """
    if "Age" in df.columns:
        df["Age"] = df["Age"].astype(str).str.split("-").str[0]
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    return df


# ============================================================
# SPEICHERN (KAGGLE-UPLOAD AUSKOMMENTIERT)
# ============================================================

def upload_dataset(df_full, df_light, output_folder=OUTPUT_FOLDER):
    """
    Speichert CSVs lokal im angegebenen Ordner.
    Der Ordner wird bei Bedarf automatisch erstellt.
    Der frühere Kaggle-Upload-Teil ist unten als Kommentar erhalten.
    """
    output_folder = Path(output_folder)

    # Ordner automatisch anlegen (inkl. Elternordner)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Dateinamen für diese Saison – abhängig von STATS_LEVEL
    season_safe = SEASON.replace("/", "-")
    prefix = "players" if STATS_LEVEL == "players" else "squads"

    full_filename = f"{prefix}_data-{season_safe}.csv"
    light_filename = f"{prefix}_data_light-{season_safe}.csv"

    full_path = output_folder / full_filename
    light_path = output_folder / light_filename

    df_full.to_csv(full_path, index=False)
    df_light.to_csv(light_path, index=False)

    print(f"Saved full file to:  {full_path}")
    print(f"Saved light file to: {light_path}")

    # ---------------------------------------------------------
    # EHEMALIGER KAGGLE-UPLOAD-TEIL (AKTUELL DEAKTIVIERT)
    # ---------------------------------------------------------
    #
    # COLUMN_DESCRIPTIONS = {
    #     "Rk": "Ranking of the player",
    #     "Player": "Name of the player",
    #     "Nation": "Nationality of the player",
    #     "Pos": "Position on the field",
    #     "Squad": "Team name",
    #     "Comp": "League competition",
    #     "Age": "Player's age in years",
    #     "Born": "Year of birth",
    #     "MP": "Matches played",
    #     "Starts": "Number of matches started",
    #     "Min": "Total minutes played",
    #     "90s": "Minutes played divided by 90 (full match equivalent)",
    #     "Gls": "Total goals scored",
    #     "Ast": "Total assists",
    #     "G+A": "Total goals and assists",
    #     "G-PK": "Goals excluding penalties",
    #     "PK": "Penalty kicks scored",
    #     "PKatt": "Penalty kick attempts",
    #     "CrdY": "Yellow cards received",
    #     "CrdR": "Red cards received",
    #     "xG": "Expected goals",
    #     "npxG": "Non-penalty expected goals",
    #     "xAG": "Expected assists",
    #     "npxG+xAG": "Sum of non-penalty xG and xAG",
    #     "PrgC": "Progressive carries",
    #     "PrgP": "Progressive passes",
    #     "PrgR": "Progressive runs",
    #     "Sh": "Total shots attempted",
    #     "SoT": "Shots on target",
    #     "SoT%": "Percentage of shots on target",
    #     "Sh/90": "Shots per 90 minutes",
    #     "SoT/90": "Shots on target per 90 minutes",
    #     "G/Sh": "Goals per shot",
    #     "G/SoT": "Goals per shot on target",
    #     "Dist": "Average shot distance (yards)",
    #     "FK": "Free kicks taken",
    # }
    #
    # column_metadata = [
    #     {"name": col, "description": desc}
    #     for col, desc in COLUMN_DESCRIPTIONS.items()
    # ]
    #
    # metadata = {
    #     "title": "Football Player Stats",
    #     "id": DATASET_NAME,  # z.B. "fancho/football-player-stats"
    #     "licenses": [{"name": "CC0-1.0"}],
    #     "columns": column_metadata,
    #     "files": [
    #         {
    #             "name": full_filename,
    #             "description": f"Complete dataset with all player statistics for the {SEASON} season.",
    #         },
    #         {
    #             "name": light_filename,
    #             "description": f"Lighter version of the dataset containing only key statistics for the {SEASON} season.",
    #         },
    #     ],
    # }
    #
    # metadata_path = output_folder / "dataset-metadata.json"
    # with open(metadata_path, "w") as f:
    #     json.dump(metadata, f, indent=4)
    #
    # print(f"Uploading new dataset version to Kaggle dataset '{DATASET_NAME}'...")
    #
    # api.dataset_create_version(
    #     output_folder,
    #     version_notes="Added / updated season " + SEASON + " (full + light CSV).",
    #     delete_old_versions=False,
    # )
    #
    # print("New dataset version has been published on Kaggle!")


# ============================================================
# PIPELINE
# ============================================================

def run_pipeline(output_folder=OUTPUT_FOLDER):
    """
    Startet Scraping, Cleaning und speichert die Daten für die Saison SEASON lokal.
    """
    print(f"Starting data scraping for season {SEASON} with STATS_LEVEL='{STATS_LEVEL}'...")
    print(f"OUTPUT_FOLDER: {output_folder}")

    dfs = scrape_all_tables()

    if not dfs:
        raise RuntimeError(
            "Keine Tabellen konnten geladen werden – prüfe URLs, STATS_LEVEL "
            "oder ob die Saison schon Daten hat."
        )

    merged_df = merge_dataframes(dfs)
    df_cleaned = remove_unwanted_columns(merged_df)
    df_cleaned_fixed_age = fix_age_format(df_cleaned)

    # Spalten, die in der "light"-Version enthalten sein sollen
    keep_columns = [
        "Rk", "Player", "Nation", "Pos", "Squad", "Comp", "Age", "Born", "MP", "Starts", "Min", "90s",
        "Gls", "Ast", "G+A", "G-PK", "PK", "PKatt", "CrdY", "CrdR",
        "xG", "npxG", "xAG", "npxG+xAG", "G+A-PK", "xG+xAG",
        "PrgC", "PrgP", "PrgR",
        "Sh", "SoT", "SoT%", "Sh/90", "SoT/90", "G/Sh", "G/SoT", "Dist", "FK",
        "PK_stats_shooting", "PKatt_stats_shooting", "xG_stats_shooting", "npxG_stats_shooting",
        "npxG/Sh", "G-xG", "np:G-xG",
        "Cmp", "Att", "Cmp%", "TotDist", "PrgDist", "Ast_stats_passing", "xAG_stats_passing", "xA", "A-xAG",
        "KP", "1/3", "PPA", "CrsPA", "PrgP_stats_passing",
        "Live", "Dead", "FK_stats_passing_types", "TB", "Sw", "Crs", "TI", "CK", "In", "Out", "Str",
        "Cmp_stats_passing_types",
        "Tkl", "TklW", "Def 3rd", "Mid 3rd", "Att 3rd", "Att_stats_defense", "Tkl%", "Lost",
        "Blocks_stats_defense", "Sh_stats_defense", "Pass", "Int", "Tkl+Int", "Clr", "Err",
        "SCA", "SCA90", "PassLive", "PassDead", "TO", "Sh_stats_gca", "Fld", "Def", "GCA", "GCA90",
        "Touches", "Def Pen", "Def 3rd_stats_possession", "Mid 3rd_stats_possession",
        "Att 3rd_stats_possession", "Att Pen",
        "Live_stats_possession", "Att_stats_possession", "Succ", "Succ%", "Tkld", "Tkld%", "Carries",
        "TotDist_stats_possession", "PrgDist_stats_possession", "PrgC_stats_possession",
        "1/3_stats_possession", "CPA",
        "Mis", "Dis", "Rec", "PrgR_stats_possession",
        "CrdY_stats_misc", "CrdR_stats_misc", "2CrdY", "Fls", "Fld_stats_misc", "Off_stats_misc",
        "Crs_stats_misc",
        "Int_stats_misc", "TklW_stats_misc", "PKwon", "PKcon", "OG", "Recov", "Won",
        "Lost_stats_misc", "Won%",
        "GA", "GA90", "SoTA", "Saves", "Save%", "W", "D", "L", "CS", "CS%", "PKatt_stats_keeper",
        "PKA", "PKsv", "PKm",
        "PSxG", "PSxG/SoT", "PSxG+/-", "/90", "Cmp_stats_keeper_adv", "Att_stats_keeper_adv",
        "Cmp%_stats_keeper_adv",
        "Att (GK)", "Thr", "Launch%", "AvgLen", "Opp", "Stp", "Stp%", "#OPA", "#OPA/90", "AvgDist",
    ]

    # robust: nur Spalten nehmen, die es wirklich gibt (damit kein KeyError kommt)
    existing_keep_columns = [c for c in keep_columns if c in df_cleaned_fixed_age.columns]

    df_light = df_cleaned_fixed_age[existing_keep_columns].copy()
    upload_dataset(df_cleaned_fixed_age, df_light, output_folder=output_folder)


# ============================================================
# SCRIPT-START
# ============================================================

if __name__ == "__main__":
    # Optional kannst du den Ausgabeordner hier überschreiben:
    # run_pipeline(output_folder=PROJECT_ROOT / "Data" / "AndereRaw")

    run_pipeline()  # nutzt OUTPUT_FOLDER und STATS_LEVEL von oben

def run_scraping_for_season(season: str, output_folder: Path | None = None):
    """
    Setzt SEASON & URL-Konfiguration dynamisch und startet dann run_pipeline.
    So kannst du von außen z.B. run_scraping_for_season("2023-2024", raw_dir) aufrufen.
    """
    global SEASON, BASE_URL, SEASON_TAG, URLS, OUTPUT_FOLDER

    # Saison setzen
    SEASON = season

    # Output-Ordner ggf. überschreiben
    if output_folder is not None:
        OUTPUT_FOLDER = Path(output_folder).resolve()

    # URLs neu aufbauen
    BASE_URL = f"https://fbref.com/en/comps/Big5/{SEASON}"
    SEASON_TAG = f"{SEASON}-Big-5-European-Leagues-Stats"

    if STATS_LEVEL == "players":
        URLS = {
            f"{BASE_URL}/stats/players/{SEASON_TAG}": "stats_standard",
            f"{BASE_URL}/shooting/players/{SEASON_TAG}": "stats_shooting",
            f"{BASE_URL}/passing/players/{SEASON_TAG}": "stats_passing",
            f"{BASE_URL}/passing_types/players/{SEASON_TAG}": "stats_passing_types",
            f"{BASE_URL}/gca/players/{SEASON_TAG}": "stats_gca",
            f"{BASE_URL}/defense/players/{SEASON_TAG}": "stats_defense",
            f"{BASE_URL}/possession/players/{SEASON_TAG}": "stats_possession",
            f"{BASE_URL}/playingtime/players/{SEASON_TAG}": "stats_playing_time",
            f"{BASE_URL}/misc/players/{SEASON_TAG}": "stats_misc",
            f"{BASE_URL}/keepers/players/{SEASON_TAG}": "stats_keeper",
            f"{BASE_URL}/keepersadv/players/{SEASON_TAG}": "stats_keeper_adv",
        }
    elif STATS_LEVEL == "squads":
        URLS = {
            f"{BASE_URL}/stats/squads/{SEASON_TAG}": "stats_teams_standard_for",
            f"{BASE_URL}/shooting/squads/{SEASON_TAG}": "stats_teams_shooting_for",
            f"{BASE_URL}/passing/squads/{SEASON_TAG}": "stats_teams_passing_for",
            f"{BASE_URL}/passing_types/squads/{SEASON_TAG}": "stats_teams_passing_types_for",
            f"{BASE_URL}/gca/squads/{SEASON_TAG}": "stats_teams_gca_for",
            f"{BASE_URL}/defense/squads/{SEASON_TAG}": "stats_teams_defense_for",
            f"{BASE_URL}/possession/squads/{SEASON_TAG}": "stats_teams_possession_for",
            f"{BASE_URL}/playingtime/squads/{SEASON_TAG}": "stats_teams_playing_time_for",
            f"{BASE_URL}/misc/squads/{SEASON_TAG}": "stats_teams_misc_for",
            f"{BASE_URL}/keepers/squads/{SEASON_TAG}": "stats_teams_keeper_for",
            f"{BASE_URL}/keepersadv/squads/{SEASON_TAG}": "stats_teams_keeper_adv_for",
        }
    else:
        raise ValueError("STATS_LEVEL must be 'players' or 'squads'")

    # Bestehende Pipeline mit neuer Konfiguration ausführen
    return run_pipeline(output_folder=OUTPUT_FOLDER)
