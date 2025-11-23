from pathlib import Path
import pandas as pd

from src.pipeline import run_full_pipeline
from src.scraping_fbref_player_stats import (
    run_scraping_for_season,
    run_scraping_for_2_bundesliga,
)

from src.multi_season import (
    load_all_seasons,
    aggregate_player_scores,
)
from src.squad import compute_squad_scores


SEASONS = [
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

# Für welche Saisons sollen zusätzlich 2.-Bundesliga-Spieler gescrapt werden?
SEASONS_WITH_2BL = {
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
    "2025-2026",
}


# Für welche Saisons sollen zusätzlich 2.-Bundesliga-Spieler gescrapt werden?
SEASONS_WITH_2BL = set(SEASONS)



if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    raw_dir = root / "Data" / "Raw"
    processed_dir = root / "Data" / "Processed"

    # ---------------------------------------------------------
    # 1) Pro Saison: Scraping (Big-5 + ggf. 2. Liga) + Scoring
    # ---------------------------------------------------------
    for season in SEASONS:
        print(f"\n=== Saison {season} ===")

        # 1a) Big-5 scrapen -> players_data_light-<Season>.csv
        run_scraping_for_season(season, output_folder=raw_dir)

        # 1b) Optional: 2. Bundesliga dazunehmen
        if season in SEASONS_WITH_2BL:
            print(f"--- Adding 2. Bundesliga for {season} ---")

            light_path = raw_dir / f"players_data_light-{season}.csv"
            backup_path = raw_dir / f"players_data_light-{season}-big5.csv"

            # Big-5-Light-File sichern
            if light_path.exists():
                light_path.rename(backup_path)
            else:
                raise FileNotFoundError(
                    f"Erwarte {light_path}, aber Datei existiert nicht. "
                    "Ist run_scraping_for_season fehlgeschlagen?"
                )

            # 2. Bundesliga scrapen -> überschreibt players_data_light-<Season>.csv
            run_scraping_for_2_bundesliga(season, output_folder=raw_dir)

            # Big-5 + 2BL mergen
            df_big5 = pd.read_csv(backup_path)
            df_2bl = pd.read_csv(light_path)
            df_merged = pd.concat([df_big5, df_2bl], ignore_index=True)

            df_merged.to_csv(light_path, index=False)
            print(f"Merged Big-5 + 2. Bundesliga → {light_path.name}")

            # Backup optional löschen
            backup_path.unlink(missing_ok=True)

        # 1c) Scoring-Pipeline: aus players_data_light-<Season>.csv -> player_scores-<Season>.csv
        out = run_full_pipeline(season, raw_dir, processed_dir)
        print(f"Fertiger Score-Export: {out}")

    # ---------------------------------------------------------
    # 2) Multi-Season-Tabellen automatisch erzeugen
    # ---------------------------------------------------------
    print("\n=== Building multi-season tables ===")
    df_all = load_all_seasons(processed_dir)
    df_agg = aggregate_player_scores(df_all)
    df_squad = compute_squad_scores(df_all)

    # Lange Tabelle über alle Saisons
    out_all = processed_dir / "player_scores_all_seasons_long.csv"
    df_all.to_csv(out_all, index=False)
    print(f"Gespeichert: {out_all}")

    # Aggregiert pro Spieler
    out_agg = processed_dir / "player_scores_agg_by_player.csv"
    df_agg.to_csv(out_agg, index=False)
    print(f"Gespeichert: {out_agg}")

    # Squad-Scores
    out_squad = processed_dir / "squad_scores_all_seasons.csv"
    df_squad.to_csv(out_squad, index=False)
    print(f"Gespeichert: {out_squad}")

    # Optional: Alias-Datei, wie von dir gewünscht
    single_out = processed_dir / "player_scores.csv"
    df_all.to_csv(single_out, index=False)
    print(f"Gespeichert: {single_out}")
