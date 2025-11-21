from pathlib import Path

from src.pipeline import run_full_pipeline
from src.scraping_fbref_player_stats import run_scraping_for_season
from src.multi_season import load_all_seasons, aggregate_player_scores
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

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    raw_dir = root / "Data" / "Raw"
    processed_dir = root / "Data" / "Processed"

    for season in SEASONS:
        print(f"\n=== Saison {season} ===")

        # 1) Scraping (Light-CSV in Data/Raw anlegen)
        run_scraping_for_season(season, output_folder=raw_dir)

        # 2) Scoring-Pipeline
        out = run_full_pipeline(season, raw_dir, processed_dir)
        print(f"Fertiger Score-Export: {out}")

    # ---------------------------------------------------------
    # 1) Pro Saison: Scraping + Scoring
    # ---------------------------------------------------------
    for season in SEASONS:
        print(f"\n=== Saison {season} ===")

        # Scraping (Light-CSV in Data/Raw anlegen)
        run_scraping_for_season(season, output_folder=raw_dir)

        # Scoring-Pipeline
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
