from __future__ import annotations

from pathlib import Path
import re
import pandas as pd


def load_all_seasons(processed_dir: Path) -> pd.DataFrame:
    """
    Lädt alle gescorten Saisons.

    Logik:
    1) Wenn es eine vorberechnete Multi-Season-Datei gibt
       (player_scores_all_seasons_long.csv), wird diese geladen.
    2) Falls nicht, werden alle player_scores-*.csv im Ordner
       Data/Processed eingelesen und zu einem langen DataFrame
       zusammengefügt.
    """
    processed_dir = Path(processed_dir)

    # 1) Versuch: fertige Long-Tabelle laden
    agg_path = processed_dir / "player_scores_all_seasons_long.csv"
    if agg_path.exists():
        df_all = pd.read_csv(agg_path)
        # Sicherheitshalber: Season-Spalte prüfen
        if "Season" not in df_all.columns:
            # Dann können wir hier nichts rekonstruieren – aber dieser Fall
            # sollte in deiner Pipeline eigentlich nicht vorkommen.
            pass
        return df_all

    # 2) Fallback: alle einzelnen Season-Files zusammensetzen
    files = sorted(processed_dir.glob("player_scores-*.csv"))
    frames = []

    for f in files:
        df_season = pd.read_csv(f)

        # Falls aus irgendeinem Grund Season fehlt, aus Dateinamen ableiten
        if "Season" not in df_season.columns:
            # Beispiel-Dateiname: player_scores-2017-2018.csv
            stem = f.stem  # "player_scores-2017-2018"
            parts = stem.split("player_scores-")
            season = parts[-1] if len(parts) > 1 else None
            if season:
                df_season["Season"] = season

        frames.append(df_season)

    if frames:
        df_all = pd.concat(frames, ignore_index=True)
    else:
        df_all = pd.DataFrame()

    return df_all



def aggregate_player_scores(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregiert über alle Saisons hinweg pro Spieler (optional: auch pro Pos),
    z.B. Durchschnitt der Scores und Summe der Einsatzzeit.

    Ergebnis:
    - eine Zeile pro Spieler
    - OffScore_mean, MidScore_mean, DefScore_mean
    - Gesamt-Minuten und Gesamt-90s
    """
    group_cols = ["Player"]  # optional später erweitern: + ["Pos"]

    agg = (
        df_all
        .groupby(group_cols, as_index=False)
        .agg({
            "OffScore_abs": "mean",
            "MidScore_abs": "mean",
            "DefScore_abs": "mean",
            "Min": "sum",
            "90s": "sum",
        })
    )

    agg = agg.rename(columns={
        "OffScore_abs": "OffScore_mean",
        "MidScore_abs": "MidScore_mean",
        "DefScore_abs": "DefScore_mean",
    })

    return agg
