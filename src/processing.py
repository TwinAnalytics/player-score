from __future__ import annotations

from typing import Sequence
import pandas as pd
import numpy as np


# ============================================================================
# Per-90-Helfer
# ============================================================================

def add_per90_from_90s(
    df: pd.DataFrame,
    stats_cols: Sequence[str],
    ninety_col: str = "90s",
    suffix: str = "_Per90",
) -> pd.DataFrame:
    """
    Rechnet aus Gesamtwerten per-90-Metriken:
    Beispiel: Gls_Per90 = Gls / 90s

    stats_cols: Liste der Spalten, die du umrechnen willst, z.B. ["Gls", "Ast", "Sh"]
    ninety_col: Spalte mit gespielten 90-Minuten-Einheiten (bei dir: '90s')
    suffix: Suffix für die neuen Spaltennamen
    """

    df = df.copy()

    if ninety_col not in df.columns:
        raise ValueError(f"Spalte '{ninety_col}' fehlt im DataFrame.")

    # Nur Spieler mit > 0 gespielten 90er-Einheiten behalten
    mask = df[ninety_col] > 0
    df = df[mask].copy()

    n90 = df[ninety_col].astype(float)

    for col in stats_cols:
        if col not in df.columns:
            print(f"Warnung: Spalte '{col}' nicht gefunden – wird übersprungen.")
            continue

        df[col + suffix] = df[col].astype(float) / n90

    return df


# Standard-Liste deiner per90-Stats
DEFAULT_PER90_COLS: list[str] = [
    # Offensiv
    "Gls",
    "Ast",
    "xG",
    "xAG",
    "KP",
    "PrgP",
    "PrgC",
    "Mis",
    "G-PK",    # NEW: non-penalty goals
    "npxG",    # NEW: non-penalty xG
    "SoT",     # NEW: shots on target
    "Succ",    # NEW: dribbles completed
    "TB",      # NEW: through balls

    # Defensiv
    "TklW",                    # gewonnene Tackles
    "Int",
    "Clr",
    "Blocks_stats_defense",
    "Fls",
    "Err",

    # Zonen-Touches
    "Def Pen",
    "Def 3rd_stats_possession",
    "Mid 3rd_stats_possession",
    "Att 3rd_stats_possession",
    "Att Pen",
]


def add_standard_per90(
    df: pd.DataFrame,
    stats_cols: Sequence[str] | None = None,
    ninety_col: str = "90s",
    suffix: str = "_Per90",
) -> pd.DataFrame:
    """
    Convenience-Funktion:
    - nutzt DEFAULT_PER90_COLS, wenn stats_cols=None
    - ruft add_per90_from_90s auf
    """
    if stats_cols is None:
        stats_cols = DEFAULT_PER90_COLS

    return add_per90_from_90s(df, stats_cols=stats_cols, ninety_col=ninety_col, suffix=suffix)

def add_squad_per90(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt allen Squad-Daten per-90-Metriken hinzu.
    Erwartet eine Spalte '90s' (Team-Minuten / 90).
    """

    df = df.copy()

    if "90s" not in df.columns:
        raise ValueError("Squad dataframe must contain column '90s'.")

    # Sicherheit: 90s = Spiele * 1.0
    df["90s"] = pd.to_numeric(df["90s"], errors="coerce").replace([np.inf, -np.inf], np.nan)

    # Mapping: neue Spalte → Rohwert-Spalte
    # processing.py
import numpy as np
import pandas as pd

def add_squad_per90(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt wichtige per-90-Stats für Team-Radars hinzu.

    Erwartet:
      - '90s' (Anzahl 90-Minuten-Einheiten)
    """

    df = df.copy()

    # 90s als Basis (sonst über Minuten)
    if "90s" in df.columns:
        ninety = pd.to_numeric(df["90s"], errors="coerce").replace(0, np.nan)
    else:
        minutes = pd.to_numeric(df.get("Min", np.nan), errors="coerce")
        ninety = (minutes / 90.0).replace(0, np.nan)

    def per90(src_col: str, out_col: str):
        if src_col in df.columns:
            num = pd.to_numeric(df[src_col], errors="coerce")
            df[out_col] = num / ninety

    # ------- OFFENSIVE METRIKEN ------- #
    per90("npxG",   "npxG/90")
    per90("Sh",     "Sh/90")       # Shots
    per90("SoT",    "SoT/90")      # Shots on target
    per90("xAG",    "xAG/90")
    per90("PPA",    "PPA/90")      # Passes to penalty area
    per90("CPA",    "CPA/90")      # Carries to penalty area
    per90("Att Pen","Att Pen/90")  # Touches in penalty area
    per90("KP",     "KP/90")       # Key passes

    # SCA / GCA haben meist schon SCA90 / GCA90 in den FBref-Exports
    # – falls nicht, bauen wir sie:
    if "SCA" in df.columns and "SCA90" not in df.columns:
        per90("SCA", "SCA90")
    if "GCA" in df.columns and "GCA90" not in df.columns:
        per90("GCA", "GCA90")

    # ------- DEFENSIVE METRIKEN ------- #
    # onxGA = xG Conceded
    per90("onxGA",  "onxGA/90")

    # GA90 ist oft schon in der Datei; wenn nicht, bauen wir es
    if "GA90" not in df.columns and "GA" in df.columns:
        per90("GA", "GA90")

    per90("SoTA",   "SoTA/90")     # Shots on target conceded
    per90("Clr",    "Clr/90")      # Clearances
    per90("Tkl+Int","Tkl+Int/90")  # Tackles + Interceptions
    per90("PSxG+/-","PSxG+/-/90")  # PSxG-GA per 90
    per90("CS",     "CS/90")       # Clean sheets per 90 (zusätzlich zu CS%)
    per90("Opp",    "Opp/90")      # Crosses faced per 90

    # Prozentwerte (Cmp%, CS%) lassen wir so – die sind schon in ihrer finalen Skala
    return df


# ============================================================================
# Positionslogik
# ============================================================================

def main_pos_from_string(pos: str) -> str | None:
    """
    Leitet eine Hauptposition aus einem FBref-Positionsstring ab, z.B.:
    - "FW, MF" -> FW
    - "MF, FW" -> MF
    - "DF,MF"  -> DF (oder MF, je nach Reihenfolge)
    
    Logik:
    1) Erster Positions-Teil hat Vorrang.
    2) Falls dort nichts erkannt wird, wird über alle Teile mit einer Prioritätsliste gesucht.
    """

    if pos is None:
        return None

    # Positionsteile splitten
    parts = [p.strip() for p in str(pos).split(",") if p.strip()]
    if not parts:
        return None

    # Codes, die wir erkennen wollen
    primary_codes = ["FW", "AM", "MF", "DF", "GK"]

    first_part = parts[0]

    # 1) Zuerst den ERSTEN Teil respektieren
    for code in primary_codes:
        if code in first_part:
            return code

    # 2) Fallback: über alle Teile mit einer Priorität suchen
    #    (hier kannst du die Reihenfolge festlegen, z.B. MF vor FW
    #     oder FW vor MF – aber das greift nur, wenn der erste Teil nichts enthält)
    priority = ["FW", "AM", "MF", "DF", "GK"]

    for code in priority:
        if any(code in part for part in parts):
            return code

    # 3) Letzter Fallback: erster Teil roh zurückgeben
    return first_part


def refine_mf_with_zones(row: pd.Series) -> str:
    """
    Unterscheidet Mittelfeldspieler in:
    - Off_MF (offensiver MF)
    - Def_MF (defensiver MF)
    - MF     (neutral)

    Basis: Touches in Defensiv-, Mittel- und Angriffszone + Strafraum.
    Erwartet, dass row["Pos_main_base"] bereits gesetzt ist.
    """

    pos_base = row.get("Pos_main_base")

    # Nur Mittelfeldspieler anpassen
    if pos_base != "MF":
        # Für FW / DF / GK etc. geben wir einfach die Basis-Pos zurück
        return pos_base

    # Zonen-Touches holen (fehlende Werte als 0 behandeln)
    def_pen = row.get("Def Pen", 0) or 0
    def_3rd = row.get("Def 3rd_stats_possession", 0) or 0
    mid_3rd = row.get("Mid 3rd_stats_possession", 0) or 0
    att_3rd = row.get("Att 3rd_stats_possession", 0) or 0
    att_pen = row.get("Att Pen", 0) or 0

    total = def_pen + def_3rd + mid_3rd + att_3rd + att_pen
    if total <= 0:
        # keine Info -> neutraler MF
        return "MF"

    def_share = (def_pen + def_3rd) / total
    att_share = (att_3rd + att_pen) / total

    # Schwellen: kannst du später noch feinjustieren
    # offensiver MF: klarer Fokus im letzten Drittel / Angriff
    if att_share >= 0.35 and att_share >= def_share:
        return "Off_MF"

    # defensiver MF: klarer Fokus in Defensivzone
    if def_share >= 0.35 and def_share > att_share:
        return "Def_MF"

    # ausgeglichen -> normaler MF
    return "MF"


def prepare_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Packt deine gesamte Positionslogik in eine Funktion:

    - interpretiert FBref-"Pos" → Hauptposition (Pos_main_base)
    - differenziert Mittelfeldspieler mit Zonen-Touches → Off_MF / Def_MF / MF
    - schreibt das Ergebnis zurück in df["Pos"]
    - filtert Torhüter (GK) raus

    Erwartet:
    - Spalte "Pos" (FBref-Position)
    - Zonen-Spalten:
        "Def Pen", "Def 3rd_stats_possession", "Mid 3rd_stats_possession",
        "Att 3rd_stats_possession", "Att Pen"
    """

    df = df.copy()

    if "Pos" not in df.columns:
        raise ValueError("Spalte 'Pos' fehlt im DataFrame – ohne Positionsangaben geht es nicht.")

    # 1) Hauptposition aus dem raw-FBref-String
    df["Pos_main_base"] = df["Pos"].apply(main_pos_from_string)

    # 2) Mittelfeldspieler anhand der Zonen-Stats verfeinern
    df["Pos_refined"] = df.apply(refine_mf_with_zones, axis=1)

    # 3) Ergebnis in die eigentliche Pos-Spalte übernehmen
    df["Pos"] = df["Pos_refined"]

    # 4) Torhüter rausfiltern (wir fokussieren uns auf Feldspieler)
    df = df[df["Pos"] != "GK"].copy()

    # 5) Hilfsspalten optional wieder entfernen
    df = df.drop(columns=["Pos_main_base", "Pos_refined"], errors="ignore")

    return df


# ============================================================================
# Minuten-Filter
# ============================================================================

def filter_by_90s(df: pd.DataFrame, min_90s: float = 5.0) -> pd.DataFrame:
    """
    Filtert Spieler mit zu wenig Einsatzzeit heraus.

    min_90s = 5.0 bedeutet z.B. mindestens 5 volle 90-Minuten-Einheiten
    (= ca. 450 Minuten).

    Kannst du für Defensivspieler später separat strenger machen,
    z.B. min_90s=10.
    """
    if "90s" not in df.columns:
        raise ValueError("Spalte '90s' fehlt im DataFrame – kann nicht nach Einsatzzeit filtern.")

    df = df.copy()
    return df[df["90s"] >= float(min_90s)].copy()
