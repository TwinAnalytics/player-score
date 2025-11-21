from __future__ import annotations
import pandas as pd


def minmax_scale(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    min_val = s.min()
    max_val = s.max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series(0.0, index=s.index)
    return (s - min_val) / (max_val - min_val)


def compute_score(
    df: pd.DataFrame,
    feature_weights: dict[str, float],
    score_name: str,
) -> pd.DataFrame:
    """
    Allgemeine Score-Funktion: nimmt ein DataFrame und ein Dict mit Gewichten
    und hängt eine neue Score-Spalte an.

    Skalierung:
    - wir machen eine Min-Max-Skalierung auf Basis von raw_score
    - Ergebnis liegt zwischen 10 und 95 (nicht 0–100), damit 100 "reserviert" bleibt
    """

    df = df.copy()
    raw_score = pd.Series(0.0, index=df.index, dtype=float)

    # 1) Rohscore als gewichtete Summe normalisierter Features
    for col, w in feature_weights.items():
        if col not in df.columns:
            print(f"⚠️ Spalte '{col}' fehlt – wird ignoriert.")
            continue

        scaled_feature = minmax_scale(df[col])  # 0–1 pro Feature
        raw_score += w * scaled_feature

    # Falls rohscore keine Varianz hat -> alle bekommen Mittelwert (z.B. 50)
    if raw_score.nunique() <= 1:
        df[score_name] = 50.0
        return df

    # 2) Min-Max über raw_score
    raw_min = raw_score.min()
    raw_max = raw_score.max()

    scaled = (raw_score - raw_min) / (raw_max - raw_min)  # 0–1

    # 3) Auf 10–95 stretchen (damit niemand 0 und niemand 100 ist)
    df[score_name] = 10.0 + 85.0 * scaled

    return df

def compute_score_absolute(
    df: pd.DataFrame,
    feature_weights: dict[str, float],
    feature_benchmarks: dict[str, float],
    score_name: str,
    max_score: float = 1000.0,
) -> pd.DataFrame:
    """
    Absoluter Score:
    - Jede Metrik wird gegen eine feste Benchmark verglichen (z.B. 1 Tor/90 = "voller Wert")
    - Pro Feature:
        ratio = (Wert / Benchmark) gecappt auf [0, 1]
        Beitrag = Gewicht * ratio
    - Am Ende: Score = max_score * (Summe der Gewichte * ratio) / Summe(Gewichte)

    WICHTIG:
    - feature_weights: sagt, wie wichtig ein Feature ist (z.B. Gls_Per90 wichtiger als KP_Per90)
    - feature_benchmarks: sagt, bei welchem Wert die Metrik "voll erfüllt" ist
      (z.B. 1.0 Tor/90, 2.5 Key Passes/90, 7 progressive Pässe/90)
    """

    df = df.copy()
    scores = pd.Series(0.0, index=df.index, dtype=float)
    weight_sum = 0.0

    for col, w in feature_weights.items():
        if col not in df.columns:
            print(f"⚠️ Spalte '{col}' fehlt – wird ignoriert.")
            continue

        bench = feature_benchmarks.get(col)
        if bench is None or bench <= 0:
            print(f"⚠️ Keine Benchmark für Spalte '{col}' – wird ignoriert.")
            continue

        # Verhältnis zum Benchmark (z.B. 0.8 Tore/90 bei Benchmark 1.0 -> 0.8)
        ratio = (df[col].astype(float) / bench).clip(lower=0.0, upper=1.0)

        scores += w * ratio
        weight_sum += w

    if weight_sum <= 0:
        df[score_name] = 0.0
        return df

    # Normierung, damit Summe(gewichtete ratio) im Idealfall = 1.0
    normalized = scores / weight_sum

    df[score_name] = max_score * normalized
    return df

def score_band_5(score: float) -> str:
    """
    Maps a 0–1000 score to 5 qualitative bands (English labels).
    """
    if score >= 900:
        return "Exceptional"
    elif score >= 750:
        return "World Class"
    elif score >= 400:
        return "Top Starter"
    elif score >= 200:
        return "Solid Squad Player"
    else:
        return "Below Big-5 Level"

def make_score_band_5(q1: float, q2: float, q3: float, q4: float):
    def band(score: float) -> str:
        if score >= q4:
            return "Exceptional"
        elif score >= q3:
            return "World Class"
        elif score >= q2:
            return "Top Starter"
        elif score >= q1:
            return "Solid Squad Player"
        else:
            return "Below Big-5 Level"
    return band

# === OFFENSIVE: FW (Striker / Winger) ==========================
OFF_FW_WEIGHTS_ABS = {
    "Gls_Per90": 0.40,
    "Ast_Per90": 0.15,
    "xG_Per90": 0.20,
    "xAG_Per90": 0.05,
    "KP_Per90": 0.10,
    "PrgP_Per90": 0.05,
    "PrgC_Per90": 0.05,
    "Sh/90": 0.03,
    "SoT/90": 0.02,
}

OFF_FW_BENCHMARKS_ABS = {
    "Gls_Per90": 0.9,   # ~0.9 Tore / 90
    "Ast_Per90": 0.6,
    "xG_Per90": 0.8,
    "xAG_Per90": 0.7,
    "KP_Per90": 2.5,
    "PrgP_Per90": 7.0,
    "PrgC_Per90": 5.0,
    "Sh/90": 4.0,
    "SoT/90": 2.0,
}

# === OFFENSIVE: Off_MF (Attacking Mid / Second Striker) ========
# weniger Tor-lastig, mehr Playmaking & Progression
OFF_AM_WEIGHTS_ABS = {
    "Gls_Per90": 0.25,
    "Ast_Per90": 0.20,
    "xG_Per90": 0.15,
    "xAG_Per90": 0.10,
    "KP_Per90": 0.12,
    "PrgP_Per90": 0.08,
    "PrgC_Per90": 0.05,
    "Sh/90": 0.03,
    "SoT/90": 0.02,
}

OFF_AM_BENCHMARKS_ABS = {
    "Gls_Per90": 0.6,
    "Ast_Per90": 0.7,
    "xG_Per90": 0.6,
    "xAG_Per90": 0.8,
    "KP_Per90": 3.0,
    "PrgP_Per90": 9.0,
    "PrgC_Per90": 6.0,
    "Sh/90": 3.0,
    "SoT/90": 1.6,
}

# === MIDFIELD: MF (zentraler Mittelfeldspieler) ================
mf_weights_abs = {
    "Ast_Per90": 0.22,
    "xAG_Per90": 0.18,
    "Gls_Per90": 0.08,
    "KP_Per90": 0.20,
    "PrgP_Per90": 0.14,
    "PrgC_Per90": 0.08,
    "Mid 3rd_stats_possession_Per90": 0.06,
    "Att 3rd_stats_possession_Per90": 0.04,
}

mf_benchmarks_abs = {
    "Ast_Per90": 0.3,
    "xAG_Per90": 0.6,
    "Gls_Per90": 0.4,
    "KP_Per90": 2.0,
    "PrgP_Per90": 10.0,
    "PrgC_Per90": 5.0,
    "Mid 3rd_stats_possession_Per90": 25.0,
    "Att 3rd_stats_possession_Per90": 12.0,
}

# === DEFENSIVE: DF (Innen-/Außenverteidiger) ===================
DEF_DF_WEIGHTS_ABS = {
    "TklW_Per90": 0.30,
    "Int_Per90": 0.25,
    "Blocks_stats_defense_Per90": 0.18,
    "Clr_Per90": 0.12,
    "Def Pen_Per90": 0.10,
    "Def 3rd_stats_possession_Per90": 0.06,
    "Mid 3rd_stats_possession_Per90": 0.04,
}

DEF_DF_BENCHMARKS_ABS = {
    "TklW_Per90": 6.0,
    "Int_Per90": 3.8,
    "Blocks_stats_defense_Per90": 3.0,
    "Clr_Per90": 9.0,
    "Def Pen_Per90": 7.5,
    "Def 3rd_stats_possession_Per90": 50.0,
    "Mid 3rd_stats_possession_Per90": 22.0,
}

# === DEFENSIVE: Def_MF (Sechser / defensiver Achter) ==========
# mehr Fokus auf Tackles/Interceptions & Mittelfeldpräsenz
DEF_DM_WEIGHTS_ABS = {
    "TklW_Per90": 0.32,
    "Int_Per90": 0.28,
    "Blocks_stats_defense_Per90": 0.12,
    "Clr_Per90": 0.08,
    "Def Pen_Per90": 0.08,
    "Def 3rd_stats_possession_Per90": 0.07,
    "Mid 3rd_stats_possession_Per90": 0.05,
}

DEF_DM_BENCHMARKS_ABS = {
    "TklW_Per90": 6.0,
    "Int_Per90": 4.5,
    "Blocks_stats_defense_Per90": 2.3,
    "Clr_Per90": 6.0,
    "Def Pen_Per90": 4.5,
    "Def 3rd_stats_possession_Per90": 45.0,
    "Mid 3rd_stats_possession_Per90": 30.0,
}


def compute_off_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Offensiv-Score:
    - FW   mit FW-Weights/Benchmarks
    - Off_MF mit eigenen Off-MF-Weights/Benchmarks
    """
    df_fw = df[df["Pos"] == "FW"].copy()
    df_am = df[df["Pos"] == "Off_MF"].copy()

    frames: list[pd.DataFrame] = []

    if not df_fw.empty:
        df_fw = compute_score_absolute(
            df_fw,
            feature_weights=OFF_FW_WEIGHTS_ABS,
            feature_benchmarks=OFF_FW_BENCHMARKS_ABS,
            score_name="OffScore_abs",
            max_score=1000.0,
        )
        df_fw["OffBand"] = df_fw["OffScore_abs"].apply(score_band_5)
        frames.append(df_fw)

    if not df_am.empty:
        df_am = compute_score_absolute(
            df_am,
            feature_weights=OFF_AM_WEIGHTS_ABS,
            feature_benchmarks=OFF_AM_BENCHMARKS_ABS,
            score_name="OffScore_abs",
            max_score=1000.0,
        )
        df_am["OffBand"] = df_am["OffScore_abs"].apply(score_band_5)
        frames.append(df_am)

    if frames:
        return pd.concat(frames, axis=0)

    # Fallback: kein FW / Off_MF
    return df[df["Pos"].isin(["FW", "Off_MF"])].copy()


def compute_def_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Defensiv-Score:
    - DF     mit Defender-Weights/Benchmarks
    - Def_MF mit eigenen DM-Weights/Benchmarks
    """
    df_df = df[df["Pos"] == "DF"].copy()
    df_dm = df[df["Pos"] == "Def_MF"].copy()

    frames: list[pd.DataFrame] = []

    if not df_df.empty:
        df_df = compute_score_absolute(
            df_df,
            feature_weights=DEF_DF_WEIGHTS_ABS,
            feature_benchmarks=DEF_DF_BENCHMARKS_ABS,
            score_name="DefScore_abs",
            max_score=1000.0,
        )
        df_df["DefBand"] = df_df["DefScore_abs"].apply(score_band_5)
        frames.append(df_df)

    if not df_dm.empty:
        df_dm = compute_score_absolute(
            df_dm,
            feature_weights=DEF_DM_WEIGHTS_ABS,
            feature_benchmarks=DEF_DM_BENCHMARKS_ABS,
            score_name="DefScore_abs",
            max_score=1000.0,
        )
        df_dm["DefBand"] = df_dm["DefScore_abs"].apply(score_band_5)
        frames.append(df_dm)

    if frames:
        return pd.concat(frames, axis=0)

    # Fallback: keine DF / Def_MF
    return df[df["Pos"].isin(["DF", "Def_MF"])].copy()


def compute_mid_scores(df: pd.DataFrame) -> pd.DataFrame:
    df_mf = df[df["Pos"] == "MF"].copy()
    df_mf = compute_score_absolute(
        df_mf,
        feature_weights=mf_weights_abs,
        feature_benchmarks=mf_benchmarks_abs,
        score_name="MidScore_abs",
        max_score=1000.0,
    )
    df_mf["MidBand"] = df_mf["MidScore_abs"].apply(score_band_5)
    return df_mf
