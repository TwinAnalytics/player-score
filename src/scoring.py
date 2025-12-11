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

        # WICHTIG: NaN-Ratios als 0 behandeln, sonst wird der gesamte Score NaN
        ratio = ratio.fillna(0.0)

        scores += w * ratio
        weight_sum += w

    if weight_sum <= 0:
        df[score_name] = 0.0
        return df

    # Normierung, damit Summe(gewichtete ratio) im Idealfall = 1.0
    normalized = scores / weight_sum

    df[score_name] = max_score * normalized
    return df

def add_light_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt einfache abgeleitete Metriken für die Light-Scores hinzu,
    z.B. Finishing_Rate = Gls_Per90 / Sh/90.
    """
    df = df.copy()

    # Finishing_Rate: wie viele Tore pro Schuss
    if "Finishing_Rate" not in df.columns:
        if "Gls_Per90" in df.columns and "Sh/90" in df.columns:
            gls = df["Gls_Per90"].astype(float)
            shots = df["Sh/90"].astype(float)
            denom = shots.where(shots > 0.0, other=1.0)
            df["Finishing_Rate"] = (gls / denom).clip(lower=0.0)

    return df


def _choose_weight_set_for_league(
    df_pos: pd.DataFrame,
    full_weights: dict[str, float],
    full_benchmarks: dict[str, float],
    light_weights: dict[str, float] | None = None,
    light_benchmarks: dict[str, float] | None = None,
    min_full_features: int = 5,
    min_light_features: int = 3,
) -> tuple[dict[str, float], dict[str, float]]:
    # 1) Full-Set: welche Features sind tatsächlich vorhanden?
    available_full = [col for col in full_weights.keys() if col in df_pos.columns]
    if len(available_full) >= min_full_features:
        return full_weights, full_benchmarks

    # 2) Optional: Light-Set, falls übergeben
    if light_weights is not None and light_benchmarks is not None:
        available_light = [col for col in light_weights.keys() if col in df_pos.columns]
        if len(available_light) >= min_light_features:
            return light_weights, light_benchmarks

    # 3) Fallback: Full-Set, auch wenn viele Spalten fehlen –
    # compute_score_absolute kommt damit klar.
    return full_weights, full_benchmarks


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
    - FW     mit FW-Weights/Benchmarks
    - Off_MF mit eigenen Off-MF-Weights/Benchmarks

    WICHTIG:
    Wir entscheiden das Score-Set (voll vs. Light) PRO WETTBEWERB (Comp),
    damit Wettbewerbe mit weniger verfügbaren Metriken automatisch
    die Light-Variante nutzen, auch wenn Big-5-Spieler im selben DataFrame sind.
    """
    frames: list[pd.DataFrame] = []

    # Helper-Funktion: pro Pos und pro Comp scoren
    def _score_pos(
        df_all: pd.DataFrame,
        pos_label: str,
        full_weights: dict[str, float],
        full_benchmarks: dict[str, float],
    ) -> list[pd.DataFrame]:
        df_pos = df_all[df_all["Pos"] == pos_label].copy()
        if df_pos.empty:
            return []

        if "Comp" in df_pos.columns:
            groups = df_pos.groupby("Comp", dropna=False)
        else:
            # Falls aus irgendeinem Grund keine Comp-Spalte existiert
            groups = [("ALL", df_pos)]

        out_frames: list[pd.DataFrame] = []

        for _, df_comp in groups:
            weights, bench = _choose_weight_set_for_league(
                df_comp,
                full_weights=full_weights,
                full_benchmarks=full_benchmarks,
                min_full_features=5,
                min_light_features=3,
            )

            df_scored = compute_score_absolute(
                df_comp,
                feature_weights=weights,
                feature_benchmarks=bench,
                score_name="OffScore_abs",
                max_score=1000.0,
            )
            df_scored["OffBand"] = df_scored["OffScore_abs"].apply(score_band_5)
            out_frames.append(df_scored)

        return out_frames

    # FW
    frames.extend(
        _score_pos(
            df_all=df,
            pos_label="FW",
            full_weights=OFF_FW_WEIGHTS_ABS,
            full_benchmarks=OFF_FW_BENCHMARKS_ABS,
        )
    )

    # Offensives Mittelfeld
    frames.extend(
        _score_pos(
            df_all=df,
            pos_label="Off_MF",
            full_weights=OFF_AM_WEIGHTS_ABS,
            full_benchmarks=OFF_AM_BENCHMARKS_ABS,
        )
    )

    if frames:
        return pd.concat(frames, axis=0)

    # Fallback: kein FW / Off_MF
    return df[df["Pos"].isin(["FW", "Off_MF"])].copy()




def compute_def_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Defensiv-Score:
    - DF     mit Defender-Weights/Benchmarks
    - Def_MF mit eigenen DM-Weights/Benchmarks

    Für Ligen mit wenigen Metriken wird automatisch eine Light-Variante
    verwendet (TklW_Per90, Int_Per90) – PRO WETTBEWERB (Comp).
    """
    frames: list[pd.DataFrame] = []

    def _score_pos(
        df_all: pd.DataFrame,
        pos_label: str,
        full_weights: dict[str, float],
        full_benchmarks: dict[str, float],
    ) -> list[pd.DataFrame]:
        df_pos = df_all[df_all["Pos"] == pos_label].copy()
        if df_pos.empty:
            return []

        if "Comp" in df_pos.columns:
            groups = df_pos.groupby("Comp", dropna=False)
        else:
            groups = [("ALL", df_pos)]

        out_frames: list[pd.DataFrame] = []

        for _, df_comp in groups:
            weights, bench = _choose_weight_set_for_league(
                df_comp,
                full_weights=full_weights,
                full_benchmarks=full_benchmarks,
                min_full_features=4,
                min_light_features=2,
            )

            df_scored = compute_score_absolute(
                df_comp,
                feature_weights=weights,
                feature_benchmarks=bench,
                score_name="DefScore_abs",
                max_score=1000.0,
            )
            df_scored["DefBand"] = df_scored["DefScore_abs"].apply(score_band_5)
            out_frames.append(df_scored)

        return out_frames

    # DF (Innen-/Außenverteidiger)
    frames.extend(
        _score_pos(
            df_all=df,
            pos_label="DF",
            full_weights=DEF_DF_WEIGHTS_ABS,
            full_benchmarks=DEF_DF_BENCHMARKS_ABS,
        )
    )

    # Defensives Mittelfeld
    frames.extend(
        _score_pos(
            df_all=df,
            pos_label="Def_MF",
            full_weights=DEF_DM_WEIGHTS_ABS,
            full_benchmarks=DEF_DM_BENCHMARKS_ABS,
        )
    )

    if frames:
        return pd.concat(frames, axis=0)

    # Fallback: keine DF / Def_MF
    return df[df["Pos"].isin(["DF", "Def_MF"])].copy()


def compute_mid_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zentrales Mittelfeld (MF):
    - Standard: mf_weights_abs / mf_benchmarks_abs
    - Fallback: MF_WEIGHTS_ABS_LIGHT / MF_BENCHMARKS_ABS_LIGHT
      (Gls, Ast, Sh/90, SoT/90, Finishing_Rate, Int_Per90)

    Auch hier: Entscheidung voll vs. Light PRO WETTBEWERB (Comp),
    damit z.B. 2. Bundesliga die Light-Variante nutzt.
    """
    df = add_light_derived_metrics(df)

    df_mf = df[df["Pos"] == "MF"].copy()
    if df_mf.empty:
        return df[df["Pos"] == "MF"].copy()

    frames: list[pd.DataFrame] = []

    if "Comp" in df_mf.columns:
        groups = df_mf.groupby("Comp", dropna=False)
    else:
        groups = [("ALL", df_mf)]

    for _, df_comp in groups:
        mf_weights, mf_bench = _choose_weight_set_for_league(
            df_comp,
            full_weights=mf_weights_abs,
            full_benchmarks=mf_benchmarks_abs,
            min_full_features=5,
            min_light_features=3,
        )

        df_scored = compute_score_absolute(
            df_comp,
            feature_weights=mf_weights,
            feature_benchmarks=mf_bench,
            score_name="MidScore_abs",
            max_score=1000.0,
        )
        df_scored["MidBand"] = df_scored["MidScore_abs"].apply(score_band_5)
        frames.append(df_scored)

    if frames:
        return pd.concat(frames, axis=0)

    return df[df["Pos"] == "MF"].copy()

