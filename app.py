import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")   # verhindert GUI-Fehler im Browser
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mplsoccer import PyPizza

plt.rcParams["figure.dpi"] = 200      # höhere Render-Auflösung
plt.rcParams["savefig.dpi"] = 200


from src.processing import prepare_positions, add_standard_per90



PRIMARY_COLOR = "#1f77b4"  # main brand color (used for headline & band chart)
# Brand / Theme colors
APP_BG = "#000000"            # sehr dunkles Blau, App- & Chart-Background
GRID_COLOR = "#374151"
TEXT_COLOR = "#e5e7eb"
SLICE_COLOR = "cornflowerblue"
PRIMARY_COLOR = "#1f77b4"   # gleiche Farbe wie Band-Distribution-Balken
VALUE_COLOR = "#00B8A9"

# drei Töne davon für die Pizza-Gruppen
COLOR_POSSESSION = "#80F5E3"   # helles Türkis
COLOR_ATTACKING  = "#00B8A9"   # dein Originalfarbton
COLOR_DEFENDING  = "#006058"   # dunkles Petrolgrün

# Metric-specific weights (lower weight = more smoothing due to high variance)
METRIC_WEIGHTS = {
    "Dribbles completed": 0.7,
    "Pass completion": 1.0,
    "Prog. carries": 0.85,
    "Prog. passes": 0.9,
    "Through balls": 1.0,

    "Assists": 1.0,
    "Key passes": 0.9,
    "Non-penalty goals": 0.6,
    "npxG": 0.8,
    "Shots on target": 0.7,

    "Tackles won": 0.9,
    "Interceptions": 1.0,
    "Blocks": 1.0,
    "Clearances": 1.0,
}

# -------------------------------------------------------------------
# Band labels and icons (English)
# -------------------------------------------------------------------
BAND_ICONS = {
    "Exceptional": "🟣 Exceptional",
    "World Class": "🟢 World Class",
    "Top Starter": "🔵 Top Starter",
    "Solid Squad Player": "🟡 Solid Squad Player",
    "Below Big-5 Level": "⚪️ Below Big-5 Level",
}

BAND_ORDER = [
    "Exceptional",
    "World Class",
    "Top Starter",
    "Solid Squad Player",
    "Below Big-5 Level",
]

BIG5_COMPS = {
    "eng Premier League",
    "es La Liga",
    "de Bundesliga",
    "it Serie A",
    "fr Ligue 1",
}

# -------------------------------------------------------------------
# Data loading (cached)
# -------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        from src.multi_season import load_all_seasons, aggregate_player_scores
        from src.squad import compute_squad_scores
    except Exception as e:
        st.error("Error importing data loading functions (src.multi_season / src.squad).")
        st.exception(e)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    try:
        root = Path(__file__).resolve().parent
        processed_dir = root / "Data" / "Processed"

        df_all = load_all_seasons(processed_dir)
        df_agg = aggregate_player_scores(df_all)
        df_squad = compute_squad_scores(df_all)

        return df_all, df_agg, df_squad

    except Exception as e:
        st.error("Error loading processed score files from Data/Processed.")
        st.exception(e)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def get_role_label(pos: str | None) -> str:
    """
    Map internal Pos labels to a human-readable role.
    """
    if pos is None:
        return "Unknown role"
    if pos in ("FW", "Off_MF"):
        return "Attacking player"
    if pos == "MF":
        return "Midfielder"
    if pos in ("DF", "Def_MF"):
        return "Defensive player"
    return f"Role: {pos}"

def mean_or_na(series: pd.Series | None) -> str:
    """Format a mean value as string or 'n/a' if no data."""
    if series is None or series.empty or series.notna().sum() == 0:
        return "n/a"
    return f"{series.mean():.1f}"

def assess_diff(diff: float) -> str:
    """
    Map the difference between player score and squad score to a verbal assessment.
    """
    if diff >= 100:
        return "clearly above squad level (strong upgrade)"
    elif diff >= 30:
        return "above squad level"
    elif diff <= -100:
        return "clearly below squad level"
    elif diff <= -30:
        return "below squad level"
    else:
        return "around squad level"
    
def get_primary_score_and_band(row):
    pos = row.get("Pos")

    if pos == "FW":
        return row.get("OffScore_abs"), row.get("OffBand")

    if pos == "MF":
        return row.get("MidScore_abs"), row.get("MidBand")

    if pos == "DF":
        return row.get("DefScore_abs"), row.get("DefBand")

    return float("nan"), None

def band_from_score(score: float | None) -> str | None:
    """Map a numeric score to a band label (using the same scale as in the UI)."""
    if score is None or pd.isna(score):
        return None

    if score >= 900:
        band = "Exceptional"
    elif score >= 750:
        band = "World Class"
    elif score >= 400:
        band = "Top Starter"
    elif score >= 200:
        band = "Solid Squad Player"
    else:
        band = "Below Big-5 Level"

    # mit Icon/Text aus BAND_ICONS kombinieren
    return BAND_ICONS.get(band, band)



def score_trend_chart(df_player_all: pd.DataFrame, score_col: str, label: str):
    """
    Build a clean line chart (no grid, just one petrol line)
    for the selected score over seasons.
    """
    if "Season" not in df_player_all.columns or score_col not in df_player_all.columns:
        return

    plot_df = (
        df_player_all[["Season", score_col]]
        .dropna()
        .sort_values("Season")
    )

    if plot_df.empty:
        st.info("No data available for the selected score.")
        return

    chart = (
        alt.Chart(plot_df)
        .mark_line(strokeWidth=2, color="#00B8A9")
        .encode(
            x=alt.X("Season:N", title="Season"),
            y=alt.Y(f"{score_col}:Q", title=label),
        )
        .properties(height=280)
        .configure_axis(
            grid=False,
            domain=True,
            labelColor="#E5E7EB",
            titleColor="#E5E7EB",
        )
        .configure_view(
            strokeWidth=0
        )
    )

    st.altair_chart(chart, use_container_width=True)

def render_pizza_chart(
    df_all: pd.DataFrame,
    df_player: pd.DataFrame,
    role: str,
    season: str | None,
):
    """
    PyPizza-Chart (StatsBomb-Style) für Big-5-Spieler.

    - Werte = Perzentil-Rank (0–100) vs Big-5-Peers gleicher Rolle + Season.
    - Slices kleiner (Donut-Style) über inner_circle_size.
    - Drei Farben (Possession / Attacking / Defending) in Tönen der Hauptfarbe.
    """
    if df_player.empty or role is None:
        st.info("No data available for pizza chart.")
        return

    row = df_player.iloc[0]

    comp = row["Comp"] if "Comp" in row.index else None
    if comp not in BIG5_COMPS:
        st.info("Pizza chart is currently only available for Big-5 competitions.")
        return

    # Vergleichsgruppe: Big-5, gleiche Rolle, gleiche Season
    df_comp = df_all.copy()
    if "Comp" in df_comp.columns:
        df_comp = df_comp[df_comp["Comp"].isin(BIG5_COMPS)]
    if "Pos" in df_comp.columns and role is not None:
        df_comp = df_comp[df_comp["Pos"] == role]
    if season is not None and "Season" in df_comp.columns:
        df_comp = df_comp[df_comp["Season"] == season]

    if df_comp.empty:
        st.info("No comparison group available for pizza chart.")
        return

    # Helper: erste vorhandene Spalte aus Kandidaten
    def resolve_metric_column(df: pd.DataFrame, row_s: pd.Series, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns and c in row_s.index:
                return c
        return None

    # --------- Metriken inkl. Gruppenzugehörigkeit ---------
    metric_defs = [
        # (Group, [Candidate columns], Label)
        ("Possession", ["Succ_Per90", "Succ/90"], "Dribbles\nCompleted"),
        ("Possession", ["Cmp%"], "Pass\nCompletion"),
        ("Possession", ["PrgC_Per90", "PrgC/90"], "Prog.\nCarries"),
        ("Possession", ["PrgP_Per90", "PrgP/90"], "Prog.\nPasses"),
        ("Possession", ["TB_Per90", "TB/90"], "Through\nBalls"),

        ("Attacking", ["Ast_Per90", "Ast/90"], "Assists"),
        ("Attacking", ["KP_Per90", "KP/90"], "Key\nPasses"),
        ("Attacking", ["G-PK_Per90", "G-PK/90"], "Non-Penalty\nGoals"),
        ("Attacking", ["npxG_Per90", "npxG/90"], "Non-Penalty\nxG"),
        ("Attacking", ["SoT_Per90", "SoT/90"], "Shots\non\nTarget"),

        ("Defending", ["TklW_Per90", "TklW/90"], "Tackles\nWon"),
        ("Defending", ["Int_Per90", "Int/90"], "Interceptions"),
        ("Defending", ["Blocks_stats_defense_Per90", "Blocks_stats_defense"], "Blocks"),
        ("Defending", ["Clr_Per90", "Clr/90"], "Clearances"),
    ]

    params: list[str] = []
    values: list[int] = []
    groups: list[str] = []

    for group, candidates, label in metric_defs:
        col = resolve_metric_column(df_comp, row, candidates)
        if col is None:
            continue

        val = row[col]
        if pd.isna(val):
            continue

        peers = pd.to_numeric(df_comp[col], errors="coerce").dropna()
        if peers.empty:
            continue

        # Perzentil-Rank (0–100)
        percentile = (peers <= val).mean() * 100.0
        percentile = int(round(percentile))

        params.append(label)
        values.append(percentile)
        groups.append(group)

    if len(params) < 3:
        st.info("Not enough metrics available to build a pizza chart for this player (Big-5 only).")
        return

    values = np.array(values)

    # Slice-Farben je Gruppe
    group_color_map = {
        "Possession": COLOR_POSSESSION,
        "Attacking": COLOR_ATTACKING,
        "Defending": COLOR_DEFENDING,
    }
    slice_colors = [group_color_map[g] for g in groups]

    # ---------- PyPizza: kleiner, dunkler Hintergrund, Donut ----------
    baker = PyPizza(
        params=params,
        background_color=APP_BG,  # <- dunkler Hintergrund
        straight_line_color="#4b5563",
        straight_line_lw=0.1,
        last_circle_color="#9ca3af",
        last_circle_lw=0.1,
        other_circle_color="#4b5563",
        other_circle_lw=0.1,
        other_circle_ls="--",
        inner_circle_size=20.0,      # <- größerer innerer Kreis = kleinere Slices
        straight_line_limit=100.0,   # Percentiles bis 100
    )

    fig, ax = baker.make_pizza(
        values,
        figsize=(4.5, 4.5),          # kleineres Chart
        param_location=118,          # Labels näher an der Pizza, ringförmig verteilt
        slice_colors=slice_colors,
        color_blank_space="same",    # „leerer“ Bereich in gleicher Slice-Farbe
        blank_alpha=0.15,
        kwargs_slices=dict(
            edgecolor=APP_BG,
            zorder=2,
            linewidth=0.1,
        ),
        kwargs_params=dict(
            color=TEXT_COLOR,
            fontsize=7,
            va="center",
        ),
        kwargs_values=dict(
            color="#000000",
            fontsize=7,
            zorder=9,
            bbox=dict(
                edgecolor="#000000",
                facecolor=VALUE_COLOR,  # gleiche Farbe wie Band-Balken
                boxstyle="round,pad=0.2",
                lw=0.5,
            ),
        ),
    )

    # Sicherheitshalber Background setzen
    fig.set_facecolor(APP_BG)
    ax.set_facecolor(APP_BG)

    # Titel / Subtitel
    player_name = row.get("Player", "")
    squad_name = row.get("Squad", "")
    season_txt = season or ""

    fig.text(
        0.5, 1.05,
        f"{player_name} | {squad_name}",
        size=8,
        ha="center",
        color=TEXT_COLOR,
    )

    fig.text(
        0.5, 1.02,
        f"Season {season} | Stats Per 90",
        size=6,
        ha="center",
        color=TEXT_COLOR,
    )

    # ---------- Legende unten (mit Text) ----------
    import matplotlib.patches as mpatches

    handles = [
        mpatches.Patch(color=COLOR_POSSESSION, label="Possession"),
        mpatches.Patch(color=COLOR_ATTACKING,  label="Attacking"),
        mpatches.Patch(color=COLOR_DEFENDING,  label="Defending"),
    ]
    leg = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    for text in leg.get_texts():
        text.set_color(TEXT_COLOR)
        text.set_fontweight("bold")

    return fig
    #st.pyplot(fig)

def render_career_pizza_chart(
    player: str,
    role: str,
    seasons: list[str],
) -> plt.Figure | None:
    """
    Career-Pizza-Chart:

    - Aggregiert die Per-90-Metriken des Spielers über alle Saisons (minuten-gewichtet).
    - Vergleichsgruppe: alle Big-5-Spieler gleicher Rolle über alle Saisons.
    """
    if not seasons or role is None:
        st.info("Not enough data available for a career pizza chart.")
        return None

    root = Path(__file__).resolve().parent
    raw_dir = root / "Data" / "Raw"

    df_features_all_list = []
    df_player_feat_list = []

    # Alle Seasons des Spielers laden (Feature-Table pro Season)
    for s in seasons:
        season_safe = s.replace("/", "-")
        csv_path = raw_dir / f"players_data_light-{season_safe}.csv"
        if not csv_path.exists():
            continue

        df_season = pd.read_csv(csv_path)
        df_season = prepare_positions(df_season)
        df_season = add_standard_per90(df_season)
        df_season["Season"] = s

        df_features_all_list.append(df_season)

        df_p = df_season[df_season["Player"] == player].copy()
        if not df_p.empty:
            df_player_feat_list.append(df_p)

    if not df_player_feat_list:
        st.info("No raw feature data found across seasons for this player.")
        return None

    df_features_all = pd.concat(df_features_all_list, ignore_index=True)
    df_player_all_feat = pd.concat(df_player_feat_list, ignore_index=True)

    # Nur Big-5 für Vergleichsgruppe
    if "Comp" in df_player_all_feat.columns:
        player_comps = set(df_player_all_feat["Comp"].dropna().unique())
        if not (player_comps & BIG5_COMPS):
            st.info("Career pizza chart is currently only available for Big-5 competitions.")
            return None

    df_comp = df_features_all.copy()
    if "Comp" in df_comp.columns:
        df_comp = df_comp[df_comp["Comp"].isin(BIG5_COMPS)]
    if "Pos" in df_comp.columns and role is not None:
        df_comp = df_comp[df_comp["Pos"] == role]

    if df_comp.empty:
        st.info("No comparison group available for career pizza chart.")
        return None

    # Helper: erste vorhandene Spalte aus Kandidaten
    def resolve_metric_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # Metrik-Definition wie im Season-Pizza
    metric_defs = [
        ("Possession", ["Succ_Per90", "Succ/90"], "Dribbles\nCompleted"),
        ("Possession", ["Cmp%"], "Pass\nCompletion"),
        ("Possession", ["PrgC_Per90", "PrgC/90"], "Prog.\nCarries"),
        ("Possession", ["PrgP_Per90", "PrgP/90"], "Prog.\nPasses"),
        ("Possession", ["TB_Per90", "TB/90"], "Through\nBalls"),

        ("Attacking", ["Ast_Per90", "Ast/90"], "Assists"),
        ("Attacking", ["KP_Per90", "KP/90"], "Key\nPasses"),
        ("Attacking", ["G-PK_Per90", "G-PK/90"], "Non-Penalty\nGoals"),
        ("Attacking", ["npxG_Per90", "npxG/90"], "Non-Penalty\nxG"),
        ("Attacking", ["SoT_Per90", "SoT/90"], "Shots\non\nTarget"),

        ("Defending", ["TklW_Per90", "TklW/90"], "Tackles\nWon"),
        ("Defending", ["Int_Per90", "Int/90"], "Interceptions"),
        ("Defending", ["Blocks_stats_defense_Per90", "Blocks_stats_defense"], "Blocks"),
        ("Defending", ["Clr_Per90", "Clr/90"], "Clearances"),
    ]

    params: list[str] = []
    values: list[int] = []
    groups: list[str] = []

    # Minuten als Gewicht
    mins = pd.to_numeric(df_player_all_feat.get("Min", pd.Series(dtype=float)), errors="coerce")
    total_min = mins.sum() if mins.notna().any() else None

    for group, candidates, label in metric_defs:
        col = resolve_metric_column(df_comp, candidates)
        if col is None or col not in df_player_all_feat.columns:
            continue

        # Spieler: minuten-gewichtete Per-90-Aggregation (oder Mittelwert als Fallback)
        vals_player = pd.to_numeric(df_player_all_feat[col], errors="coerce")
        if vals_player.notna().any():
            # 1) Minuten-Gewichtung (wie bisher)
            if total_min is not None and total_min > 0:
                num_min = (vals_player * mins).sum()
                denom_min = total_min
                base_value = num_min / denom_min
            else:
                base_value = vals_player.mean()

            # 2) Metrik-Gewichtung (neu)
            metric_weight = METRIC_WEIGHTS.get(label.replace("\n"," ").strip(), 1.0)

            # Glättung durch Mischung aus basierten und glattem Wert
            # Weight <1 → mehr smoothing
            player_val = base_value * metric_weight + base_value.mean() * (1 - metric_weight)
        else:
            continue

        if pd.isna(player_val):
            continue

        # Vergleichsverteilung
        peers = pd.to_numeric(df_comp[col], errors="coerce").dropna()
        if peers.empty:
            continue

        percentile = (peers <= player_val).mean() * 100.0
        percentile = int(round(percentile))

        params.append(label)
        values.append(percentile)
        groups.append(group)

    if len(params) < 3:
        st.info("Not enough metrics available to build a career pizza chart for this player (Big-5 only).")
        return None

    values = np.array(values)

    group_color_map = {
        "Possession": COLOR_POSSESSION,
        "Attacking": COLOR_ATTACKING,
        "Defending": COLOR_DEFENDING,
    }
    slice_colors = [group_color_map[g] for g in groups]

    baker = PyPizza(
        params=params,
        background_color=APP_BG,
        straight_line_color="#4b5563",
        straight_line_lw=0.1,
        last_circle_color="#9ca3af",
        last_circle_lw=0.1,
        other_circle_color="#4b5563",
        other_circle_lw=0.1,
        other_circle_ls="--",
        inner_circle_size=20.0,
        straight_line_limit=100.0,
    )

    fig, ax = baker.make_pizza(
        values,
        figsize=(4.5, 4.5),
        param_location=118,
        slice_colors=slice_colors,
        color_blank_space="same",
        blank_alpha=0.15,
        kwargs_slices=dict(
            edgecolor=APP_BG,
            zorder=2,
            linewidth=0.1,
        ),
        kwargs_params=dict(
            color=TEXT_COLOR,
            fontsize=7,
            va="center",
        ),
        kwargs_values=dict(
            color="#000000",
            fontsize=7,
            zorder=9,
            bbox=dict(
                edgecolor="#000000",
                facecolor=VALUE_COLOR,
                boxstyle="round,pad=0.2",
                lw=0.5,
            ),
        ),
    )

    fig.set_facecolor(APP_BG)
    ax.set_facecolor(APP_BG)

    # Titel / Subtitle
    first_season = min(seasons)
    last_season = max(seasons)

    fig.text(
        0.5, 1.05,
        f"{player}",
        size=8,
        ha="center",
        color=TEXT_COLOR,
    )

    fig.text(
        0.5, 1.02,
        f"Career stats per 90 | Seasons {first_season} – {last_season}",
        size=6,
        ha="center",
        color=TEXT_COLOR,
    )

    handles = [
        mpatches.Patch(color=COLOR_POSSESSION, label="Possession"),
        mpatches.Patch(color=COLOR_ATTACKING,  label="Attacking"),
        mpatches.Patch(color=COLOR_DEFENDING,  label="Defending"),
    ]
    leg = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    for text in leg.get_texts():
        text.set_color(TEXT_COLOR)
        text.set_fontweight("bold")

    return fig

def render_role_scatter(
    df_comp: pd.DataFrame,
    df_player: pd.DataFrame,
    role: str,
):
    """
    Scatter plot for role-specific attacking/creation metrics.

    FW / Off_MF:
        y = Goals per 90      (Gls_Per90 / Gls/90)
        x = xG per 90         (xG_Per90 / xG/90)

    MF:
        y = Assists per 90    (Ast_Per90 / Ast/90)
        x = xAG per 90        (xAG_Per90 / xAG/90 / xA_Per90 / xA/90)

    DF / Def_MF:
        y = Tackles won per 90      (TklW_Per90 / TklW/90)
        x = Interceptions per 90    (Int_Per90 / Int/90)
    """
    if df_player.empty or role is None:
        st.info("No data available for role scatter plot.")
        return None

    row = df_player.iloc[0]


    # ---- Helper: Spaltennamen auflösen ----
    def resolve_col(candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df_comp.columns and c in row.index:
                return c
        return None

    # ---- Rollen-spezifische Definition ----
    if role in ("FW", "Off_MF"):
        y_candidates = ["Gls_Per90", "Gls/90"]
        x_candidates = ["xG_Per90", "xG/90"]
        y_label = "Goals per 90"
        x_label = "xG per 90"
        chart_title = "Big-5 comparison – FW metrics"
    elif role == "MF":
        y_candidates = ["Ast_Per90", "Ast/90"]
        x_candidates = ["xAG_Per90", "xAG/90", "xA_Per90", "xA/90"]
        y_label = "Assists per 90"
        x_label = "xAG per 90"
        chart_title = "Big-5 comparison – MF metrics"
    else:
        # Default: Defender-Metriken
        y_candidates = ["TklW_Per90", "TklW/90"]
        x_candidates = ["Int_Per90", "Int/90"]
        y_label = "Tackles won per 90"
        x_label = "Interceptions per 90"
        chart_title = "Big-5 comparison – DF metrics"

    x_col = resolve_col(x_candidates)
    y_col = resolve_col(y_candidates)

    if x_col is None or y_col is None:
        st.info("Not enough data available to build the role scatter plot for this player.")
        return None

    # ---- Vergleichsgruppe vorbereiten ----
    cols_needed = ["Player", "Squad", x_col, y_col]
    if "Season" in df_comp.columns:
        cols_needed.append("Season")

    plot_df = df_comp[cols_needed].copy()
    plot_df = plot_df.dropna(subset=[x_col, y_col])

    if plot_df.empty:
        st.info("No comparison data available for role scatter plot.")
        return None

    # markiere ausgewählten Spieler
    plot_df["is_player"] = plot_df["Player"] == row.get("Player")

    
    # ---- Outlier nur für Peers filtern, Spieler immer behalten ----
    s_x = pd.to_numeric(plot_df[x_col], errors="coerce")
    s_y = pd.to_numeric(plot_df[y_col], errors="coerce")

    qx = s_x.quantile(0.99)
    qy = s_y.quantile(0.99)

    player_name = row.get("Player")
    player_mask = plot_df["Player"] == player_name

    if np.isfinite(qx) and np.isfinite(qy):
        # Nur Peers oberhalb des 99%-Quantils rausschneiden,
        # der aktuelle Spieler (player_mask) bleibt IMMER drin.
        plot_df = plot_df[((s_x <= qx) & (s_y <= qy)) | player_mask]

    if plot_df.empty:
        st.info("No comparison data available for role scatter plot.")
        return None

    # markiere ausgewählten Spieler (nach dem Filtern neu berechnen)
    plot_df["is_player"] = plot_df["Player"] == row.get("Player")

    # ---- Hilfsfunktion für robuste Skalen ----
    def compute_domain(series, default_max: float = 1.0, q: float = 0.99) -> tuple[float, float]:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return (0.0, default_max)

        try:
            qv = s.quantile(q)
        except Exception:
            qv = s.max()

        if not np.isfinite(qv) or qv <= 0:
            qv = s.max()

        if not np.isfinite(qv) or qv <= 0:
            qv = default_max

        upper = float(qv) * 1.05
        if upper <= 0:
            upper = default_max

        # Sicherstellen, dass max > min
        if upper <= 0.0:
            upper = default_max

        return (0.0, upper)

    # ---- Skalen: FW feste Range, MF/DF dynamisch ----
    if role in ("FW", "Off_MF"):
        # feste, stabile Skala für Stürmer
        x_domain = (0.0, 0.9)   # xG per 90
        y_domain = (0.0, 1.3)   # Goals per 90
    else:
        # MF / DF: quantilbasiert, aber robust
        x_domain = compute_domain(plot_df[x_col], default_max=1.0, q=0.99)
        y_domain = compute_domain(plot_df[y_col], default_max=1.0, q=0.99)

    base = alt.Chart(plot_df)

    # Peers
    peers = base.transform_filter(
        alt.datum.is_player == False
    ).mark_circle(
        size=35,
        opacity=0.18,
    ).encode(
        x=alt.X(
            f"{x_col}:Q",
            title=x_label,
            scale=alt.Scale(domain=list(x_domain)),
        ),
        y=alt.Y(
            f"{y_col}:Q",
            title=y_label,
            scale=alt.Scale(domain=list(y_domain)),
        ),
        tooltip=["Player", "Squad"] + (["Season"] if "Season" in plot_df.columns else []),
    )

    
    # Selected Player
    player_layer = base.transform_filter(
        alt.datum.is_player == True
    ).mark_circle(
        size=140,          # größer
        opacity=1.0,       # volle Deckkraft
        stroke="#F9FAFB",  # dünne helle Umrandung
        strokeWidth=1.5,
    ).encode(
        x=alt.X(f"{x_col}:Q"),
        y=alt.Y(f"{y_col}:Q"),
        color=alt.value(VALUE_COLOR),
        tooltip=["Player", "Squad"] + (["Season"] if "Season" in plot_df.columns else []),
    )

    player_label = base.transform_filter(
        alt.datum.is_player == True
    ).mark_text(
        dx=30,              # leichte Verschiebung nach rechts
        dy=-15,             # leichte Verschiebung nach oben
        fontSize=11,
        fontWeight="bold",
        color="#E5E7EB",
    ).encode(
        x=alt.X(f"{x_col}:Q"),
        y=alt.Y(f"{y_col}:Q"),
        text="Player",
    )

    chart = (peers + player_layer + player_label).properties(
        height=550,
        title=chart_title,
    ).configure_axis(
        grid=True,
        gridOpacity=0.15,
        gridColor="#4b5563",
        domain=True,
        domainColor="#4b5563",
        labelColor="#E5E7EB",
        titleColor="#E5E7EB",
    ).configure_title(
        color="#E5E7EB",
        fontSize=13,
        anchor="start",
    ).configure_view(
        strokeWidth=0
    )

    return chart

@st.cache_data
def load_feature_table_for_season(season: str) -> pd.DataFrame:
    """
    Loads the players_data_light-<Season>.csv from Data/Raw,
    applies position logic and per-90 metric calculation,
    and returns a feature table for the pizza chart.
    """
    from pathlib import Path
    from src.processing import prepare_positions, add_standard_per90

    root = Path(__file__).resolve().parent
    raw_dir = root / "Data" / "Raw"

    # the Season value may contain "/", so normalize it for filenames
    season_safe = season.replace("/", "-")
    csv_path = raw_dir / f"players_data_light-{season_safe}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"No feature file found for season: {season}")

    df = pd.read_csv(csv_path)

    # Apply position logic to get FW/MF/DF/etc
    df = prepare_positions(df)

    # Add per-90 metrics (Succ_Per90, TB_Per90, etc)
    df = add_standard_per90(df)

    return df


# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="PlayerScore",
        layout="wide",
        page_icon="⚽",
    )

    # ---- Custom CSS for a more polished look ----
    st.markdown(
        """
        <style>
        /* Global font tweaks */
        html, body, [class*="css"]  {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        /* Main title + subtitle */
        .ps-title {
            font-size: 2.4rem;
            font-weight: 700;
            color: #F9FAFB;  /* near-white, fits dark theme */
            margin-bottom: 0.15rem;
        }
        .ps-subtitle {
            font-size: 0.95rem;
            color: #94A3B8;  /* subtle grey for subtitle */
            margin-bottom: 1.6rem;
        }

        /* Dataframe font size */
        .stDataFrame tbody td {
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    df_all, df_agg, df_squad = load_data()

    # ---- Simplify positions to FW / MF / DF ----
    pos_map = {
        "FW": "FW",
        "Off_MF": "FW",  # Offensiver Mittelfeldspieler -> Offensivrolle
        "MF": "MF",
        "Def_MF": "DF",  # Defensiver Mittelfeldspieler -> Defensivrolle
        "DF": "DF",
    }

    df_all["Pos"] = df_all["Pos"].map(pos_map).fillna(df_all["Pos"])


    if df_all.empty:
        st.info("No processed data found yet. Run the pipeline locally and push the CSVs or Kaggle sync.")
        st.stop()

    # ===================== Sidebar: View selection =====================
    st.sidebar.header("View")
    mode = st.sidebar.radio(
        "Select mode",
        ["Home", "Player profile", "Top lists"],
    )

    if mode in ("Player profile", "Top lists"):
        st.markdown(
        "Score scale (0–1000): 🟣 Exceptional ≥ 900  ·  🟢 World Class ≥ 750  ·  🔵 Top Starter ≥ 400  ·  🟡 Solid Squad Player ≥ 200  ·  ⚪️ Below Big-5 Level < 200"
    )
        
    # ---- Global Footer (appears on all modes) ----
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            right: 15px;
            bottom: 10px;
            padding: 5px 10px;
            opacity: 0.75;
            font-size: 13px;
        }
        </style>
        <div class="footer">
            Data Source: Fbref | Created by <b>TwinAnalytics</b>
        </div>
        """,
        unsafe_allow_html=True
    )


    # ==================================================================
    # MODE 0: HOME / LANDING PAGE
    # ==================================================================
    if mode == "Home":
        # Hero section
        st.markdown(
            f"""
            <h1 style="color:{VALUE_COLOR}; margin-bottom:0.25rem;">PlayerScore</h1>
            <p style="font-size:1.05rem; margin-top:0;">
                Advanced football player analytics across leagues and seasons
            </p>
            """,
            unsafe_allow_html=True,
        )

        # What is PlayerScore?
        st.markdown(
             f"""
             <h3 style="color:{VALUE_COLOR};">What is PlayerScore?</h3>
            """,
            unsafe_allow_html=True,
        )
        # Intro sections
        st.markdown(
            """
            PlayerScore is built on a self-developed, data-driven scoring framework that makes it possible 
            to compare football players across different leagues — regardless of country, competition level, 
            or data availability.

            The analysis engine continuously processes the latest publicly available performance data from FBref 
            and combines it with a custom scoring logic that uses comprehensive Big-5 metrics as well as reduced 
            “Light” metrics for leagues with fewer statistical features.
            """
        )

        # What does PlayerScore deliver?
        st.markdown(
             f"""
            <h3 style="color:{VALUE_COLOR};">What does PlayerScore deliver?</h3>
            """,
            unsafe_allow_html=True,
        )


        st.markdown(
            """
            Through this approach, PlayerScore provides:

            - ⚽ **Quantifiable player performance** with a unified scoring scale  
            - 🌍 **Comparability across leagues and countries**
            - 🔍 **Transparent and structured stats**, tailored to attacker, midfield, and defender roles  
            - 🧠 **Insights powered by modern analytical methods** instead of pure gut feeling  
            - 📈 A **continuously growing global football database**, updated via automated pipelines  
            """
        )

        st.markdown(
            """
            The PlayerScore database already contains several thousand players from Europe’s top-5 leagues
            and multiple historical seasons — and it grows with every pipeline run.
            """
        )

        st.info(
            "Use the sidebar to switch to **Player profile** to explore an individual player, "
            "or to **Top lists** to see ranked players by role and season."
        )

        st.stop()



        st.info("Wechsle links im Sidebar auf **Player profile**, um mit einem Spielerprofil zu starten.")
        st.stop()



    # ==================================================================
    # MODE 1: PLAYER PROFILE
    # ==================================================================
    if mode == "Player profile":
        st.sidebar.subheader("Profile filters")

        # ----- Player-Auswahl -----
        players_all = sorted(df_all["Player"].dropna().unique())
        if not players_all:
            st.warning("No players found in the dataset.")
            return

        placeholder = "Select a player..."
        player_options = [placeholder] + players_all

        current_selection = st.session_state.get("selected_player", placeholder)
        if current_selection not in player_options:
            current_selection = placeholder

        player = st.sidebar.selectbox(
            "Player",
            player_options,
            index=player_options.index(current_selection),
        )
        st.session_state["selected_player"] = player

        if player == placeholder:
            st.subheader("Player profile")
            st.info("Please select a player in the sidebar on the left to view their profile.")
            return

        # Alle Saisons dieses Spielers (für Filter & Career)
        df_player_all = df_all[df_all["Player"] == player].copy()

        if "Season" in df_player_all.columns:
            seasons = sorted(df_player_all["Season"].dropna().unique())
        else:
            seasons = sorted(df_all["Season"].dropna().unique())

        if not seasons:
            st.warning("No seasons found for this player.")
            return

        # ----- Profile view: Per season vs Career -----
        st.sidebar.markdown("---")
        profile_view = st.sidebar.radio(
            "Profile view",
            ["Per season", "Career"],
            key="profile_view",
        )

        # ----- Season nur im Per-season-Modus -----
        season = None
        if profile_view == "Per season":
            default_season_idx = len(seasons) - 1 if seasons else 0
            season = st.sidebar.selectbox(
                "Season",
                seasons,
                index=default_season_idx,
                key=f"profile_season_{player}",
            )

        # ----- Player-View-Daten -----
        if profile_view == "Per season" and season is not None:
            df_player = df_player_all[df_player_all["Season"] == season].copy()
        else:
            df_player = df_player_all.copy()

        # ----- Header -----
        if profile_view == "Per season" and season is not None:
            st.subheader(f"Player Profile – {player}")
        else:
            st.subheader(f"Player Profile – {player}")

        # ----- Caption: Position | Squad oder typische Rolle -----
        typical_pos = df_player_all["Pos"].dropna().mode()
        typical_pos = typical_pos.iloc[0] if not typical_pos.empty else None

        if profile_view == "Per season" and not df_player.empty:
            pos_txt = df_player["Pos"].iloc[0] if "Pos" in df_player.columns else None
            squad_txt = df_player["Squad"].iloc[0] if "Squad" in df_player.columns else None
            details = " | ".join(x for x in [pos_txt, squad_txt] if x)
            if details:
                st.caption(details)
        else:
            st.caption(get_role_label(typical_pos))

        # ----- Primäre Rolle und Score-Dimension -----
        role = typical_pos or (df_player["Pos"].iloc[0] if not df_player.empty and "Pos" in df_player.columns else None)

        if role in ("FW", "Off_MF"):
            primary_dim = "Offensive"
            score_col = "OffScore_abs"
            squad_col = "OffScore_squad"
        elif role == "MF":
            primary_dim = "Midfield"
            score_col = "MidScore_abs"
            squad_col = "MidScore_squad"
        elif role in ("DF", "Def_MF"):
            primary_dim = "Defensive"
            score_col = "DefScore_abs"
            squad_col = "DefScore_squad"
        else:
            primary_dim = None
            score_col = None
            squad_col = None

        # ----- Season / Seasons Label -----
        if profile_view == "Per season" and season is not None:
            st.markdown(f"Season: {season}")
        else:
            if "Season" in df_player_all.columns and not df_player_all.empty:
                first_season = df_player_all["Season"].min()
                last_season = df_player_all["Season"].max()
                st.markdown(f"Seasons: {first_season} – {last_season}")
            else:
                st.markdown("Seasons: n/a")

        # ----- MainScore / MainBand berechnen -----
        if not df_player.empty:
            df_player = df_player.copy()
            df_player[["MainScore", "MainBand"]] = df_player.apply(
                get_primary_score_and_band,
                axis=1,
                result_type="expand",
            )

        # ===================== SUMMARY =====================
        st.markdown("### Summary")

        col1, col2, col3, col4 = st.columns(4)

        if profile_view == "Per season":
            # ---- Per-season Summary ----
            age = None
            n_90s = None
            main_score = None
            main_band = None

            if not df_player.empty:
                if "Age" in df_player.columns:
                    age = df_player["Age"].iloc[0]
                if "90s" in df_player.columns:
                    n_90s = df_player["90s"].iloc[0]
                if "MainScore" in df_player.columns:
                    main_score = df_player["MainScore"].iloc[0]
                if "MainBand" in df_player.columns:
                    main_band = df_player["MainBand"].iloc[0]

            band_icon = BAND_ICONS.get(main_band, main_band) if main_band is not None else None

            # Age
            with col1:
                age_value = f"{age:.0f}" if isinstance(age, (int, float)) and not pd.isna(age) else "n/a"
                st.markdown(
                    f"""
                    <div style="
                        border-radius: 0.5rem;
                        padding: 0.4rem 0.5rem;
                        text-align: left;
                        background-color: rgba(15, 23, 42, 0.35);
                    ">
                        <div style="font-size: 0.85rem; opacity: 0.9; margin-bottom: 0.1rem; color: #ffffff;">Age</div>
                        <div style="font-size: 1.45rem; font-weight: 600; color: #ffffff;">{age_value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 90s played
            with col2:
                n_90s_value = f"{n_90s:.1f}" if isinstance(n_90s, (int, float)) and not pd.isna(n_90s) else "n/a"
                st.markdown(
                    f"""
                    <div style="
                        border-radius: 0.5rem;
                        padding: 0.4rem 0.5rem;
                        text-align: left;
                        background-color: rgba(15, 23, 42, 0.35);
                    ">
                        <div style="font-size: 0.85rem; opacity: 0.9; margin-bottom: 0.1rem; color: #ffffff;">90s played</div>
                        <div style="font-size: 1.45rem; font-weight: 600; color: #ffffff;">{n_90s_value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Score
            with col3:
                score_value = f"{main_score:.0f}" if main_score is not None and not pd.isna(main_score) else "n/a"
                st.markdown(
                    f"""
                    <div style="
                        border-radius: 0.5rem;
                        padding: 0.4rem 0.5rem;
                        text-align: left;
                        background-color: rgba(15, 23, 42, 0.35);
                    ">
                        <div style="font-size: 0.85rem; opacity: 0.9; margin-bottom: 0.1rem; color: #ffffff;">Score</div>
                        <div style="font-size: 2.5rem; font-weight: 600; color: {VALUE_COLOR};">{score_value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Band
            with col4:
                band_value = band_icon if band_icon is not None else "n/a"
                st.markdown(
                    f"""
                    <div style="
                        border-radius: 0.5rem;
                        padding: 0.4rem 0.5rem;
                        text-align: left;
                        background-color: rgba(15, 23, 42, 0.35);
                    ">
                        <div style="font-size: 0.85rem; opacity: 0.9; margin-bottom: 0.5rem; color: #ffffff;">Band</div>
                        <div style="font-size: 1.25rem; font-weight: 600; color: #ffffff;">{band_value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:
            # ---- Career Summary ----
            if "Season" in df_player_all.columns and "Age" in df_player_all.columns:
                df_sorted = df_player_all.sort_values("Season")
                age_career = df_sorted["Age"].iloc[-1]
            else:
                age_career = None

            total_90s = float(df_player_all["90s"].sum()) if "90s" in df_player_all.columns else 0.0

            avg_score = None
            if "MainScore" in df_player.columns and df_player["MainScore"].notna().any():
                avg_score = df_player["MainScore"].mean()

            avg_band_label = band_from_score(avg_score) if avg_score is not None else None

            # Age (last season)
            with col1:
                age_value = f"{age_career:.0f}" if isinstance(age_career, (int, float)) and not pd.isna(age_career) else "n/a"
                st.markdown(
                    f"""
                    <div style="
                        border-radius: 0.5rem;
                        padding: 0.4rem 0.5rem;
                        text-align: left;
                        background-color: rgba(15, 23, 42, 0.35);
                    ">
                        <div style="font-size: 0.85rem; opacity: 0.9; margin-bottom: 0.1rem; color: #ffffff;">Age</div>
                        <div style="font-size: 1.45rem; font-weight: 600; color: #ffffff;">{age_value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 90s played (career)
            with col2:
                n_90s_value = f"{total_90s:.1f}" if total_90s > 0 else "n/a"
                st.markdown(
                    f"""
                    <div style="
                        border-radius: 0.5rem;
                        padding: 0.4rem 0.5rem;
                        text-align: left;
                        background-color: rgba(15, 23, 42, 0.35);
                    ">
                        <div style="font-size: 0.85rem; opacity: 0.9; margin-bottom: 0.1rem; color: #ffffff;">90s Played (Career)</div>
                        <div style="font-size: 1.45rem; font-weight: 600; color: #ffffff;">{n_90s_value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Average score (career)
            with col3:
                score_value = f"{avg_score:.0f}" if avg_score is not None and not pd.isna(avg_score) else "n/a"
                st.markdown(
                    f"""
                    <div style="
                        border-radius: 0.5rem;
                        padding: 0.4rem 0.5rem;
                        text-align: left;
                        background-color: rgba(15, 23, 42, 0.35);
                    ">
                        <div style="font-size: 0.85rem; opacity: 0.9; margin-bottom: 0.1rem; color: #ffffff;">Score (Career Avg)</div>
                        <div style="font-size: 2.5rem; font-weight: 600; color: {VALUE_COLOR};">{score_value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Band (career avg)
            with col4:
                band_value = avg_band_label if avg_band_label is not None else "n/a"
                st.markdown(
                    f"""
                    <div style="
                        border-radius: 0.5rem;
                        padding: 0.4rem 0.5rem;
                        text-align: left;
                        background-color: rgba(15, 23, 42, 0.35);
                    ">
                        <div style="font-size: 0.85rem; opacity: 0.9; margin-bottom: 0.5rem; color: #ffffff;">Band (Career Avg)</div>
                        <div style="font-size: 1.25rem; font-weight: 600; color: #ffffff;">{band_value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ===================== ROLE METRICS =====================
        st.markdown("### Role metrics")

        if profile_view == "Per season":
            # 1) Feature-Tabelle für diese Season laden
            try:
                df_features_season = load_feature_table_for_season(season)
            except FileNotFoundError:
                st.info("No raw feature data found for this season.")
                df_features_season = pd.DataFrame()

            # 2) Spielerzeilen aus Feature-Tabelle ziehen
            if not df_features_season.empty:
                df_feat_player = df_features_season[
                    df_features_season["Player"] == player
                ].copy()
            else:
                df_feat_player = pd.DataFrame()

            if role is not None and not df_feat_player.empty:
                col_pizza, col_scatter = st.columns(2)

                # --- links: Pizza-Chart ---
                with col_pizza:
                    fig = render_pizza_chart(df_features_season, df_feat_player, role, season)
                    if fig is not None:
                        st.pyplot(fig, use_container_width=True)

                # --- rechts: Scatter-Plot ---
                with col_scatter:
                    scatter_chart = render_role_scatter(df_features_season, df_feat_player, role)
                    if scatter_chart is not None:
                        st.altair_chart(scatter_chart, use_container_width=True)
                    

        else:
            # Career-Pizza
            if "Season" in df_player_all.columns:
                player_seasons = sorted(df_player_all["Season"].dropna().unique())
            else:
                player_seasons = []

            if not player_seasons or role is None:
                st.info("Not enough career data available to build a pizza chart for this player.")
            else:
                col_chart, _ = st.columns([1, 1])
                with col_chart:
                    fig = render_career_pizza_chart(player, role, player_seasons)
                    if fig is not None:
                        st.pyplot(fig)


        # ===================== SCORE TREND =====================
        if profile_view == "Career":

            st.markdown("### Career Score Trend")

            # automatische Score-Spalte basierend auf Rolle
            if role == "FW":
                score_col = "OffScore_abs"
                score_label = "Offensive score"
            elif role == "MF":
                score_col = "MidScore_abs"
                score_label = "Midfield score"
            elif role == "DF":
                score_col = "DefScore_abs"
                score_label = "Defensive score"
            else:
                st.info("No primary role score available for this player.")
                return

            # Daten vorbereiten: Season, Squad, Score
            plot_df = (
                df_player_all[["Season", "Squad", score_col]]
                .dropna(subset=[score_col])
                .sort_values("Season")
            )



            # feste Y-Achse + Ticks 0 / 500 / 1000
            y_enc = alt.Y(
                f"{score_col}:Q",
                title="Score",
                scale=alt.Scale(domain=[0, 1100]),
                axis=alt.Axis(values=[0, 500, 1000]),
            )

            # smoothe Kurve
            line = (
                alt.Chart(plot_df)
                .mark_line(
                    point=False,
                    strokeWidth=2,
                    interpolate="monotone",
                    color=VALUE_COLOR,
                )
                .encode(
                    x=alt.X("Season:O", title="Season"),
                    y=y_enc,
                    tooltip=[
                        alt.Tooltip("Season:O", title="Season"),
                        alt.Tooltip("Squad:N", title="Squad"),
                        alt.Tooltip(f"{score_col}:Q", title="Score", format=".0f"),
                    ],
                )
            )
    

            # Punkte (exakte Werte)
            points = (
                alt.Chart(plot_df)
                .mark_point(
                    filled=True,
                    size=70,
                    color=VALUE_COLOR,
                )
                .encode(
                    x="Season:O",
                    y=f"{score_col}:Q",
                    tooltip=[
                        alt.Tooltip("Season:O", title="Season"),
                        alt.Tooltip("Squad:N", title="Squad"),
                        alt.Tooltip(f"{score_col}:Q", title="Score", format=".0f"),
                    ],
                )
            )

            # Labels über den Punkten
            labels = (
                alt.Chart(plot_df)
                .mark_text(
                    dy=-10,
                    color="#e5e7eb",
                    fontSize=12,
                    fontWeight="bold",
                )
                .encode(
                    x="Season:O",
                    y=f"{score_col}:Q",
                    text=alt.Text(f"{score_col}:Q", format=".0f"),
                )
            )

      

            # ----------------- Dezente Band-Linien + Labels rechts -----------------
            # letzte Saison als Anker für die Label-X-Position
            last_season = plot_df["Season"].iloc[-1]

            band_data = pd.DataFrame(
                {
                    "y": [200, 400, 750, 900],
                    "label": [
                        "200  •  Solid squad",
                        "400  •  Top starter",
                        "750  •  World class",
                        "900  •  Exceptional",
                    ],
                    "x": [last_season] * 4,   # alle Labels an der letzten Season verankern
                }
            )

            band_lines = (
                alt.Chart(band_data)
                .mark_rule(
                    strokeDash=[4, 4],
                    strokeWidth=0.6,
                    opacity=0.4,
                    color="#6b7280",   # dezentes Grau
                )
                .encode(
                    y="y:Q",
                )
            )

            # Labels RECHTS von der Kurve (an der letzten Saison, mit dx nach rechts)
            band_labels = (
                alt.Chart(band_data)
                .mark_text(
                    align="left",          # Text linksbündig
                    baseline="middle",
                    dx=30,                  # Abstand nach rechts von der letzten Saison-Position
                    color="#e5e7eb",       # helles Grau
                    fontSize=9,
                )
                .encode(
                    x="x:O",               # an letzter Season verankert
                    y="y:Q",
                    text="label:N",
                )
            )

            chart = (
                (line + points + labels + band_lines + band_labels)
                .properties(
                    height=280,
                    title="",
                )
                .configure_title(
                    color="#e5e7eb",
                    fontSize=16,
                )
                .configure_axis(
                    grid=False,
                    ticks=True,
                    tickColor="#6b7280",
                    tickSize=4,
                    domain=True,
                    domainColor="#6b7280",
                    labelColor="#e5e7eb",
                    titleColor="#e5e7eb",
                )
                .configure_view(strokeWidth=0)
            )

            st.altair_chart(chart, use_container_width=True)




    # ==================================================================
    # MODE 2: TOP LISTS
    # ==================================================================
    else:
        st.sidebar.subheader("Top list filters")

        # ----- Season filter (default = latest season) -----
        seasons = sorted(df_all["Season"].dropna().unique())
        default_season_idx = len(seasons) - 1 if seasons else 0
        season = st.sidebar.selectbox(
            "Season",
            seasons,
            index=default_season_idx,
            key="toplists_season",
        )

        # Ausgangsbasis für Toplists (eine Saison)
        df_view = df_all[df_all["Season"] == season].copy()

        # ----- Club filter (persistent across seasons) -----
        if "Squad" in df_view.columns:
            clubs = sorted(df_view["Squad"].dropna().unique())
            club_options = ["All"] + clubs

            # Session-State für Club-Auswahl initialisieren
            if "toplists_club" not in st.session_state:
                st.session_state["toplists_club"] = "All"

            # Falls der aktuell gespeicherte Club in dieser Saison nicht vorkommt -> auf "All" zurück
            if st.session_state["toplists_club"] not in club_options:
                st.session_state["toplists_club"] = "All"

            # Selectbox: nutzt den gespeicherten Wert als Default
            club_sel = st.sidebar.selectbox(
                "Club",
                club_options,
                index=club_options.index(st.session_state["toplists_club"]),
                key="toplists_club",
            )

            # Filtern nach Auswahl (außer "All")
            if club_sel != "All":
                df_view = df_view[df_view["Squad"] == club_sel]

        # ----- Positions: Checkbox-Gruppe (Standard: alle an, persistent) -----
        if "Pos" in df_view.columns:
            pos_values = sorted(df_view["Pos"].dropna().unique())
            st.sidebar.markdown("**Positions**")

            selected_positions = []
            for pos in pos_values:
                checked = st.sidebar.checkbox(
                    pos,
                    value=True,              # beim ersten Laden: alle an
                    key=f"top_pos_{pos}",   # persistent über Seasons
                )
                if checked:
                    selected_positions.append(pos)

            if selected_positions:
                df_view = df_view[df_view["Pos"].isin(selected_positions)]

        # ----- Minutes filter -----
        min_90s = st.sidebar.slider(
            "Minimum 90s played",
            1.0,
            40.0,
            5.0,
            0.5,
            key="top_min_90s",
        )
        if "90s" in df_view.columns:
            df_view = df_view[df_view["90s"] >= min_90s]

        # ----- Age filter (optional, persistent) -----
        if "Age" in df_view.columns:
            st.sidebar.markdown("**Age filter**")
            use_age_filter = st.sidebar.checkbox(
                "Enable age filter (e.g. U23)",
                value=False,
                key="top_use_age_filter",
            )

            if use_age_filter:
                age_numeric = pd.to_numeric(df_view["Age"], errors="coerce")
                if age_numeric.notna().any():
                    min_age = int(age_numeric.min())
                    max_age = int(age_numeric.max())

                    # ---- Session-State für max Age initialisieren / clampen ----
                    if "top_age_max" not in st.session_state:
                        # Startwert nur beim allerersten Mal
                        st.session_state["top_age_max"] = min(23, max_age)
                    else:
                        # Falls sich der Altersbereich mit der Season ändert:
                        current = st.session_state["top_age_max"]
                        if current < min_age:
                            st.session_state["top_age_max"] = min_age
                        elif current > max_age:
                            st.session_state["top_age_max"] = max_age

                    max_age_selected = st.sidebar.slider(
                        "Maximum age",
                        min_value=min_age,
                        max_value=max_age,
                        value=st.session_state["top_age_max"],
                        step=1,
                        key="top_age_max",
                    )

                    df_view = df_view[age_numeric <= max_age_selected]


        # ----- Primary score & band je Spieler bestimmen -----
        if df_view.empty:
            st.warning("No players found for the selected filters.")
            return

        df_view[["MainScore", "MainBand"]] = df_view.apply(
            get_primary_score_and_band,
            axis=1,
            result_type="expand",
        )

        # Spieler ohne relevanten Score (z.B. GK) raus
        df_view = df_view[df_view["MainScore"].notna()].copy()

        if df_view.empty:
            st.warning("No players with a primary score found for the selected filters.")
            return

        # ----- Bands: Checkbox-Gruppe (Standard: alle an, persistent) -----
        if "MainBand" in df_view.columns:
            bands_available = list(df_view["MainBand"].dropna().unique())
            bands_sorted = [b for b in BAND_ORDER if b in bands_available] + [
                b for b in bands_available if b not in BAND_ORDER
            ]

            st.sidebar.markdown("**Bands**")

            selected_bands = []
            for b in bands_sorted:
                label = BAND_ICONS.get(b, b)
                checked = st.sidebar.checkbox(
                    label,
                    value=True,               # beim ersten Laden: alle aktiv
                    key=f"top_band_{b}",      # persistent über Seasons
                )
                if checked:
                    selected_bands.append(b)

            if selected_bands:
                df_view = df_view[df_view["MainBand"].isin(selected_bands)]

        if df_view.empty:
            st.warning("No players found after applying band filters.")
            return

        # ----- Top N -----
        top_n = st.sidebar.slider(
            "Top N players",
            10,
            200,
            50,
            10,
            key="top_topn",
        )

        st.markdown(f"### Top {top_n} players by primary role score – Season {season}")

        # Spalten für die Tabelle
        cols_top = [
            "Player",
            "Squad",
            "Age",
            "Pos",
            "Min",
            "90s",
            "MainScore",
            "MainBand",
        ]
        cols_top = [c for c in cols_top if c in df_view.columns]

        df_top = (
            df_view[cols_top]
            .sort_values("MainScore", ascending=False)
            .head(top_n)
        )

        # Band mit Icons ersetzen
        display_cols = [c for c in cols_top if c not in ("MainBand",)]
        if "MainBand" in df_top.columns:
            df_top["Band"] = df_top["MainBand"].map(BAND_ICONS).fillna(df_top["MainBand"])
            display_cols.append("Band")

        # Anzeige: MainScore als "Score" benennen
        df_top_display = df_top[display_cols].rename(columns={"MainScore": "Score"})

        # 👉 Score als Integer ohne Nachkommastellen anzeigen
        if "Score" in df_top_display.columns:
            df_top_display["Score"] = (
                df_top_display["Score"]
                .round()
                .astype("Int64")
            )
        st.dataframe(df_top_display, use_container_width=True)

        # Band-Verteilung (gefilterte Spieler)
        if "MainBand" in df_view.columns:
            st.markdown("### Band distribution (filtered players)")

            # Schritt 1: Counts berechnen und in DataFrame umwandeln
            band_counts = (
                df_view["MainBand"]
                .value_counts()
                .reindex(BAND_ORDER, fill_value=0)
            )

            band_counts = band_counts.reset_index()
            band_counts.columns = ["Band", "Count"]  # sorgt garantiert für die Spaltennamen

            # Schritt 2: Kategorie-Reihenfolge explizit setzen
            band_counts["Band"] = pd.Categorical(
                band_counts["Band"],
                categories=BAND_ORDER,
                ordered=True,
            )

            # Schritt 3: Altair-Bar-Chart mit fixer Sortierung
            chart = (
                alt.Chart(band_counts)
                 .mark_bar(size=30)
                 .encode(
                     x=alt.X(
                        "Band:N",
                        sort=BAND_ORDER,
                        title="Band",
                         scale=alt.Scale(paddingInner=0.4, paddingOuter=0.2),
                ),
                    y=alt.Y("Count:Q", title="Number of players"),
                    tooltip=["Band", "Count"],
                    color=alt.value(PRIMARY_COLOR),  # <- same color as Home headline
                )
             )
            st.altair_chart(chart, use_container_width=True)


if __name__ == "__main__":
    main()
