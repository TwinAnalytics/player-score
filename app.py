import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mplsoccer import PyPizza
from streamlit.components.v1 import html as st_html

from src.processing import prepare_positions, add_standard_per90, add_squad_per90
from src.scoring import score_band_5

from src.charts.team_scatter import (
    render_team_scatter_under_table,
    render_big5_facet_scatter,
    render_scatter_linkedin_optimized,
    render_budget_scatter,
)
from src.charts.player_card_export import generate_player_card_png
from src.charts.player_report_pdf import generate_player_report_pdf
from src.club_crests import get_crest_b64, get_crest_path
from src.similar_players import find_similar_players
from src.charts.age_curve import render_age_curve

import io

import matplotlib
matplotlib.use("Agg")

# -------------------------------------------------------------------
# Global Matplotlib + Theme
# -------------------------------------------------------------------
plt.rcParams["figure.dpi"] = 200      # höhere Render-Auflösung
plt.rcParams["savefig.dpi"] = 200

APP_BG = "#0D1117"
GRID_COLOR = "#374151"
TEXT_COLOR = "#e5e7eb"
SLICE_COLOR = "cornflowerblue"
VALUE_COLOR = "#00B8A9"

# drei Töne davon für die Pizza-Gruppen
COLOR_POSSESSION = "#80F5E3"   
COLOR_ATTACKING  = "#00B8A9"   
COLOR_DEFENDING  = "#006058"   

BIG5_COMPS = {
    "eng Premier League",
    "es La Liga",
    "de Bundesliga",
    "it Serie A",
    "fr Ligue 1",
}


OFFENSE_PRESETS = {
        "Core Offense": [
            "Gls/90",
            "xG/90",
            "npxG/90",
            "Shots on target/90",
            "xAG/90",
            "KP/90",
        ],
        "Chance Creation": [
            "xAG/90",
            "KP/90",
            "Prog Passes/90",
            "Prog Carries/90",
            "Att 3rd/90",
            "Att Pen/90",
        ],
        "Box Presence": [
            "Att 3rd/90",
            "Att Pen/90",
            "Shots on target/90",
            "npxG/90",
        ],
        "Custom": None,  # frei wählbar nur über Multiselect
    }

DEFENSE_PRESETS = {
        "Core Defense": [
            "TklW/90",
            "Int/90",
            "Clr/90",
            "Blocks/90",
            "Fouls/90",
        ],
        "Box Defense": [
            "Clr/90",
            "Blocks/90",
            "GA/90",
            "PSxG+/- /90",
        ],
        "Custom": None,
    }

# Welche Spalten werden für das Radar benutzt?
# label (Radar) -> (Spaltenname im df_raw, invert: weniger = besser?)

OFFENSE_METRICS_CONFIG: dict[str, tuple[str, bool]] = {
    # label        # Spalte in df_raw           invert?
    "npxG":                      ("npxG/90",             False),
    "npxG/Shot":                 ("npxG/Sh",             False),
    "Shots":                     ("Sh/90",               False),
    "Shots \non \nTarget":       ("SoT/90",              False),
    "xAG":                       ("xAG/90",              False),
    "Passes \nto \nPA":          ("PPA/90",              False),
    "Carries \nto \nPA":         ("CPA/90",              False),
    "Touches \nin \nPA":         ("Att Pen/90",          False),
    "Pass \ncompletion %":       ("Cmp%",                False),
    "Goal \nCreation \nAction":  ("GCA90",               False),
    "Shot \nCreation \nAction":  ("SCA90",               False),
    "Key \nPasses":              ("KP/90",               False),
}

DEFENSE_METRICS_CONFIG: dict[str, tuple[str, bool]] = {
    # defensiv: teilweise invertiert (weniger = besser)
    "xG \nAllowed":             ("onxGA/90",            True),   # weniger xG conceded besser
    "Goals \nAllowed":          ("GA90",                True),   # weniger Gegentore besser
    "SoT \nAllowed":            ("SoTA/90",             True),   # weniger Schüsse aufs Tor besser
    "Clearances":               ("Clr/90",              False),  # mehr Clearances = aktiver Verteidiger
    "Tackles+Interc.":          ("Tkl+Int/90",          False),
    "PSxG-GA":                  ("PSxG+/-/90",          False),  # höher = besser (mehr gehalten als erwartet)
    "Clean \nSheets %":         ("CS%",                 False),  # höher = mehr Clean Sheets
    "Crosses Faced":            ("Opp/90",              True),   # mehr Flanken gegen dich ist eher schlecht
}

# -------------------------------------------------------------------
# Band labels, colors and icons (single source of truth)
# -------------------------------------------------------------------
BAND_COLORS = {
    "Exceptional":        "#a855f7",
    "World Class":        "#22c55e",
    "Top Starter":        "#3b82f6",
    "Solid Squad Player": "#eab308",
    "Below Big-5 Level":  "#6B7280",
}

BAND_ICONS = {
    "Exceptional":        "🟣 Exceptional",
    "World Class":        "🟢 World Class",
    "Top Starter":        "🔵 Top Starter",
    "Solid Squad Player": "🟡 Solid Squad Player",
    "Below Big-5 Level":  "⚪️ Below Big-5 Level",
}

BAND_ORDER = [
    "Exceptional",
    "World Class",
    "Top Starter",
    "Solid Squad Player",
    "Below Big-5 Level",
]

VERSION_FILE = Path(__file__).resolve().parent / "Data" / "Processed" / "_last_update.txt"

def get_data_version() -> str:
    if VERSION_FILE.exists():
        return VERSION_FILE.read_text().strip()
    return "dev"

def _get_coordinates(n: int) -> np.ndarray:
    """
    Erzeugt n gleichmäßig verteilte Winkel um den Kreis.
    Rückgabe: (n, 3) -> wir benutzen nur Spalte 2 (Winkel in Rad).
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    coords = np.zeros((n, 3), dtype=float)
    coords[:, 2] = angles
    return coords


def _get_indices_between(range_list: np.ndarray,
                         coord_list: np.ndarray,
                         value: float,
                         reverse: bool = False) -> tuple[float, float]:
    """
    Vereinfacht: wir suchen den Range-Punkt, der am nächsten am Wert liegt.
    (Statt Interpolation – optisch reicht das völlig für den Radar.)
    """
    diffs = np.abs(range_list - value)
    idx = int(diffs.argmin())
    x_coord, y_coord = coord_list[idx]
    return float(x_coord), float(y_coord)


class Radar:
    """
    StatsBomb-ähnliche Radar-Chart-Klasse mit konzentrischen Ringen.
    """

    def __init__(
        self,
        background_color: str = "#FFFFFF",
        patch_color: str = "#D6D6D6",
        fontfamily: str = "DejaVu Sans",

        # Label (Metriknamen außen)
        label_fontsize: float = 7,
        label_color: str = "#000000",
        label_weight: str = "bold",

        # Range-Werte (Zahlen auf den Ringen)
        range_fontsize: float = 7,
        range_color: str = "#000000",
        range_weight: str = "normal",
    ):
        self.background_color = background_color
        self.patch_color = patch_color
        self.fontfamily = fontfamily

        self.label_fontsize = label_fontsize
        self.label_color = label_color
        self.label_weight = label_weight

        self.range_fontsize = range_fontsize
        self.range_color = range_color
        self.range_weight = range_weight

    def plot_radar(
        self,
        ranges,
        params,
        values,
        radar_color,
        filename: str | None = None,
        dpi: int = 300,
        title: dict = dict(),
        alphas=[0.6, 0.6],
        compare: bool = False,
        endnote: str | None = None,
        end_size: int = 9,
        end_color: str = "#95919B",
        image=None,
        image_coord=None,
        figax=None,
        **kwargs,
    ):
        """
        ranges: list[(min,max)] pro Metrik (in REALEN Einheiten)
        params: Label-Liste (Strings)
        values:
          - single team: list[float]
          - compare: [list[float], list[float]]
        radar_color:
          - single: "#hex"
          - compare: ["#color_team1", "#color_team2"]
        """

        assert len(ranges) >= 3, "Length of ranges should be >= 3"
        assert len(params) >= 3, "Length of params should be >= 3"

        if compare:
            assert (
                len(values) == len(radar_color) == len(alphas)
            ), "Length for values, radar_color and alphas must match"
        else:
            assert len(values) >= 3, "Length of values should be >= 3"
            assert len(ranges) == len(params) == len(values), \
                "Length of ranges, params, values must match"

        if figax:
            fig, ax = figax
        else:
            fig, ax = plt.subplots(
                figsize=(10, 10),
                facecolor=self.background_color,
            )
            ax.set_facecolor(self.background_color)

        ax.set_aspect("equal")
        ax.set(xlim=(-22, 22), ylim=(-23, 25))

        if isinstance(radar_color, str):
            radar_color = [radar_color, "#D6D6D6"]

        # äußere Labels (Metrik-Namen)
        ax = self.__add_labels(params=params, ax=ax)

        # numerische Ringe
        ax, xy, range_values = self.__add_ranges(ranges=ranges, ax=ax)

        if compare:
            for i in range(len(values)):
                val = values[i]
                vertices = self.__get_vertices(val, xy, range_values)
                ax = self.__plot_circles(
                    ax=ax,
                    radar_color=radar_color[i],
                    vertices=vertices,
                    alpha=alphas[i],
                    compare=True,
                )
        else:
            vertices = self.__get_vertices(values, xy, range_values)
            ax = self.__plot_circles(
                ax=ax,
                radar_color=radar_color,
                vertices=vertices,
            )

        # Endnote
        if endnote is not None:
            y_add = -22.5
            for note in endnote.split("\n"):
                ax.text(
                    22,
                    y_add,
                    note,
                    fontfamily=self.fontfamily,
                    ha="right",
                    fontdict={"color": end_color},
                    fontsize=end_size,
                )
                y_add -= 1

        ax.axis("off")

        if len(title) > 0:
            ax = self.__plot_titles(ax, title)

        if filename:
            fig.savefig(filename, dpi=dpi, bbox_inches="tight")

        return fig, ax

    # ---------- Titel ----------

    def __plot_titles(self, ax, title):
        if title.get("title_color") is None:
            title["title_color"] = "#000000"
        if title.get("subtitle_color") is None:
            title["subtitle_color"] = "#000000"
        if title.get("title_fontsize") is None:
            title["title_fontsize"] = 20
        if title.get("subtitle_fontsize") is None:
            title["subtitle_fontsize"] = 15
        if title.get("title_fontsize_2") is None:
            title["title_fontsize_2"] = title["title_fontsize"]
        if title.get("subtitle_fontsize_2") is None:
            title["subtitle_fontsize_2"] = title["subtitle_fontsize"]

        if title.get("title_name"):
            ax.text(
                -22,
                24,
                title["title_name"],
                fontsize=title["title_fontsize"],
                fontweight="bold",
                fontdict={"color": title["title_color"]},
                fontfamily=self.fontfamily,
            )

        if title.get("subtitle_name"):
            ax.text(
                -22,
                22,
                title["subtitle_name"],
                fontsize=title["subtitle_fontsize"],
                fontdict={"color": title["subtitle_color"]},
                fontfamily=self.fontfamily,
            )

        if title.get("title_color_2") is None:
            title["title_color_2"] = "#000000"
        if title.get("subtitle_color_2") is None:
            title["subtitle_color_2"] = "#000000"

        if title.get("title_name_2"):
            ax.text(
                22,
                24,
                title["title_name_2"],
                fontsize=title["title_fontsize_2"],
                fontweight="bold",
                fontdict={"color": title["title_color_2"]},
                ha="right",
                fontfamily=self.fontfamily,
            )

        if title.get("subtitle_name_2"):
            ax.text(
                22,
                22,
                title["subtitle_name_2"],
                fontsize=title["subtitle_fontsize_2"],
                fontdict={"color": title["subtitle_color_2"]},
                ha="right",
                fontfamily=self.fontfamily,
            )

        return ax

    # ---------- Polygon + Kreise ----------

    def __plot_circles(self, ax, radar_color, vertices, alpha=None, compare=False):
        """
        Polygon + Kreise zeichnen.
        """

        radius = [3.35, 6.7, 10.05, 13.4, 16.75]
        lw_circle, zorder_circle = 18, 1

        # Polygon
        if compare:
            radar_1 = Polygon(
                vertices,
                fc=radar_color,
                ec="None",
                lw=0,
                alpha=alpha if alpha is not None else 0.7,
                zorder=zorder_circle,
            )
            ax.add_patch(radar_1)
        else:
            radar_1 = Polygon(
                vertices,
                fc=radar_color[0],
                alpha=alpha if alpha is not None else 0.7,
                zorder=zorder_circle,
            )
            ax.add_patch(radar_1)

        # konzentrische Kreise
        for rad in radius:
            circle_1 = plt.Circle(
                xy=(0, 0),
                radius=rad,
                fc="none",
                ec=self.patch_color,
                lw=lw_circle,
                alpha=0.7,
                zorder=zorder_circle - 1,
            )
            ax.add_patch(circle_1)

            if not compare:
                circle_2 = plt.Circle(
                    xy=(0, 0),
                    radius=rad,
                    fc="none",
                    ec=radar_color[1],
                    lw=lw_circle,
                    alpha=0.7,
                    zorder=zorder_circle,
                )
                circle_2.set_clip_path(radar_1)
                ax.add_patch(circle_2)

        return ax

    # ---------- Labels & Ranges ----------

    def __add_labels(self, params, ax, return_list=False, radius=20, range_val=False):
        coord = _get_coordinates(n=len(params))

        if return_list:
            x_y = []

        for i in range(len(params)):
            rot = coord[i, 2]
            x, y = (radius * np.sin(rot), radius * np.cos(rot))

            if return_list:
                x_y.append((x, y))

            if y < 0:
                rot += np.pi

            if isinstance(params[i], np.floating):
                p = round(params[i], 2)
            else:
                p = params[i]

            if range_val:
                size = self.range_fontsize
                color = self.range_color
                weight = self.range_weight
            else:
                size = self.label_fontsize
                color = self.label_color
                weight = self.label_weight

            ax.text(
                x,
                y,
                p,
                rotation=-np.rad2deg(rot),
                ha="center",
                va="center",
                fontsize=size,
                fontfamily=self.fontfamily,
                fontweight=weight,
                color=color,
                zorder=10,  # immer vor Polygon & Kreisen
            )

        if return_list:
            return ax, x_y
        else:
            return ax

    def __add_ranges(self, ranges, ax):
        radius = [2.5, 4.1, 5.8, 7.5, 9.2, 10.9, 12.6, 14.3, 15.9, 17.6]
        x_y = []
        range_values = np.array([])

        for rng in ranges:
            value = np.linspace(start=rng[0], stop=rng[1], num=10)
            range_values = np.append(range_values, value)

        range_values = range_values.reshape((len(ranges), 10))

        for i in range(len(radius)):
            params = range_values[:, i]
            ax, xy = self.__add_labels(
                params=params,
                ax=ax,
                return_list=True,
                radius=radius[i],
                range_val=True,
            )
            x_y.append(xy)

        return ax, np.array(x_y), range_values

    # ---------- Vertices ----------

    def __get_vertices(self, values, xy, range_values):
        vertices = []

        for i in range(len(range_values)):
            range_list = range_values[i, :]
            coord_list = xy[:, i]

            if range_list[0] > range_list[-1]:
                if values[i] >= range_list[0]:
                    x_coord, y_coord = coord_list[0, 0], coord_list[0, 1]
                elif values[i] <= range_list[-1]:
                    x_coord, y_coord = coord_list[-1, 0], coord_list[-1, 1]
                else:
                    x_coord, y_coord = _get_indices_between(
                        range_list=range_list,
                        coord_list=coord_list,
                        value=values[i],
                        reverse=True,
                    )
            else:
                if values[i] >= range_list[-1]:
                    x_coord, y_coord = coord_list[-1, 0], coord_list[-1, 1]
                elif values[i] <= range_list[0]:
                    x_coord, y_coord = coord_list[0, 0], coord_list[0, 1]
                else:
                    x_coord, y_coord = _get_indices_between(
                        range_list=range_list,
                        coord_list=coord_list,
                        value=values[i],
                        reverse=False,
                    )

            vertices.append([x_coord, y_coord])

        return vertices

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"background_color={self.background_color}, "
            f"patch_color={self.patch_color}, "
            f"fontfamily={self.fontfamily}, "
            f"label_fontsize={self.label_fontsize}, "
            f"label_color={self.label_color}, "
            f"label_weight={self.label_weight}, "
            f"range_fontsize={self.range_fontsize}, "
            f"range_color={self.range_color}, "
            f"range_weight={self.range_weight})"
        )

    __str__ = __repr__

# -------------------------------------------------------------------
# Data loading (cached)
# -------------------------------------------------------------------
@st.cache_data
def load_data(version: str):
    from src.multi_season import load_all_seasons, aggregate_player_scores
    from src.squad import compute_squad_scores

    root = Path(__file__).resolve().parent
    processed_dir = root / "Data" / "Processed"
    raw_dir = root / "Data" / "Raw"

    df_all = load_all_seasons(processed_dir)
    df_agg = aggregate_player_scores(df_all)
    df_squad = compute_squad_scores(df_all)

    df_big5 = pd.DataFrame()
    big5_path = processed_dir / "big5_table_all_seasons.csv"
    if big5_path.exists():
        df_big5 = pd.read_csv(big5_path)

    # --- Transfermarkt market values ---
    mv_path = processed_dir / "player_market_values.csv"
    if mv_path.exists():
        df_mv = pd.read_csv(mv_path)[["Player", "Squad", "MarketValue_EUR", "tm_player_id"]]
        df_all = df_all.merge(df_mv, on=["Player", "Squad"], how="left")

        # Squad-level: sum of market values per Season + Squad
        if "MarketValue_EUR" in df_all.columns:
            squad_mv = (
                df_all.groupby(["Season", "Squad"])["MarketValue_EUR"]
                .sum()
                .reset_index()
                .rename(columns={"MarketValue_EUR": "TotalMarketValue_squad"})
            )
            df_squad = df_squad.merge(squad_mv, on=["Season", "Squad"], how="left")

    # Historical market value timeline (for player profile chart)
    df_valuations = pd.DataFrame()
    val_path = raw_dir / "tm_player_valuations.csv"
    if val_path.exists():
        df_valuations = pd.read_csv(val_path)

    return df_all, df_agg, df_squad, df_big5, df_valuations



# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def get_role_label(pos: str | None) -> str:
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
    if series is None or series.empty or series.notna().sum() == 0:
        return "n/a"
    return f"{series.mean():.1f}"


def assess_diff(diff: float) -> str:
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


def get_primary_score_and_band(row: pd.Series) -> tuple[float | None, str | None]:
    """
    Returns primary score + band for a player row.

    Prefers the pre-computed MainScore (L1+L3 normalized, cross-position comparable)
    from the processed CSV. Falls back to the old role-based raw score for legacy data
    that was processed before normalization was added.
    """
    # Use pre-computed normalized MainScore if present and valid
    ms_raw = row.get("MainScore")
    if ms_raw is not None and not pd.isna(ms_raw):
        try:
            ms = float(ms_raw)
            mb = row.get("MainBand") or band_from_score(ms)
            return ms, mb
        except (TypeError, ValueError):
            pass

    # Fallback: old role-based raw score (legacy CSVs without MainScore column)
    pos_raw = row.get("Pos_raw") or row.get("Pos")
    if pos_raw in ("FW", "Off_MF"):
        return row.get("OffScore_abs"), row.get("OffBand")
    if pos_raw in ("MF",):
        return row.get("MidScore_abs"), row.get("MidBand")
    if pos_raw in ("DF", "Def_MF"):
        return row.get("DefScore_abs"), row.get("DefBand")

    return (None, None)


def band_from_score(score: float | None) -> str | None:
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

    return BAND_ICONS.get(band, band)


def score_trend_chart(df_player_all: pd.DataFrame, score_col: str, label: str):
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

# -------------------------------------------------------------------
# Pizza charts (per season)
# -------------------------------------------------------------------
def render_pizza_chart(
    df_all: pd.DataFrame,
    df_player: pd.DataFrame,
    role: str,
    season: str | None,
):
    if df_player.empty or role is None:
        st.info("No data available for pizza chart.")
        return

    row = df_player.iloc[0]

    comp = row["Comp"] if "Comp" in row.index else None
    if comp not in BIG5_COMPS:
        st.info("Pizza chart is currently only available for Big-5 competitions.")
        return

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

    def resolve_metric_column(df: pd.DataFrame, row_s: pd.Series, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns and c in row_s.index:
                return c
        return None

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

        percentile = (peers <= val).mean() * 100.0
        percentile = int(round(percentile))

        params.append(label)
        values.append(percentile)
        groups.append(group)

    if len(params) < 3:
        st.info("Not enough metrics available to build a pizza chart for this player (Big-5 only).")
        return

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

    player_name = row.get("Player", "")
    squad_name = row.get("Squad", "")

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

    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=COLOR_POSSESSION, label="Possession"),
        mpatches.Patch(color=COLOR_ATTACKING, label="Attacking"),
        mpatches.Patch(color=COLOR_DEFENDING, label="Defending"),
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

# -------------------------------------------------------------------
# Career Pizza chart
# -------------------------------------------------------------------
def render_career_pizza_chart(
    player: str,
    role: str,
    seasons: list[str],
) -> plt.Figure | None:
    if not seasons or role is None:
        st.info("Not enough data available for a career pizza chart.")
        return None

    root = Path(__file__).resolve().parent
    raw_dir = root / "Data" / "Raw"

    df_features_all_list = []
    df_player_feat_list = []

    for s in seasons:
        season_safe = s.replace("/", "-")
        csv_path = raw_dir / f"players_data-{season_safe}.csv"
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

    def resolve_metric_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

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

    mins = pd.to_numeric(df_player_all_feat.get("Min", pd.Series(dtype=float)), errors="coerce")
    total_min = mins.sum() if mins.notna().any() else None

    for group, candidates, label in metric_defs:
        col = resolve_metric_column(df_comp, candidates)
        if col is None or col not in df_player_all_feat.columns:
            continue

        vals_player = pd.to_numeric(df_player_all_feat[col], errors="coerce")
        if not vals_player.notna().any():
            continue

        if total_min is not None and total_min > 0 and mins.notna().any():
            num_min = (vals_player * mins).sum()
            denom_min = total_min
            player_val = num_min / denom_min
        else:
            player_val = vals_player.mean()

        if pd.isna(player_val):
            continue

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

    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=COLOR_POSSESSION, label="Possession"),
        mpatches.Patch(color=COLOR_ATTACKING, label="Attacking"),
        mpatches.Patch(color=COLOR_DEFENDING, label="Defending"),
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

# -------------------------------------------------------------------
# Role scatter
# -------------------------------------------------------------------
def render_role_scatter(
    df_comp: pd.DataFrame,
    df_player: pd.DataFrame,
    role: str,
):
    if df_player.empty or role is None:
        st.info("No data available for role scatter plot.")
        return None

    row = df_player.iloc[0]

    player_name = row.get("Player")
    player_squad = row.get("Squad")

    def resolve_col(candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df_comp.columns and c in row.index:
                return c
        return None

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

    cols_needed = ["Player", "Squad", x_col, y_col]
    if "Season" in df_comp.columns:
        cols_needed.append("Season")

    plot_df = df_comp[cols_needed].copy()
    plot_df = plot_df.dropna(subset=[x_col, y_col])

    if plot_df.empty:
        st.info("No comparison data available for role scatter plot.")
        return None

    if "Squad" in plot_df.columns and player_squad is not None:
        player_mask = (
            (plot_df["Player"] == player_name) &
            (plot_df["Squad"] == player_squad)
        )
    else:
        player_mask = (plot_df["Player"] == player_name)

    plot_df["is_player"] = player_mask

    s_x = pd.to_numeric(plot_df[x_col], errors="coerce")
    s_y = pd.to_numeric(plot_df[y_col], errors="coerce")

    qx = s_x.quantile(0.99)
    qy = s_y.quantile(0.99)

    if np.isfinite(qx) and np.isfinite(qy):
        plot_df = plot_df[((s_x <= qx) & (s_y <= qy)) | player_mask]

    if plot_df.empty:
        st.info("No comparison data available for role scatter plot.")
        return None

    if "Squad" in plot_df.columns and player_squad is not None:
        plot_df["is_player"] = (
            (plot_df["Player"] == player_name) &
            (plot_df["Squad"] == player_squad)
        )
    else:
        plot_df["is_player"] = (plot_df["Player"] == player_name)

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

        return (0.0, upper)

    if role in ("FW", "Off_MF"):
        x_domain = (0.0, 0.9)
        y_domain = (0.0, 1.3)
    else:
        x_domain = compute_domain(plot_df[x_col], default_max=1.0, q=0.99)
        y_domain = compute_domain(plot_df[y_col], default_max=1.0, q=0.99)

    base = alt.Chart(plot_df)

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

    player_layer = base.transform_filter(
        alt.datum.is_player == True
    ).mark_circle(
        size=140,
        opacity=1.0,
        stroke="#F9FAFB",
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
        dx=30,
        dy=-15,
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

# -------------------------------------------------------------------
# Top list charts
# -------------------------------------------------------------------
def render_toplist_bar(
    df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    base_title: str,
    top_n: int,
    season: str | None = None,
    ascending: bool = False,
):
    import plotly.graph_objects as go

    if metric_col not in df.columns:
        st.info(f"Keine Spalte '{metric_col}' gefunden.")
        return None

    base_cols = ["Player", "Squad", metric_col, "Pos", "Comp", "Age", "Min"]
    cols_available = [c for c in base_cols if c in df.columns]

    df_plot = (
        df[cols_available]
        .dropna(subset=[metric_col])
        .copy()
    )

    if df_plot.empty:
        st.info(f"Keine gültigen Werte für {metric_label}.")
        return None

    df_plot = df_plot.sort_values(metric_col, ascending=ascending).head(top_n)

    # Build title
    pos_label = comp_label = squad_label = None
    if "Pos" in df_plot.columns:
        pos_vals = sorted(df_plot["Pos"].dropna().unique())
        if len(pos_vals) == 1:
            pos_map = {"FW": "Forwards", "Off_MF": "Offensive midfielders",
                       "MF": "Midfielders", "DF": "Defenders", "Def_MF": "Defensive midfielders"}
            pos_label = pos_map.get(pos_vals[0], pos_vals[0])
    if "Comp" in df_plot.columns:
        comps = sorted(df_plot["Comp"].dropna().unique())
        if len(comps) == 1:
            comp_label = comps[0]
    if "Squad" in df_plot.columns:
        squads = sorted(df_plot["Squad"].dropna().unique())
        if len(squads) == 1:
            squad_label = squads[0]
    parts: list[str] = []
    if pos_label:
        parts.append(pos_label)
    if squad_label:
        parts.append(squad_label)
    elif comp_label:
        parts.append(comp_label)
    context = " – ".join(parts) if parts else "All Big-5 leagues"
    full_title = f"{base_title} – {context}"
    if season:
        full_title += f" (Season {season})"

    players = df_plot["Player"].tolist()
    scores = df_plot[metric_col].round(0).astype(int).tolist()
    max_val = float(df_plot[metric_col].max()) if scores else 1000.0
    x_limit = max_val * 1.55  # extra room for player name labels

    # Hover text
    def _bar_hover(row: pd.Series) -> str:
        parts_h = [f"<b>{row['Player']}</b>"]
        if "Squad" in row.index:
            parts_h.append(f"Club: {row['Squad']}")
        if "Comp" in row.index:
            parts_h.append(f"League: {row['Comp']}")
        if "Age" in row.index:
            parts_h.append(f"Age: {row['Age']:.0f}")
        parts_h.append(f"Score: {row[metric_col]:.0f}")
        return "<br>".join(parts_h)

    hover_texts = [_bar_hover(r) for _, r in df_plot.iterrows()]

    fig = go.Figure()

    # Bars with score text inside
    fig.add_trace(go.Bar(
        y=players,
        x=scores,
        orientation="h",
        marker=dict(color=VALUE_COLOR, line=dict(width=0)),
        text=[str(s) for s in scores],
        textposition="inside",
        textfont=dict(color="#0f172a", size=15, family="Arial Black, Arial, sans-serif"),
        customdata=[[p] for p in players],
        hovertext=hover_texts,
        hoverinfo="text",
        showlegend=False,
        name="",
    ))

    # Player name labels at bar end
    fig.add_trace(go.Scatter(
        y=players,
        x=[max_val * 1.07] * len(players),
        mode="text",
        text=players,
        textfont=dict(color="#E5E7EB", size=12),
        textposition="middle right",
        customdata=[[p] for p in players],
        showlegend=False,
        hoverinfo="skip",
        name="",
    ))

    fig.update_layout(
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
        font=dict(color="#E5E7EB"),
        height=max(300, 28 * len(df_plot) + 80),
        yaxis=dict(
            autorange="reversed",
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        xaxis=dict(
            range=[0, x_limit],
            showgrid=False,
            showticklabels=False,
            showline=False,
            zeroline=False,
        ),
        title=dict(
            text=full_title,
            font=dict(color="#E5E7EB", size=20),
            x=0,
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        clickmode="event+select",
        dragmode="pan",
        bargap=0.25,
    )

    return fig

# -------------------------------------------------------------------
# Score vs Age Beeswarm
# -------------------------------------------------------------------
def render_score_age_beeswarm(
    df_all_filtered: pd.DataFrame,
    df_top: pd.DataFrame,
):
    if "MainScore" not in df_all_filtered.columns:
        st.info("Beeswarm: Spalte 'MainScore' nicht gefunden.")
        return None
    if "Age" not in df_all_filtered.columns:
        st.info("Beeswarm: Spalte 'Age' nicht gefunden.")
        return None

    cols = ["Player", "Squad", "Pos", "MainScore", "Age"]
    if "90s" in df_all_filtered.columns:
        cols.append("90s")

    df_plot = df_all_filtered[cols].copy()

    df_plot["Age_num"] = pd.to_numeric(df_plot["Age"], errors="coerce")
    df_plot = df_plot.dropna(subset=["Age_num", "MainScore"])

    if df_plot.empty:
        st.info("Beeswarm: Keine gültigen Daten (Age & MainScore) nach Filter.")
        return None

    if "90s" in df_plot.columns:
        df_plot["MinutesFactor"] = pd.to_numeric(df_plot["90s"], errors="coerce").fillna(0)
    else:
        df_plot["MinutesFactor"] = 0.0

    df_top = df_top.copy()
    df_plot = df_plot.copy()

    df_top["ps_key"] = df_top["Player"].astype(str) + " | " + df_top["Squad"].astype(str)
    df_plot["ps_key"] = df_plot["Player"].astype(str) + " | " + df_plot["Squad"].astype(str)

    top_keys = set(df_top["ps_key"].unique())
    df_plot["is_top"] = df_plot["ps_key"].isin(top_keys)

    jitter = np.random.uniform(-0.4, 0.4, size=len(df_plot))
    df_plot["Age_jitter"] = df_plot["Age_num"] + jitter

    import plotly.graph_objects as go

    peers = df_plot[~df_plot["is_top"]]
    tops  = df_plot[df_plot["is_top"]]

    def _hover(df_sub: pd.DataFrame) -> list[str]:
        rows = []
        for _, r in df_sub.iterrows():
            mins = f"{r['MinutesFactor']:.1f}" if r["MinutesFactor"] > 0 else "—"
            rows.append(
                f"<b>{r['Player']}</b><br>"
                f"Club: {r['Squad']}<br>"
                f"Pos: {r['Pos']}<br>"
                f"Age: {r['Age_num']:.0f}<br>"
                f"Score: {r['MainScore']:.0f}<br>"
                f"90s: {mins}"
            )
        return rows

    fig = go.Figure()

    # Hollow circles — all non-top players
    fig.add_trace(go.Scatter(
        x=peers["MainScore"],
        y=peers["Age_jitter"],
        mode="markers",
        marker=dict(
            size=6,
            color="rgba(0,0,0,0)",
            line=dict(color="#F9FAFB", width=1),
        ),
        hoverinfo="text",
        hovertext=_hover(peers),
        customdata=peers[["Player", "Squad"]].values,
        name="All players",
        showlegend=True,
    ))

    # Filled teal circles + player name labels — top N
    fig.add_trace(go.Scatter(
        x=tops["MainScore"],
        y=tops["Age_jitter"],
        mode="markers+text",
        marker=dict(
            size=12,
            color=VALUE_COLOR,
            line=dict(color="#F9FAFB", width=1.5),
        ),
        text=tops["Player"],
        textposition="middle right",
        textfont=dict(color="#E5E7EB", size=10),
        hoverinfo="text",
        hovertext=_hover(tops),
        customdata=tops[["Player", "Squad"]].values,
        name="Top N",
        showlegend=True,
    ))

    fig.update_layout(
        title=dict(
            text="Score vs Age — click a dot to open the player profile",
            font=dict(color="#E5E7EB", size=16),
            x=0,
        ),
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
        font=dict(color="#E5E7EB"),
        xaxis=dict(
            title="Score",
            range=[0, 1000],
            gridcolor="#374151",
            gridwidth=0.5,
            color="#E5E7EB",
        ),
        yaxis=dict(
            title="Age",
            gridcolor="#374151",
            gridwidth=0.5,
            color="#E5E7EB",
        ),
        legend=dict(
            font=dict(color="#E5E7EB"),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=470,
        margin=dict(l=50, r=20, t=50, b=40),
        clickmode="event+select",
        dragmode="pan",
    )

    return fig

# -------------------------------------------------------------------
# Band histogram
# -------------------------------------------------------------------
def render_band_histogram(df: pd.DataFrame, season: str | None = None):
    if "MainBand" not in df.columns:
        st.info("Band-Histogramm: Spalte 'MainBand' nicht gefunden.")
        return None

    if "BAND_ORDER" in globals():
        all_bands = BAND_ORDER
    else:
        all_bands = sorted(df["MainBand"].dropna().unique().tolist())

    if not all_bands:
        st.info("Band-Histogramm: Keine Bands verfügbar.")
        return None

    df_hist = df[["MainBand"]].dropna().copy()
    df_counts = (
        df_hist
        .groupby("MainBand")
        .size()
        .reset_index(name="Count")
    )

    base_bands = pd.DataFrame({"MainBand": all_bands})
    df_counts = base_bands.merge(df_counts, on="MainBand", how="left")
    df_counts["Count"] = df_counts["Count"].fillna(0).astype(int)

    comp_name = None
    squad_name = None

    if "Comp" in df.columns:
        comps = sorted(df["Comp"].dropna().unique())
        if len(comps) == 1:
            comp_name = comps[0]

    if "Squad" in df.columns:
        squads = sorted(df["Squad"].dropna().unique())
        if len(squads) == 1:
            squad_name = squads[0]

    if squad_name is not None:
        base_title = f"Band distribution – {squad_name}"
    elif comp_name is not None:
        base_title = f"Band distribution – {comp_name}"
    else:
        base_title = "Band distribution – All Big-5 leagues"

    if season is not None:
        title = f"{base_title} (Season {season})"
    else:
        title = base_title

    if squad_name is not None:
        y_max = 20
        step = 5
    elif comp_name is not None:
        y_max = 250
        step = 25
    else:
        y_max = 1200
        step = 100

    y_scale = alt.Scale(domain=(0, y_max), nice=False)
    y_axis = alt.Axis(
        title=None,
        values=list(range(0, y_max + 1, step)),
        labelColor="#E5E7EB",
    )

    bars = (
        alt.Chart(df_counts)
        .mark_bar(
            cornerRadiusTopLeft=6,
            cornerRadiusTopRight=6,
        )
        .encode(
            x=alt.X(
                "MainBand:N",
                title=None,
                sort=all_bands,
                axis=alt.Axis(
                    labelColor="#E5E7EB",
                ),
            ),
            y=alt.Y(
                "Count:Q",
                scale=y_scale,
                axis=y_axis,
            ),
            tooltip=[
                alt.Tooltip("MainBand:N", title="Band"),
                alt.Tooltip("Count:Q", title="Players", format=".0f"),
            ],
            color=alt.value(VALUE_COLOR),
        )
    )

    labels = (
        alt.Chart(df_counts)
        .mark_text(
            align="center",
            baseline="bottom",
            dy=-8,
            fontSize=11,
            color="#E5E7EB",
            fontWeight="bold",
        )
        .encode(
            x=alt.X("MainBand:N", sort=all_bands),
            y=alt.Y("Count:Q", scale=y_scale),
            text=alt.Text("Count:Q", format=".0f"),
        )
    )

    chart = (
        (bars + labels)
        .properties(
            width=550,
            height=600,
            title=title,
            padding={
                "top": 40,
                "left": 40,
                "right": 20,
                "bottom": 40,
            },
        )
        .configure_axis(
            grid=True,
            gridOpacity=0.1,
            gridColor="#4b5563",
        )
        .configure_title(
            color="#E5E7EB",
            fontSize=20,
            anchor="start",
        )
        .configure_view(
            strokeWidth=0,
        )
    )

    return chart

# -------------------------------------------------------------------
# FIFA-style rating card
# -------------------------------------------------------------------
def fmt_market_value(v: float) -> str:
    """Format a market value in EUR to a human-readable string."""
    if v >= 1_000_000:
        return f"€{v / 1_000_000:.0f}M"
    return f"€{v / 1_000:.0f}K"


def render_fifa_card(
    row: pd.Series,
    primary_score_col: str,
    band_col: str,
    *,
    title: str | None = None,
):
    player_name = str(row.get("Player", "Unknown"))
    pos_display = str(row.get("Pos", "-"))
    club = str(row.get("Squad", ""))
    age = row.get("Age", None)

    crest_b64 = get_crest_b64(club)
    crest_html = (
        f'<img src="{crest_b64}" style="width:18px;height:18px;object-fit:contain;vertical-align:middle;margin-right:4px;">'
        if crest_b64
        else ""
    )

    try:
        overall_raw = float(row.get(primary_score_col, 0.0))
        overall = int(round(overall_raw))
    except Exception:
        overall = 0

    band_label = str(row.get(band_col, "") or "")
    comp = str(row.get("Comp", ""))

    mv_raw = row.get("MarketValue_EUR", None)
    mv_badge_html = ""
    try:
        if mv_raw is not None and not pd.isna(mv_raw):
            mv_badge_html = (
                f'<div class="fifa-card-mv-badge">{fmt_market_value(float(mv_raw))}</div>'
            )
    except Exception:
        pass

    def fmt_int(val) -> str:
        try:
            if val is None or pd.isna(val):
                return "-"
            return f"{int(round(float(val)))}"
        except Exception:
            return "-"

    def pick_value(col_names, decimals: int = 2, is_percent: bool = False) -> str:
        for c in col_names:
            if c in row.index:
                val = row.get(c)
                if val is None or (isinstance(val, (int, float)) and pd.isna(val)):
                    continue
                try:
                    v = float(val)
                    if is_percent:
                        return f"{v:.0f}%"
                    return f"{v:.{decimals}f}"
                except Exception:
                    continue
        return "-"

    minutes_raw = row.get("Min", row.get("Minutes", None))
    if (minutes_raw is None or pd.isna(minutes_raw)) and "90s" in row.index:
        try:
            minutes_raw = float(row.get("90s")) * 90.0
        except Exception:
            minutes_raw = None
    minutes_str = fmt_int(minutes_raw)

    pass_acc_str = pick_value(
        ["Cmp%", "Cmp_Pct", "PassCmp_Pct"],
        decimals=0,
        is_percent=True,
    )

    base_attrs = [
        ("Min", minutes_str),
        ("Pass Acc", pass_acc_str),
    ]

    pos_upper = (pos_display or "").upper()
    if "FW" in pos_upper:
        role_type = "FW"
    elif "DF" in pos_upper or "CB" in pos_upper or "FB" in pos_upper:
        role_type = "DF"
    elif "MF" in pos_upper:
        role_type = "MF"
    else:
        role_type = "MF"

    if role_type == "FW":
        g_90 = pick_value(["Gls_Per90", "G_90", "Gls90"])
        xg_90 = pick_value(["xG_Per90", "xG_90", "xG90"])
        npxg_90 = pick_value(["npxG_Per90", "npxG_90", "npxG90"])
        sh_90 = pick_value(["Sh_Per90", "SoT_Per90", "Sh_90"])

        role_attrs = [
            ("G/90", g_90),
            ("xG/90", xg_90),
            ("npxG/90", npxg_90),
            ("Shots/90", sh_90),
        ]

    elif role_type == "DF":
        blk_90 = pick_value(
            ["Blocks_Per90", "Blocks_stats_defense_Per90", "Blocks_stats_defense_90", "Blocks_90"]
        )
        int_90 = pick_value(["Int_Per90", "Tkl+Int_Per90", "Int_90"])
        clr_90 = pick_value(["Clr_Per90", "Clr_90"])
        aer_pct = pick_value(
            ["Won%", "AerialWon_Pct", "AerialDuelsWon_Pct", "AerialDuels_Won%"],
            decimals=0,
            is_percent=True,
        )

        role_attrs = [
            ("Blocks/90", blk_90),
            ("Int/90", int_90),
            ("Clr/90", clr_90),
            ("Aerials Won", aer_pct),
        ]

    else:
        ast_90 = pick_value(["Ast_Per90", "Ast_90", "Ast90"])
        xag_90 = pick_value(["xAG_Per90", "xA_Per90", "xA_90"])
        kp_90 = pick_value(["KP_Per90", "KP_90"])
        sca_90 = pick_value(["SCA_Per90", "SCA90", "SCA_90"])

        role_attrs = [
            ("Ast/90", xag_90),
            ("xAG/90", ast_90),
            ("KP/90", kp_90),
            ("SCA90", sca_90),
        ]

    all_attrs = base_attrs + role_attrs

    attr_rows_html = "".join(
        f'<div><span class="fifa-card-attr-value">{val}</span>'
        f'<span class="fifa-card-attr-label">{label}</span></div>'
        for (label, val) in all_attrs
    )

    if title is not None:
        st.markdown(f"### {title}")

    card_html = f"""
    <style>
    .fifa-card-container {{
        display: flex;
        justify-content: center;
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
    }}
    .fifa-card {{
        position: relative;
        width: 260px;
        height: 360px;
        border-radius: 22px;
        padding: 16px 18px;
        box-sizing: border-box;
        background: radial-gradient(circle at 0% 0%, #2EF2E0 0%, #00897B 40%, #0B1F1E 100%);
        box-shadow: 0 12px 30px rgba(0,0,0,0.45);
        color: #fff;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .fifa-card-top-row {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
    }}
    .fifa-card-overall {{
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }}
    .fifa-card-overall-value {{
        font-size: 46px;
        font-weight: 800;
        line-height: 1;
    }}
    .fifa-card-pos {{
        font-size: 16px;
        font-weight: 600;
        letter-spacing: 1px;
        margin-top: 4px;
    }}
    .fifa-card-band-pill {{
        margin-top: 10px;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        background: rgba(0,0,0,0.25);
        border: 1px solid rgba(255,255,255,0.35);
    }}
    .fifa-card-meta {{
        text-align: right;
        font-size: 11px;
        opacity: 0.9;
    }}
    .fifa-card-meta div {{
        margin-bottom: 2px;
    }}
    .fifa-card-player-name {{
        margin-top: 28px;
        font-size: 20px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        text-shadow: 0 2px 4px rgba(0,0,0,0.45);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .fifa-card-club {{
        margin-top: 4px;
        font-size: 12px;
        opacity: 0.95;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .fifa-card-divider {{
        width: 60%;
        height: 1px;
        margin-top: 18px;
        margin-bottom: 16px;
        background: linear-gradient(to right, rgba(255,255,255,0.5), rgba(255,255,255,0));
    }}
    .fifa-card-attributes-title {{
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.85;
        margin-bottom: 6px;
    }}
    .fifa-card-attributes {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        row-gap: 6px;
        column-gap: 8px;
        font-size: 12px;
    }}
    .fifa-card-attr-label {{
        opacity: 0.85;
        font-weight: 500;
    }}
    .fifa-card-attr-value {{
        font-weight: 700;
        font-size: 13px;
        margin-right: 4px;
    }}
    .fifa-card-footer {{
        position: absolute;
        bottom: 10px;
        left: 18px;
        right: 18px;
        display: flex;
        justify-content: space-between;
        font-size: 10px;
        opacity: 0.8;
    }}
    .fifa-card-mv-badge {{
        margin-top: 6px;
        font-size: 13px;
        font-weight: 700;
        color: #fde68a;
        letter-spacing: 0.5px;
    }}
    </style>

    <div class="fifa-card-container">
      <div class="fifa-card">
        <div class="fifa-card-top-row">
          <div class="fifa-card-overall">
            <div class="fifa-card-overall-value">{overall}</div>
            <div class="fifa-card-pos">{pos_display}</div>
            <div class="fifa-card-band-pill">{band_label}</div>
            {mv_badge_html}
          </div>
          <div class="fifa-card-meta">
            {"<div>Age: " + str(int(age)) + "</div>" if age is not None else ""}
            {"<div>" + comp + "</div>" if comp else ""}
          </div>
        </div>

        <div class="fifa-card-player-name">{player_name}</div>
        <div class="fifa-card-club">{crest_html}{club}</div>

        <div class="fifa-card-divider"></div>

        <div class="fifa-card-attributes-title">Key attributes</div>
        <div class="fifa-card-attributes">
          {attr_rows_html}
        </div>

        <div class="fifa-card-footer">
          <div>PlayerScore</div>
          <div>FBref Big-5</div>
        </div>
      </div>
    </div>
    """

    st_html(card_html, height=380)

# -------------------------------------------------------------------
# Feature-Table Loader & Enrichment
# -------------------------------------------------------------------
@st.cache_data
def load_feature_table_for_season(season: str) -> pd.DataFrame:
    root = Path(__file__).resolve().parent
    raw_dir = root / "Data" / "Raw"

    season_safe = season.replace("/", "-")
    csv_path = raw_dir / f"players_data-{season_safe}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"No feature file found for season: {season} ({csv_path})")

    df = pd.read_csv(csv_path)
    df = prepare_positions(df)

    if "Blocks" not in df.columns and "Blocks_stats_defense" in df.columns:
        df["Blocks"] = df["Blocks_stats_defense"]

    df = add_standard_per90(df)

    return df

@st.cache_data
def load_squad_raw_for_season(season: str) -> pd.DataFrame:
    """
    Lädt die vollständige squads_data-<Season>.csv aus Data/Raw.

    Erwartet Dateien wie:
      Data/Raw/squads_data-2017-2018.csv
      Data/Raw/squads_data-2025-2026.csv
    """
    root = Path(__file__).resolve().parent
    raw_dir = root / "Data" / "Raw"

    season_safe = season.replace("/", "-")
    csv_path = raw_dir / f"squads_data-{season_safe}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"No squad raw file found for season: {season} ({csv_path})"
        )

    df = pd.read_csv(csv_path)

    # NICHT als Index setzen – wir filtern später über die Spalte "Squad"
    # ein paar wichtige Spalten sicher numerisch casten
    num_cols = [
        "npxG",
        "Sh",
        "SoT",
        "xA",
        "SCA90",
        "GCA90",
        "Poss",
        "PrgP_stats_teams_passing_for",
        "PrgC_stats_teams_possession_for",
        "Att 3rd_stats_teams_possession_for",
        "GA90",
        "Sh_stats_teams_defense_for",
        "PSxG",
        "/90",
        "Tkl+Int",
        "Blocks_stats_teams_defense_for",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# =========================
# TEAM RADAR – HELPERS
# =========================

def _col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Gibt die erste existierende Spalte aus 'candidates' zurück.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def prepare_squad_advanced_metrics(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Erweitert das Squad-DF (pro Season) um abgeleitete Metriken ähnlich deinem Beispiel:
    - Shots on target%
    - Passes to PA (per 90)
    - Carries to PA (per 90)
    - Touches in PA (per 90)
    - usw.

    Arbeitet vorsichtig mit Candidate-Listen, weil Spaltennamen je nach FBref-Tabelle leicht variieren können.
    """

    df = df_raw.copy()

    # --- Minuten & Spiele (für per90) ---
    col_90s = _col(df, ["90s"])
    col_games = _col(df, ["MP", "Games", "matches"])

    # Helper per90
    def per90(col_total_candidates: list[str], new_name: str):
        col_total = _col(df, col_total_candidates)
        if col_total is None:
            return
        if col_90s and col_90s in df.columns:
            n90 = pd.to_numeric(df[col_90s], errors="coerce").replace(0, np.nan)
            df[new_name] = pd.to_numeric(df[col_total], errors="coerce") / n90
        elif col_games and col_games in df.columns:
            g = pd.to_numeric(df[col_games], errors="coerce").replace(0, np.nan)
            df[new_name] = pd.to_numeric(df[col_total], errors="coerce") / g
        # wenn gar nichts da ist, lassen wir die Metrik einfach weg

    # --- Shots on target% ---
    col_sh = _col(df, ["Sh", "Shots"])
    col_sot = _col(df, ["SoT", "Shots on target", "SoTA"])  # SoTA = gegen – zur Sicherheit
    if col_sh and col_sot:
        shots = pd.to_numeric(df[col_sh], errors="coerce")
        sot = pd.to_numeric(df[col_sot], errors="coerce")
        df["Shots_on_target_pct"] = np.where(
            shots > 0,
            (sot / shots) * 100.0,
            np.nan,
        )

    # --- Passes to final 3rd% ---
    col_p_final3 = _col(df, ["1/3", "Passes into final third", "Final3rdPass"])
    col_p_cmp = _col(df, ["Cmp", "Passes completed"])
    if col_p_final3 and col_p_cmp:
        p3 = pd.to_numeric(df[col_p_final3], errors="coerce")
        pc = pd.to_numeric(df[col_p_cmp], errors="coerce")
        df["Passes_final3_pct"] = np.where(
            pc > 0,
            (p3 / pc) * 100.0,
            np.nan,
        )

    # --- Passes to PA (per 90) ---
    per90(["PPA", "Passes into penalty area"], "Passes_to_PA_per90")

    # --- Box crosses% (Crosses into PA / PPA) ---
    col_crs_pa = _col(df, ["CrsPA", "Crosses into penalty area"])
    col_ppa = _col(df, ["PPA", "Passes into penalty area"])
    if col_crs_pa and col_ppa:
        crs = pd.to_numeric(df[col_crs_pa], errors="coerce")
        ppa = pd.to_numeric(df[col_ppa], errors="coerce")
        df["Box_crosses_pct"] = np.where(
            ppa > 0,
            (crs / ppa) * 100.0,
            np.nan,
        )

    # --- Carries to final 3rd% ---
    col_c_final3 = _col(df, ["Carries into final third", "Carries Att 3rd"])
    col_c_total = _col(df, ["Carries"])
    if col_c_final3 and col_c_total:
        c3 = pd.to_numeric(df[col_c_final3], errors="coerce")
        ct = pd.to_numeric(df[col_c_total], errors="coerce")
        df["Carries_final3_pct"] = np.where(
            ct > 0,
            (c3 / ct) * 100.0,
            np.nan,
        )

    # --- Carries to PA (per 90) ---
    per90(
        ["Carries into penalty area", "Carries Att Pen"],
        "Carries_to_PA_per90",
    )

    # --- Touches final 3rd% ---
    col_t_att3 = _col(df, ["Att 3rd", "Touches Att 3rd"])
    col_t_live = _col(df, ["Touches", "Touches_live_ball"])
    if col_t_att3 and col_t_live:
        t3 = pd.to_numeric(df[col_t_att3], errors="coerce")
        tl = pd.to_numeric(df[col_t_live], errors="coerce")
        df["Touches_final3_pct"] = np.where(
            tl > 0,
            (t3 / tl) * 100.0,
            np.nan,
        )

    # --- npxG/Shot on target ---
    col_npxg_per90 = _col(df, ["npxG/90", "npxG_per90"])
    if col_npxg_per90 and col_sot:
        npxg90 = pd.to_numeric(df[col_npxg_per90], errors="coerce")
        sot90 = pd.to_numeric(df[col_sot], errors="coerce")
        df["npxG_per_SoT"] = np.where(
            sot90 > 0,
            npxg90 / sot90,
            np.nan,
        )

    # --- xA per 90 (Squad) ---
    col_xa = _col(df, ["xAG", "xA"])
    if col_xa:
        per90([col_xa], "xA_team_per90")

    # --- touches in PA per 90 ---
    col_t_pen = _col(df, ["Att Pen", "Touches Att Pen"])
    if col_t_pen:
        per90([col_t_pen], "Touches_in_PA_per90")

    # --- npxG per 90, Shots per 90, SoT per 90 etc. falls nötig ---
    per90(["npxG"], "npxG_team_per90")
    per90(["Sh"], "Shots_team_per90")
    per90(["SoT"], "SoT_team_per90")

    # --- xG conceded etc. könnten analog ergänzt werden, wenn Spalten vorhanden sind ---

    return df

def enrich_card_row_with_per90(card_row: pd.Series) -> pd.Series:
    season = card_row.get("Season")
    player_name = card_row.get("Player")
    squad_name = card_row.get("Squad")

    if season is None or player_name is None:
        return card_row

    try:
        df_feat = load_feature_table_for_season(str(season))
    except Exception:
        return card_row

    if df_feat is None or df_feat.empty:
        return card_row

    mask = df_feat["Player"] == player_name
    if "Squad" in df_feat.columns and isinstance(squad_name, str):
        mask = mask & (df_feat["Squad"] == squad_name)

    df_feat_player = df_feat.loc[mask]
    if df_feat_player.empty:
        df_feat_player = df_feat[df_feat["Player"] == player_name]
    if df_feat_player.empty:
        return card_row

    feat_row = df_feat_player.iloc[0]

    base_cols = [
        c for c in feat_row.index
        if c.endswith("_Per90") or c.endswith("90") or c in ("MP", "Min", "90s")
    ]

    extra_cols = [
        c for c in ["Cmp%", "Cmp_Pct", "PassCmp_Pct", "Won%"]
        if c in feat_row.index
    ]

    cols_to_copy = list(dict.fromkeys(base_cols + extra_cols))

    for c in cols_to_copy:
        card_row[c] = feat_row[c]

    return card_row

def build_career_card_row(df_player_all: pd.DataFrame, player: str) -> pd.Series | None:
    if df_player_all.empty or "Season" not in df_player_all.columns:
        return None

    seasons = sorted(df_player_all["Season"].dropna().unique())
    if not seasons:
        return None

    root = Path(__file__).resolve().parent
    raw_dir = root / "Data" / "Raw"

    feat_rows = []
    for s in seasons:
        season_safe = str(s).replace("/", "-")
        csv_path = raw_dir / f"players_data-{season_safe}.csv"
        if not csv_path.exists():
            continue

        df_season = pd.read_csv(csv_path)
        df_season = prepare_positions(df_season)

        if "Blocks" not in df_season.columns and "Blocks_stats_defense" in df_season.columns:
            df_season["Blocks"] = df_season["Blocks_stats_defense"]

        df_season = add_standard_per90(df_season)
        df_season["Season"] = s

        df_p = df_season[df_season["Player"] == player].copy()
        if df_p.empty:
            continue

        feat_rows.append(df_p)

    if not feat_rows:
        return None

    df_feat = pd.concat(feat_rows, ignore_index=True)

    if "90s" in df_feat.columns:
        weight = pd.to_numeric(df_feat["90s"], errors="coerce").fillna(0.0)
        total_90s = float(weight.sum())
    else:
        weight = None
        total_90s = 0.0

    career_row = df_player_all.sort_values("Season").iloc[-1].copy()

    if "Min" in df_feat.columns:
        career_row["Min"] = float(pd.to_numeric(df_feat["Min"], errors="coerce").fillna(0.0).sum())
    elif total_90s > 0:
        career_row["Min"] = total_90s * 90.0

    for col in ["Cmp%", "Cmp_Pct", "PassCmp_Pct", "Won%"]:
        if col in df_feat.columns:
            career_row[col] = float(pd.to_numeric(df_feat[col], errors="coerce").dropna().mean())

    if total_90s > 0 and weight is not None:
        for col in df_feat.columns:
            if col.endswith("_Per90") or col.endswith("90"):
                vals = pd.to_numeric(df_feat[col], errors="coerce").fillna(0.0)
                num = float((vals * weight).sum())
                career_row[col] = num / total_90s if total_90s > 0 else np.nan

    career_row["Season"] = "Career"

    return career_row

def compute_career_main_score(df_player_all: pd.DataFrame) -> tuple[float | None, str | None]:
    if df_player_all.empty:
        return None, None

    df = df_player_all.copy()

    df[["MainScore", "MainBand_row"]] = df.apply(
        get_primary_score_and_band,
        axis=1,
        result_type="expand",
    )

    df = df.dropna(subset=["MainScore"])
    if df.empty:
        return None, None

    if "90s" in df.columns:
        w = pd.to_numeric(df["90s"], errors="coerce").fillna(0.0)
        total_w = float(w.sum())
        if total_w > 0:
            career_score = float((df["MainScore"] * w).sum() / total_w)
        else:
            career_score = float(df["MainScore"].mean())
    else:
        career_score = float(df["MainScore"].mean())

    try:
        career_band = score_band_5(career_score)
    except Exception:
        bands = df["MainBand_row"].dropna()
        career_band = bands.mode().iloc[0] if not bands.empty else None

    return career_score, career_band

# =========================================================
# StatsBomb-style Radar helper (self-contained)
# =========================================================

def _radar_get_coordinates(n: int) -> np.ndarray:
    """
    Liefert für n Parameter einen Winkel pro Achse.
    Output: (n, 3) – wir nutzen nur [:, 2] als Winkel.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    coord = np.zeros((n, 3), dtype=float)
    coord[:, 2] = angles
    return coord

def _radar_get_indices_between(range_list, coord_list, value, reverse=False):
    """
    Hilfsfunktion: findet die zwei Range-Werte, zwischen denen 'value' liegt,
    und interpoliert die Koordinate linear dazwischen.
    """
    rl = np.asarray(range_list, dtype=float)
    cl = np.asarray(coord_list, dtype=float)

    if reverse:
        # Werte fallen von groß -> klein
        for i in range(len(rl) - 1):
            high_v = rl[i]
            low_v = rl[i + 1]
            if high_v >= value >= low_v:
                # Anteil zwischen den beiden Werten
                span = high_v - low_v
                t = 0.0 if span == 0 else (high_v - value) / span
                x = cl[i, 0] + t * (cl[i + 1, 0] - cl[i, 0])
                y = cl[i, 1] + t * (cl[i + 1, 1] - cl[i, 1])
                return x, y
    else:
        # Werte steigen von klein -> groß
        for i in range(len(rl) - 1):
            low_v = rl[i]
            high_v = rl[i + 1]
            if low_v <= value <= high_v:
                span = high_v - low_v
                t = 0.0 if span == 0 else (value - low_v) / span
                x = cl[i, 0] + t * (cl[i + 1, 0] - cl[i, 0])
                y = cl[i, 1] + t * (cl[i + 1, 1] - cl[i, 1])
                return x, y

    # Fallback (sollte selten passieren)
    return cl[-1, 0], cl[-1, 1]

    
def render_team_radar_statsbomb_pretty(
    season: str,
    df_squad_season: pd.DataFrame,
    primary_team: str,
    compare_team: str | None = None,
    radar_type: str = "Offense",   # "Offense" oder "Defense"
):
    """
    StatsBomb-Style Squad-Radar für Offense oder Defense.

    Nutzt Rohwerte + per90 aus squads_data-<Season>.csv.
    """

    # ---- Raw laden + per90 hinzufügen ----
    try:
        df_raw = load_squad_raw_for_season(season)
        df_raw = add_squad_per90(df_raw)
    except FileNotFoundError:
        st.info("Raw squad stats missing for this season.")
        return

    if "Squad" not in df_raw.columns:
        st.info("Column 'Squad' not found in squad raw file.")
        return

    # Nur Teams der aktuellen Liga / Season
    df_raw = df_raw[df_raw["Squad"].isin(df_squad_season["Squad"])].copy()
    if df_raw.empty:
        st.info("No raw squad stats for this league/season.")
        return

    if primary_team not in df_raw["Squad"].values:
        st.info(f"Team '{primary_team}' not found in raw squad stats.")
        return

    if compare_team and compare_team not in df_raw["Squad"].values:
        st.info(f"Comparison team '{compare_team}' not found in raw squad stats.")
        compare_team = None

    df_raw = df_raw.drop_duplicates(subset=["Squad"]).set_index("Squad")

    # ---- Metrik-Set je nach Radar-Typ ----
    radar_type = radar_type.lower()
    if radar_type == "offense":
        metrics_config = OFFENSE_METRICS_CONFIG
        radar_title_suffix = "Attacking radar"
    elif radar_type == "defense":
        metrics_config = DEFENSE_METRICS_CONFIG
        radar_title_suffix = "Defending radar"
    else:
        st.warning(f"Unknown radar_type '{radar_type}'. Use 'Offense' or 'Defense'.")
        return

    # ---- Metrikliste: (Label, Spaltenname, invert) ----
    METRICS = [
        (label, col_name, invert)
        for label, (col_name, invert) in metrics_config.items()
        if col_name in df_raw.columns
    ]

    if not METRICS:
        st.info("No matching metric columns found for this radar.")
        return

    params = []
    ranges = []
    primary_vals = []
    compare_vals = []
    invert_flags = []

    for label, col, invert in METRICS:
        s = pd.to_numeric(df_raw[col], errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if s.size < 4:
            continue

        # etwas konservativer: 10.–90. Perzentil
        low = float(np.nanpercentile(s, 10))
        high = float(np.nanpercentile(s, 90))

        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(s.min())
            high = float(s.max())
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                continue

        v1 = float(df_raw.loc[primary_team, col])
        if pd.isna(v1):
            continue

        # extremwerte in den Bereich clampen (sonst kleben sie 100% außen)
        v1 = max(min(v1, high), low)

        params.append(label)
        ranges.append((low, high))
        primary_vals.append(v1)
        invert_flags.append(invert)

        if compare_team:
            v2 = float(df_raw.loc[compare_team, col])
            v2 = max(min(v2, high), low)
            compare_vals.append(v2)

    if len(params) < 3:
        st.info("Not enough metrics for the squad radar.")
        return

    # Radar-Objekt im StatsBomb-Design
    radar = Radar(
        background_color="#020617",
        patch_color="#1f2937",
        fontfamily="DejaVu Sans",

        # Labels außen: groß, weiß, fett
        label_fontsize=10,
        label_color="#F9FAFB",
        label_weight="bold",

        # Range-Werte: kleiner, grau, normal
        range_fontsize=8,
        range_color="#FFFFFF",
        range_weight="normal",
    )

    # Werte vorbereiten (invert ignoriert der Radar selbst – wir lösen das über Ranges)
    # Eine elegante Möglichkeit wäre: bei invert einfach low/high swapen:
    for i, inv in enumerate(invert_flags):
        if inv:
            lo, hi = ranges[i]
            ranges[i] = (hi, lo)

    if compare_team:
        vals = [primary_vals, compare_vals]
        colors = [VALUE_COLOR, "#4ade80"]
        # erste Fläche gleich „single team", zweite leicht transparenter
        alphas = [0.35, 0.35]
        compare_flag = True
    else:
        vals = primary_vals
        colors = [VALUE_COLOR, "#0f172a"]
        alphas = [0.35, 0.35]  # oder weglassen, dann nimmt die Radar-Klasse den Default
        compare_flag = False

    title_dict = {
        "title_name": primary_team,
        "subtitle_name": f"{radar_title_suffix} {season}",
    }
    if compare_team:
        title_dict["title_name_2"] = compare_team
        title_dict["subtitle_name_2"] = "Comparison squad"

    fig, ax = radar.plot_radar(
        ranges=ranges,
        params=params,
        values=vals,
        radar_color=colors,
        compare=compare_flag,
        alphas=alphas,
        endnote="\nData: FBref (Big-5)",
        title=title_dict,
    )

    # Hintergrund
    fig.patch.set_facecolor("#020617")
    ax.set_facecolor("#020617")

    # ---------------- LEGENDEN FÜR TEAMVERGLEICH ---------------- #
    if compare_team:
        # Einfache Marker-Handles für beide Teams
        handles = [
            plt.Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markersize=10,
                markerfacecolor=colors[0],
                markeredgecolor="none",
                label=primary_team,
            ),
            plt.Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markersize=10,
                markerfacecolor=colors[1],
                markeredgecolor="none",
                label=compare_team,
            ),
        ]

        ax.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(1.25, 1.15),   # etwas außerhalb rechts oben
            frameon=False,
            fontsize=11,
            labelcolor="#E5E7EB",
        )

    # In Streamlit anzeigen
    st.pyplot(fig)

# ---------------- TEAM SCORES MAIN VIEW (5 MODULE) ---------------- #
def render_team_scores_view(df_all: pd.DataFrame, df_squad: pd.DataFrame, df_big5: pd.DataFrame) -> None:
    """
    Main view for 'Team scores' mode.

    Modules:
      1) League ranking by squad score (table) + Big5 context (LgRk, GD, xGD, Pts/MP)
      2) Squad radar (Offense/Defense) from raw squad data
      3) Team in league context (bar)
      4) Top contributors within squad
      5) Squad development over seasons
    """
    st.sidebar.subheader("Team score filters")

    if df_squad is None or df_squad.empty:
        st.info("No squad scores available. Run the multi-season pipeline first.")
        return

    # ----- Season selection -----
    seasons = sorted(df_squad["Season"].dropna().unique())
    if not seasons:
        st.info("No seasons found in squad scores.")
        return

    season_default_idx = len(seasons) - 1
    season = st.sidebar.selectbox("Season", seasons, index=season_default_idx)

    df_squad_season = df_squad[df_squad["Season"] == season].copy()
    if df_squad_season.empty:
        st.info("No squad scores for this season.")
        return


    # ----- League / Comp filter (NOW safe) -----
    if "Comp" in df_squad_season.columns:
        comps = sorted(df_squad_season["Comp"].dropna().unique().tolist())
        comp_options = ["All"] + comps
        comp_sel = st.sidebar.selectbox(
            "League",
            comp_options,
            index=0,
            key="team_scores_league_filter",
        )

        if comp_sel != "All":
            df_squad_season = df_squad_season[df_squad_season["Comp"] == comp_sel].copy()
            if df_squad_season.empty:
                st.info("No teams for this league in this season.")
                return
    else:
        st.sidebar.caption("League filter disabled (no 'Comp' column in squad scores).")


    # Default metric (sorting)
    metric_col = "OverallScore_squad"
    metric_name = "Overall"

    # -------------------------------------------------------
    # Big5 Table merge (optional) -> add: LgRk, GD, xGD, Pts/MP
    # -------------------------------------------------------
    if df_big5 is not None and not df_big5.empty:
        df_big5_season = df_big5[df_big5["Season"] == season].copy()

        # Candidate columns (FBref kann leicht variieren)
        def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        col_squad = pick_col(df_big5_season, ["Squad", "Squad_x", "Team"])
        col_lgrk  = pick_col(df_big5_season, ["LgRk", "LeagueRank", "Lg Rank", "Rk"])
        col_gd    = pick_col(df_big5_season, ["GD"])
        col_xgd   = pick_col(df_big5_season, ["xGD", "xGD/90", "xGDiff", "xG_Diff"])
        col_ptsmp = pick_col(df_big5_season, ["Pts/MP", "Pts_per_MP", "Pts per MP"])
        col_pts = pick_col(df_big5_season, ["Pts", "Points"])


        if col_squad is None:
            st.warning("Big5 table loaded, but no Squad column found -> skipping Big5 merge.")
        else:
            keep_cols = ["Season", col_squad]
            rename_map = {col_squad: "Squad"}

            if col_lgrk:
                keep_cols.append(col_lgrk)
                rename_map[col_lgrk] = "LeagueRank"
            if col_gd:
                keep_cols.append(col_gd)
                rename_map[col_gd] = "GD"
            if col_xgd:
                keep_cols.append(col_xgd)
                rename_map[col_xgd] = "xGD"
            if col_ptsmp:
                keep_cols.append(col_ptsmp)
                rename_map[col_ptsmp] = "Pts/MP"
            if col_pts:
                keep_cols.append(col_pts)
                rename_map[col_pts] = "Pts"


            df_big5_season = df_big5_season[keep_cols].rename(columns=rename_map)

            # Make sure numeric where it should be numeric
            for c in ["LeagueRank", "Pts", "GD", "xGD", "Pts/MP"]:
                if c in df_big5_season.columns:
                    df_big5_season[c] = pd.to_numeric(df_big5_season[c], errors="coerce")

            # Merge into squad season table
            df_squad_season = df_squad_season.merge(
                df_big5_season,
                on=["Season", "Squad"],
                how="left",
            )

    # ----- Numeric + rounding prep -----
    for col in ["OverallScore_squad", "OffScore_squad", "MidScore_squad", "DefScore_squad"]:
        if col in df_squad_season.columns:
            df_squad_season[col] = pd.to_numeric(df_squad_season[col], errors="coerce")

    df_rank = (
        df_squad_season
        .dropna(subset=[metric_col])
        .sort_values("LeagueRank", ascending=True)
        .reset_index(drop=True)
    )
    if df_rank.empty:
        st.info("No valid squad scores to display.")
        return

    df_rank["Rank"] = df_rank.index + 1

    # ========== 1) LEAGUE RANKING TABLE ==========
    st.markdown("## Team scores")
    st.markdown("League-wide squad ranking using the same 0–1000 scale as the player scores.")

    # Scores -> Integer UI columns
    for src, tgt in [
        ("OverallScore_squad", "Squad Score"),
        ("OffScore_squad",     "Offense"),
        ("MidScore_squad",     "Midfield"),
        ("DefScore_squad",     "Defense"),
    ]:

        if src in df_rank.columns:
            df_rank[tgt] = (
                pd.to_numeric(df_rank[src], errors="coerce")
                .round(0)
                .astype("Int64")
            )

    # Big5 columns (optional formatting)
    if "LeagueRank" in df_rank.columns:
        df_rank["LeagueRank"] = pd.to_numeric(df_rank["LeagueRank"], errors="coerce").astype("Int64")
    if "GD" in df_rank.columns:
        df_rank["GD"] = pd.to_numeric(df_rank["GD"], errors="coerce").round(0).astype("Int64")
    if "xGD" in df_rank.columns:
        df_rank["xGD"] = pd.to_numeric(df_rank["xGD"], errors="coerce").round(2)
    if "Pts/MP" in df_rank.columns:
        df_rank["Pts/MP"] = pd.to_numeric(df_rank["Pts/MP"], errors="coerce").round(2)
    if "Pts" in df_rank.columns:
        df_rank["Pts"] = pd.to_numeric(df_rank["Pts"], errors="coerce").round(0).astype("Int64")

    # Existing extras
    if "Age_squad_mean" in df_rank.columns:
        df_rank["Age (avg)"] = pd.to_numeric(df_rank["Age_squad_mean"], errors="coerce").round(1)

    if "Min_squad" in df_rank.columns:
        df_rank["Minutes (total)"] = pd.to_numeric(df_rank["Min_squad"], errors="coerce").round(0).astype("Int64")

    if "NumPlayers_squad" in df_rank.columns:
        df_rank["Players"] = pd.to_numeric(df_rank["NumPlayers_squad"], errors="coerce").astype("Int64")

    # ---- Your desired table columns (Scores + Big5 context) ----
    cols_show = [
        "Rank",
        "Squad",
        "LeagueRank", 
        "Pts",
        "Pts/MP",  
        "GD",           
        "xGD",                 
        "Squad Score",
        "Offense",
        "Midfield",
        "Defense",
        "Age (avg)",
        "Minutes (total)",
        "Players",
    ]
    cols_show = [c for c in cols_show if c in df_rank.columns]

    st.markdown("### League ranking by squad score")

    def _build_crest_table_html(df: pd.DataFrame, cols: list[str]) -> str:
        header_cells = ""
        for c in cols:
            align = "left" if c == "Squad" else "center"
            header_cells += f'<th style="padding:6px 10px;text-align:{align};font-weight:600;border-bottom:2px solid #444;">{c}</th>'

        body_rows = ""
        for _, r in df.iterrows():
            cells = ""
            for c in cols:
                val = r.get(c, "")
                val_str = "" if (val is None or (isinstance(val, float) and pd.isna(val))) else str(val)
                if c == "Squad":
                    b64 = get_crest_b64(val_str)
                    img_tag = (
                        f'<img src="{b64}" style="width:20px;height:20px;object-fit:contain;margin-right:6px;vertical-align:middle;">'
                        if b64
                        else ""
                    )
                    cells += f'<td style="padding:6px 10px;white-space:nowrap;">{img_tag}{val_str}</td>'
                else:
                    cells += f'<td style="padding:6px 10px;text-align:center;">{val_str}</td>'
            body_rows += f'<tr style="border-bottom:1px solid #333;">{cells}</tr>'

        return (
            '<div style="overflow-x:auto;">'
            '<table style="width:100%;border-collapse:collapse;background:#1e1e1e;color:#f0f0f0;font-size:13px;font-family:system-ui,sans-serif;">'
            f'<thead><tr style="background:#2a2a2a;">{header_cells}</tr></thead>'
            f'<tbody>{body_rows}</tbody>'
            '</table></div>'
        )

    st.markdown(_build_crest_table_html(df_rank, cols_show), unsafe_allow_html=True)

    # Standard scatter (unter der Tabelle)
    render_team_scatter_under_table(df_rank, value_color=VALUE_COLOR)

    # Budget vs Squad Score scatter (only when market value data is available)
    if "TotalMarketValue_squad" in df_rank.columns:
        render_budget_scatter(df_rank, value_color=VALUE_COLOR)

    squads = df_rank["Squad"].tolist()
    if not squads:
        return

    # ================= TEAM SELECTION IN SIDEBAR =================
    placeholder_label = "Select a team..."
    options = [placeholder_label] + squads

    prev_team = st.session_state.get("team_scores_selected_team", placeholder_label)
    if prev_team not in squads:
        prev_team = placeholder_label

    team_sel = st.sidebar.selectbox(
        "Team",
        options,
        index=options.index(prev_team),
        key="team_scores_team_select",
    )
    st.session_state["team_scores_selected_team"] = team_sel

    st.markdown("---")
    st.markdown("### Squad detail")

    if team_sel == placeholder_label:
        st.info("Please select a team to see squad details.")
        return

    df_team_row = df_rank[df_rank["Squad"] == team_sel].iloc[0]

    # ---------- Squad Summary Card ----------
    overall_val = pd.to_numeric(df_team_row.get("OverallScore_squad", np.nan), errors="coerce")
    off_val     = pd.to_numeric(df_team_row.get("OffScore_squad",     np.nan), errors="coerce")
    mid_val     = pd.to_numeric(df_team_row.get("MidScore_squad",     np.nan), errors="coerce")
    def_val     = pd.to_numeric(df_team_row.get("DefScore_squad",     np.nan), errors="coerce")

    rank_val    = pd.to_numeric(df_team_row.get("Rank", np.nan), errors="coerce")
    age_val     = pd.to_numeric(df_team_row.get("Age_squad_mean", np.nan), errors="coerce")
    mins_val    = pd.to_numeric(df_team_row.get("Min_squad", np.nan), errors="coerce")
    players_val = pd.to_numeric(df_team_row.get("NumPlayers_squad", np.nan), errors="coerce")

    # Big5 card values (optional)
    lgrk_val  = pd.to_numeric(df_team_row.get("LeagueRank", np.nan), errors="coerce")
    gd_val    = pd.to_numeric(df_team_row.get("GD", np.nan), errors="coerce")
    xgd_val   = pd.to_numeric(df_team_row.get("xGD", np.nan), errors="coerce")
    ptsmp_val = pd.to_numeric(df_team_row.get("Pts/MP", np.nan), errors="coerce")

    def fmt_int(x):
        return "n/a" if pd.isna(x) else f"{int(round(float(x)))}"

    def fmt_float1(x):
        return "n/a" if pd.isna(x) else f"{float(x):.1f}"

    def fmt_float2(x):
        return "n/a" if pd.isna(x) else f"{float(x):.2f}"

    def fmt_score(x):
        return "n/a" if pd.isna(x) else f"{float(x):.0f}"

    # show Big5 line only if we actually have values
    big5_line = ""
    if any([pd.notna(lgrk_val), pd.notna(gd_val), pd.notna(xgd_val), pd.notna(ptsmp_val)]):
        big5_line = (
            f'<div style="margin-top:0.20rem; font-size:0.75rem; color:#9CA3AF;">'
            f'LgRk {fmt_int(lgrk_val)} · GD {fmt_int(gd_val)} · xGD {fmt_float2(xgd_val)} · Pts/MP {fmt_float2(ptsmp_val)}'
            f'</div>'
        )

    summary_html = f"""
    <div style="
        border-radius: 0.9rem;
        padding: 0.8rem 1.0rem;
        margin: 0.5rem 0 1.0rem 0;
        background: rgba(15,23,42,0.75);
        border: 1px solid rgba(148,163,184,0.35);
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    ">
      <div>
        <div style="font-size:0.8rem; opacity:0.8; color:#E5E7EB;">Score rank</div>
        <div style="font-size:1.6rem; font-weight:600; color:{VALUE_COLOR};">
          {fmt_int(rank_val)}
        </div>
        <div style="font-size:0.7rem; color:#9CA3AF;">by squad score</div>
      </div>
      <div>
        <div style="font-size:0.8rem; opacity:0.8; color:#E5E7EB;">Overall score</div>
        <div style="font-size:1.6rem; font-weight:600; color:{VALUE_COLOR};">
          {fmt_score(overall_val)}
        </div>
        <div style="margin-top:0.15rem; font-size:0.75rem; color:#9CA3AF;">
          Off {fmt_score(off_val)} · Mid {fmt_score(mid_val)} · Def {fmt_score(def_val)}
        </div>
        {big5_line}
      </div>
      <div>
        <div style="font-size:0.8rem; opacity:0.8; color:#E5E7EB;">Players used</div>
        <div style="font-size:1.4rem; font-weight:600; color:#F9FAFB;">
          {fmt_int(players_val)}
        </div>
        <div style="margin-top:0.15rem; font-size:0.75rem; color:#9CA3AF;">
          Total minutes {fmt_int(mins_val)}
        </div>
      </div>
      <div>
        <div style="font-size:0.8rem; opacity:0.8; color:#E5E7EB;">Avg squad age</div>
        <div style="font-size:1.4rem; font-weight:600; color:#F9FAFB;">
          {fmt_float1(age_val)} yrs
        </div>
      </div>
    </div>
    """
    st.markdown(summary_html, unsafe_allow_html=True)

    st.markdown("#### Team comparison")
    compare_options = ["(no comparison)"] + [s for s in squads if s != team_sel]
    compare_team_sel = st.selectbox("Compare with", compare_options, index=0, key="team_compare_select")
    compare_team = None if compare_team_sel == "(no comparison)" else compare_team_sel

    # ========== 2) SQUAD RADAR ==========
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### Attacking radar: {team_sel} ({season})")
        render_team_radar_statsbomb_pretty(
            season=season,
            df_squad_season=df_squad_season,
            primary_team=team_sel,
            compare_team=compare_team,
            radar_type="Offense",
        )
    with col2:
        st.markdown(f"#### Defending radar: {team_sel} ({season})")
        render_team_radar_statsbomb_pretty(
            season=season,
            df_squad_season=df_squad_season,
            primary_team=team_sel,
            compare_team=compare_team,
            radar_type="Defense",
        )

    # ========== 3) TEAM IN LEAGUE CONTEXT (BAR) ==========
    st.markdown("#### Team in Big 5 League context")
    metric_values = df_squad_season[["Squad", metric_col]].dropna().copy()

    if not metric_values.empty:
        metric_values["Score"] = pd.to_numeric(metric_values[metric_col], errors="coerce").round(0)
        metric_values["is_selected"] = metric_values["Squad"] == team_sel

        chart = (
            alt.Chart(metric_values)
            .mark_bar()
            .encode(
                x=alt.X("Squad:N", sort="-y", title=None),
                y=alt.Y("Score:Q", title=metric_name, axis=alt.Axis(format=".0f")),
                color=alt.condition(
                    alt.datum.is_selected,
                    alt.value(VALUE_COLOR),
                    alt.value("#4b5563"),
                ),
                tooltip=[
                    "Squad",
                    alt.Tooltip("Score:Q", title=metric_name, format=".0f"),
                ],
            )
            .properties(height=260)
            .configure_axis(labelColor="#E5E7EB", titleColor="#E5E7EB")
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Not enough data to show the league context chart.")

    # ========== 4) TOP CONTRIBUTORS ==========
    st.markdown("#### Top contributors within the squad")
    st.markdown(
        "<p style='font-size:0.85rem; opacity:0.8; margin-top:0.6rem;'>"
        "Impact share is the share of minutes-weighted season score within this squad."
        "</p>",
        unsafe_allow_html=True,
    )

    df_team_players = df_all[(df_all["Season"] == season) & (df_all["Squad"] == team_sel)].copy()
    if df_team_players.empty:
        st.info("No player-level data found for this squad in this season.")
        return

    df_team_players["SeasonScore"] = np.nan
    if "OverallScore_abs" in df_team_players.columns:
        df_team_players["SeasonScore"] = pd.to_numeric(df_team_players["OverallScore_abs"], errors="coerce")
    else:
        score_cols = [c for c in ["OffScore_abs", "MidScore_abs", "DefScore_abs"] if c in df_team_players.columns]
        if score_cols:
            df_team_players["SeasonScore"] = pd.to_numeric(df_team_players[score_cols].mean(axis=1), errors="coerce")

    df_team_players = df_team_players[df_team_players["SeasonScore"].notna()].copy()
    if df_team_players.empty:
        st.info("No players with valid season scores for this squad.")
        return

    minutes = pd.to_numeric(df_team_players.get("Min", 0), errors="coerce").fillna(0.0)
    df_team_players["ContributionScore"] = df_team_players["SeasonScore"] * minutes
    total_contribution = float(df_team_players["ContributionScore"].sum())
    df_team_players["ContributionShare"] = (
        df_team_players["ContributionScore"] / total_contribution * 100.0
        if total_contribution > 0 else 0.0
    )

    df_top = df_team_players.sort_values("ContributionShare", ascending=False).head(15).copy()
    if df_top.empty:
        st.info("No players with minutes and scores available for contribution chart.")
        return

    df_top["PlayerLabel"] = df_top["Player"].astype(str)

    tooltip_fields = [
        "PlayerLabel",
        alt.Tooltip("Min:Q", title="Minutes played", format=".0f"),
        alt.Tooltip("SeasonScore:Q", title="Season score", format=".0f"),
        alt.Tooltip("ContributionShare:Q", title="Impact share (%)", format=".1f"),
    ]

    row_height = 24
    chart_height = max(260, row_height * len(df_top))

    base_chart = alt.Chart(df_top).encode(
        y=alt.Y(
            "PlayerLabel:N",
            sort="-x",
            title=None,
            axis=alt.Axis(labelLimit=320, labelOverlap=False),
        ),
    )

    bars = base_chart.mark_bar(color=VALUE_COLOR, cornerRadiusEnd=6).encode(
        x=alt.X("ContributionShare:Q", title="Impact share (%)", axis=alt.Axis(format=".1f")),
        tooltip=tooltip_fields,
    )

    labels = base_chart.mark_text(
        align="right",
        baseline="middle",
        dx=-6,
        color="#F9FAFB",
        fontSize=15,
        fontWeight="bold",
    ).encode(
        x="ContributionShare:Q",
        text=alt.Text("ContributionShare:Q", format=".1f"),
    )

    contrib_chart = (
        (bars + labels)
        .properties(height=chart_height)
        .configure_axis(labelColor="#E5E7EB", titleColor="#E5E7EB")
        .configure_view(strokeWidth=0)
    )

    top3_share = df_team_players.nlargest(3, "ContributionShare")["ContributionShare"].sum()
    top5_share = df_team_players.nlargest(5, "ContributionShare")["ContributionShare"].sum()
    rest_share = max(0.0, 100.0 - top5_share)

    c_top1, c_top2, c_top3 = st.columns(3)
    with c_top1:
        st.metric("Top 3 impact share", f"{top3_share:.1f}%")
    with c_top2:
        st.metric("Top 5 impact share", f"{top5_share:.1f}%")
    with c_top3:
        st.metric("Rest of squad", f"{rest_share:.1f}%")


    # ---------- 6) Core-contributor Board (Top 3 Cards) ----------
    st.markdown("### Top 3 Impact-Players")

    top3 = df_top.head(3).copy()
    medals = ["🥇", "🥈", "🥉"]

    c1, c2, c3 = st.columns(3)
    cols = [c1, c2, c3]

    for i, (_, row) in enumerate(top3.iterrows()):
        col = cols[i]
        medal = medals[i]

        name         = str(row.get("Player", ""))
        share        = float(row.get("ContributionShare", 0.0))
        minutes_val  = float(row.get("Min", 0.0))
        season_score = float(row.get("SeasonScore", 0.0))

        card_html = f"""
        <div style="
            display:flex;
            flex-direction:column;
            justify-content:flex-start;
            align-items:flex-start;
            height:220px;
            width:100%;
            border-radius: 0.9rem;
            padding: 1rem 1.1rem;
            background: rgba(15,23,42,0.65);
            border: 1px solid rgba(255,255,255,0.08);
            box-sizing:border-box;
            color:#F9FAFB;
            font-family: system-ui, sans-serif;
        ">
            <div style="font-size:1.6rem; margin-bottom:0.4rem;">{medal}</div>

            <div style="font-size:1.15rem; font-weight:600;">
                {name}
            </div>

            <div style="margin-top:0.45rem; font-size:0.9rem; opacity:0.9;">
                Impact share:
                <span style="font-weight:700; color:{VALUE_COLOR};">{share:.1f}%</span>
            </div>

            <div style="margin-top:0.25rem; font-size:0.9rem; opacity:0.9;">
                Primary role score:
                <span style="font-weight:700;">{season_score:.0f}</span>
            </div>

            <div style="margin-top:0.25rem; font-size:0.9rem; opacity:0.9;">
                Minutes:
                <span style="font-weight:700;">{minutes_val:.0f}</span>
            </div>
        </div>
        """

        with col:
            st_html(card_html, height=240)

    st.altair_chart(contrib_chart, use_container_width=True)

    # ========== 5) SQUAD DEVELOPMENT OVER SEASONS ==========
    st.markdown("#### Squad development over seasons")

    df_team_hist = df_squad[df_squad["Squad"] == team_sel].copy()
    if df_team_hist.empty:
        st.info("No historical squad data available for this team yet.")
        return

    df_team_hist = df_team_hist.sort_values("Season")

    value_cols = [c for c in ["OffScore_squad", "MidScore_squad", "DefScore_squad", "OverallScore_squad"] if c in df_team_hist.columns]
    if not value_cols:
        st.info("No squad score columns available for historical development chart.")
        return

    rename_map = {
        "OffScore_squad": "Offense Score",
        "MidScore_squad": "Midfield Score",
        "DefScore_squad": "Defense Score",
        "OverallScore_squad": "Team Score",
    }

    for src in value_cols:
        df_team_hist[src] = pd.to_numeric(df_team_hist[src], errors="coerce")

    df_long = df_team_hist.melt(
        id_vars=["Season"],
        value_vars=value_cols,
        var_name="Component_raw",
        value_name="Score",
    )
    df_long["Component"] = df_long["Component_raw"].map(rename_map)

    color_domain = ["Team Score", "Offense Score", "Midfield Score", "Defense Score"]
    color_range = [VALUE_COLOR, "#61abd2", "#f59e0b", "#ffffff"]

    hist_chart = (
        alt.Chart(df_long)
        .mark_line(interpolate="monotone", strokeWidth=3, point=alt.OverlayMarkDef(filled=True, size=55))
        .encode(
            x=alt.X("Season:N", title="Season", sort="ascending"),
            y=alt.Y("Score:Q", title="Squad score", axis=alt.Axis(format=".0f")),
            color=alt.Color(
                "Component:N",
                title="Component",
                scale=alt.Scale(domain=color_domain, range=color_range),
                legend=alt.Legend(labelColor="#E5E7EB", titleColor="#E5E7EB"),
            ),
            tooltip=["Season:N", "Component:N", alt.Tooltip("Score:Q", format=".0f")],
        )
        .properties(height=280)
        .configure_axis(grid=False, domain=True, labelColor="#E5E7EB", titleColor="#E5E7EB")
        .configure_view(strokeWidth=0)
    )

    st.altair_chart(hist_chart, use_container_width=True)


# -------------------------------------------------------------------
# Hidden Gems page
# -------------------------------------------------------------------
def render_hidden_gems(df_all: pd.DataFrame, df_valuations: pd.DataFrame) -> None:
    """
    Shows undervalued players: high score relative to market value.
    Requires MarketValue_EUR in df_all (from Transfermarkt merge).
    """
    st.markdown("## Hidden Gems — Undervalued Players")

    if "MarketValue_EUR" not in df_all.columns:
        st.info(
            "Market value data is not yet available. "
            "Run the pipeline with SCRAPE_TRANSFERMARKT=true first."
        )
        return

    # Season selector
    seasons = sorted(df_all["Season"].dropna().unique())
    default_idx = len(seasons) - 1 if seasons else 0
    season = st.sidebar.selectbox("Season", seasons, index=default_idx, key="gems_season")

    df = df_all[df_all["Season"] == season].copy()

    # Compute primary score
    df[["MainScore", "MainBand"]] = df.apply(
        get_primary_score_and_band, axis=1, result_type="expand"
    )
    df = df[df["MainScore"].notna()].copy()

    mv_m = pd.to_numeric(df["MarketValue_EUR"], errors="coerce") / 1_000_000
    df["MarketValue_M"] = mv_m
    df = df[df["MarketValue_M"].notna() & (df["MarketValue_M"] > 0)].copy()
    df["ValueForMoney"] = df["MainScore"] / df["MarketValue_M"]

    if df.empty:
        st.info("No players with market value data found for this season.")
        return

    # Sidebar filters
    st.sidebar.subheader("Filters")
    min_score = st.sidebar.slider("Minimum score", 0, 1000, 450, 10, key="gems_min_score")
    max_mv = st.sidebar.slider(
        "Max market value (€M)", 1, 200, 20, 1, key="gems_max_mv"
    )
    min_90s = st.sidebar.slider("Minimum 90s played", 1.0, 40.0, 5.0, 0.5, key="gems_min_90s")

    _all_leagues = sorted(df["Comp"].dropna().unique().tolist()) if "Comp" in df.columns else []
    sel_leagues = st.sidebar.multiselect(
        "League", _all_leagues, default=_all_leagues, key="gems_leagues"
    )

    _all_pos = sorted(df["Pos"].dropna().unique().tolist()) if "Pos" in df.columns else []
    sel_pos = st.sidebar.multiselect(
        "Position", _all_pos, default=_all_pos, key="gems_pos"
    )

    df_gems = df[df["MainScore"] >= min_score].copy()
    df_gems = df_gems[df_gems["MarketValue_M"] <= max_mv].copy()
    if "90s" in df_gems.columns:
        df_gems = df_gems[pd.to_numeric(df_gems["90s"], errors="coerce") >= min_90s]
    if sel_leagues and "Comp" in df_gems.columns:
        df_gems = df_gems[df_gems["Comp"].isin(sel_leagues)]
    if sel_pos and "Pos" in df_gems.columns:
        df_gems = df_gems[df_gems["Pos"].isin(sel_pos)]

    if df_gems.empty:
        st.warning("No players match the current filters. Try relaxing the criteria.")
        return

    df_gems = df_gems.sort_values("ValueForMoney", ascending=False)

    # ── Gem Score (1–10 percentile-based efficiency rating) ──────────────────
    pct = df_gems["ValueForMoney"].rank(pct=True)   # 1.0 = best
    df_gems["GemScore"] = (pct * 9 + 1).round(1)

    # Table
    display_cols_map = {
        "Player": "Player",
        "Squad": "Club",
        "Comp": "League",
        "Pos": "Position",
        "Age": "Age",
        "MainScore": "Score",
        "MarketValue_M": "Market Value (€M)",
        "GemScore": "Gem Score",
        "MainBand": "Band",
    }
    table_cols = [c for c in display_cols_map if c in df_gems.columns]
    df_show = df_gems[table_cols].copy()
    df_show = df_show.rename(columns=display_cols_map)
    if "Score" in df_show.columns:
        df_show["Score"] = df_show["Score"].round(0).astype("Int64")
    if "Market Value (€M)" in df_show.columns:
        df_show["Market Value (€M)"] = df_show["Market Value (€M)"].round(1)

    st.markdown(f"### Top {len(df_show)} Hidden Gems — {season}")
    st.caption("Click a row to open the player profile · Gem Score = score-to-market-value efficiency, ranked 1–10 within the current selection")
    gems_event = st.dataframe(
        df_show,
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        key="gems_df_select",
        column_config={
            "Gem Score": st.column_config.ProgressColumn(
                "Gem Score",
                help="Score-to-market-value efficiency ranked within the current selection. 10 = best deal.",
                min_value=1,
                max_value=10,
                format="%.1f",
            ),
        },
    )
    if gems_event.selection.rows:
        idx = gems_event.selection.rows[0]
        st.session_state["_nav_to_player"] = df_gems.iloc[idx]["Player"]
        st.rerun()

    # Download button
    csv_bytes = df_show.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download as CSV",
        data=csv_bytes,
        file_name=f"hidden_gems_{season}.csv",
        mime="text/csv",
        key="gems_download",
    )

    # ── Quadrant Chart: Score vs Market Value ─────────────────────────────────
    st.markdown("### Score vs. Market Value")
    import plotly.graph_objects as go

    def _strip_icon(band: str) -> str:
        for b in BAND_COLORS:
            if b in str(band):
                return b
        return str(band)

    df_plot = df_gems[
        ["Player", "Squad", "MainScore", "MarketValue_M", "MainBand", "ValueForMoney"]
    ].dropna().copy()
    df_plot["BandClean"] = df_plot["MainBand"].apply(_strip_icon)

    if not df_plot.empty:
        med_score = float(df_plot["MainScore"].median())
        med_mv    = float(df_plot["MarketValue_M"].median())
        x_min     = max(0.0, float(df_plot["MainScore"].min()) - 50)
        x_max     = min(float(df_plot["MainScore"].max()) * 1.06, 1080)
        y_min     = 0.5   # fixed lower bound — log scale starts at €0.5M
        y_max     = float(df_plot["MarketValue_M"].max()) * 1.40

        # Corner x positions: 4% inside each quadrant's outer edge
        _x_left  = x_min + (med_score - x_min) * 0.04
        _x_right = x_max - (x_max - med_score) * 0.04

        # ── Quadrant zones ───────────────────────────────────────────────────
        # (x0, x1, y0, y1, fill, label, lx, ly_paper, xanchor, yanchor)
        quadrants = [
            (x_min,     med_score, med_mv, y_max,  "rgba(234,179,8,0.07)",   "Overpriced",     _x_left,  0.96, "left",  "top"),
            (med_score, x_max,     med_mv, y_max,  "rgba(59,130,246,0.07)",  "Stars",          _x_right, 0.96, "right", "top"),
            (x_min,     med_score, y_min,  med_mv, "rgba(107,114,128,0.07)", "Budget Options", _x_left,  0.04, "left",  "bottom"),
            (med_score, x_max,     y_min,  med_mv, "rgba(34,197,94,0.10)",   "Hidden Gems",    _x_right, 0.04, "right", "bottom"),
        ]
        shapes = []
        annotations = []
        for x0, x1, y0, y1, color, label, lx, ly_paper, xanch, yanch in quadrants:
            shapes.append(dict(
                type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                fillcolor=color, line_width=0, layer="below",
            ))
            annotations.append(dict(
                x=lx, y=ly_paper,
                xref="x", yref="paper",
                text=f"<b>{label}</b>",
                showarrow=False,
                font=dict(size=10, color="rgba(229,231,235,0.45)"),
                xanchor=xanch, yanchor=yanch,
            ))

        # ── Median reference lines ────────────────────────────────────────────
        shapes += [
            dict(type="line", x0=med_score, x1=med_score, y0=y_min, y1=y_max,
                 line=dict(color="rgba(229,231,235,0.30)", width=1, dash="dot")),
            dict(type="line", x0=x_min, x1=x_max, y0=med_mv, y1=med_mv,
                 line=dict(color="rgba(229,231,235,0.30)", width=1, dash="dot")),
        ]
        annotations += [
            dict(x=med_score, y=y_max * 0.92,
                 text=f"Median {med_score:.0f}",
                 showarrow=False,
                 font=dict(size=9, color="rgba(229,231,235,0.50)"),
                 xanchor="center", yanchor="top"),
            dict(x=x_max * 0.998, y=med_mv * 1.15,
                 text=f"Median \u20ac{med_mv:.1f}M",
                 showarrow=False,
                 font=dict(size=9, color="rgba(229,231,235,0.50)"),
                 xanchor="right", yanchor="bottom"),
        ]

        # ── Scatter traces per band ───────────────────────────────────────────
        fig = go.Figure()
        for band, color in BAND_COLORS.items():
            mask = df_plot["BandClean"] == band
            if mask.sum() == 0:
                continue
            sub = df_plot[mask]
            hover = [
                (
                    f"<b>{row['Player']}</b><br>{row['Squad']}<br>"
                    f"Score: {row['MainScore']:.0f} &nbsp;|&nbsp; "
                    f"\u20ac{row['MarketValue_M']:.1f}M<br>"
                    f"Value/\u20acM: {row['ValueForMoney']:.1f} pts<br>"
                    f"Band: {row['BandClean']}"
                )
                for _, row in sub.iterrows()
            ]
            fig.add_trace(go.Scatter(
                x=sub["MainScore"],
                y=sub["MarketValue_M"],
                mode="markers",
                name=band,
                marker=dict(
                    color=color, size=9, opacity=0.88,
                    line=dict(width=0.6, color="rgba(255,255,255,0.25)"),
                ),
                hovertext=hover,
                hoverinfo="text",
            ))

        fig.update_layout(
            height=430,
            paper_bgcolor="#0D1117",
            plot_bgcolor="#161B22",
            font=dict(color="#E5E7EB", family="sans-serif"),
            xaxis=dict(
                title="PlayerScore",
                range=[x_min, x_max],
                showgrid=False, zeroline=False,
                tickcolor="#374151", linecolor="#374151",
                title_font=dict(size=12),
            ),
            yaxis=dict(
                title="Market Value (\u20acM) — log scale",
                type="log",
                range=[np.log10(y_min), np.log10(y_max)],
                tickvals=[v for v in [0.5, 1, 2, 5, 10, 20, 50, 100, 200]
                          if y_min <= v <= y_max],
                ticktext=[f"\u20ac{v:.0f}M" for v in [0.5, 1, 2, 5, 10, 20, 50, 100, 200]
                          if y_min <= v <= y_max],
                showgrid=True,
                gridcolor="rgba(55,65,81,0.35)",
                zeroline=False,
                tickcolor="#374151", linecolor="#374151",
                title_font=dict(size=12),
            ),
            shapes=shapes,
            annotations=annotations,
            legend=dict(
                title="Band",
                bgcolor="rgba(22,27,34,0.85)",
                bordercolor="#374151", borderwidth=1,
                font=dict(size=10),
                x=1.01, y=1, xanchor="left", yanchor="top",
            ),
            margin=dict(l=60, r=20, t=30, b=50),
            hoverlabel=dict(
                bgcolor="#161B22",
                bordercolor="#374151",
                font=dict(color="#F9FAFB", size=12),
            ),
            dragmode="pan",
        )

        st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------------
# Player Comparison mode
# -------------------------------------------------------------------
def render_player_comparison(df_all: pd.DataFrame, df_valuations: pd.DataFrame) -> None:
    """Side-by-side comparison of two players."""

    st.header("Compare Players")

    all_seasons = sorted(df_all["Season"].dropna().unique(), reverse=True) if "Season" in df_all.columns else []
    if not all_seasons:
        st.info("No season data available.")
        return

    season = st.sidebar.selectbox("Season", all_seasons, key="comp_season")
    df_season = df_all[df_all["Season"] == season].copy() if season else df_all.copy()

    leagues = sorted(df_season["Comp"].dropna().unique()) if "Comp" in df_season.columns else []

    # ── Player 1 ──────────────────────────────────────────────────────
    st.sidebar.markdown("**Player 1**")
    p1_league = st.sidebar.selectbox("League (P1)", ["All"] + leagues, key="comp_p1_league")
    df_p1 = df_season[df_season["Comp"] == p1_league] if p1_league != "All" else df_season
    p1_clubs = sorted(df_p1["Squad"].dropna().unique()) if "Squad" in df_p1.columns else []
    p1_club = st.sidebar.selectbox("Club (P1)", ["All"] + p1_clubs, key="comp_p1_club")
    df_p1 = df_p1[df_p1["Squad"] == p1_club] if p1_club != "All" else df_p1
    p1_players = sorted(df_p1["Player"].dropna().unique()) if "Player" in df_p1.columns else []
    p1_name = st.sidebar.selectbox("Player 1", ["—"] + p1_players, key="comp_p1_player")

    # ── Player 2 ──────────────────────────────────────────────────────
    st.sidebar.markdown("**Player 2**")
    p2_league = st.sidebar.selectbox("League (P2)", ["All"] + leagues, key="comp_p2_league")
    df_p2 = df_season[df_season["Comp"] == p2_league] if p2_league != "All" else df_season
    p2_clubs = sorted(df_p2["Squad"].dropna().unique()) if "Squad" in df_p2.columns else []
    p2_club = st.sidebar.selectbox("Club (P2)", ["All"] + p2_clubs, key="comp_p2_club")
    df_p2 = df_p2[df_p2["Squad"] == p2_club] if p2_club != "All" else df_p2
    p2_players = sorted(df_p2["Player"].dropna().unique()) if "Player" in df_p2.columns else []
    p2_name = st.sidebar.selectbox("Player 2", ["—"] + p2_players, key="comp_p2_player")

    if p1_name == "—" or p2_name == "—":
        st.info("Select two players in the sidebar to compare them.")
        return

    if p1_name == p2_name:
        st.warning("Please select two different players.")
        return

    # ── Fetch rows ────────────────────────────────────────────────────
    def _get_player_row(df: pd.DataFrame, name: str) -> pd.Series | None:
        rows = df[df["Player"] == name]
        if rows.empty:
            return None
        return enrich_card_row_with_per90(rows.iloc[0].copy())

    row1 = _get_player_row(df_season, p1_name)
    row2 = _get_player_row(df_season, p2_name)

    if row1 is None or row2 is None:
        st.error("Could not load data for one of the selected players.")
        return

    score1, band1 = get_primary_score_and_band(row1)
    score2, band2 = get_primary_score_and_band(row2)
    row1["MainScore"] = score1
    row1["MainBand"] = band1
    row2["MainScore"] = score2
    row2["MainBand"] = band2

    tab_overview, tab_deepdive = st.tabs(["Overview", "Deep Dive"])

    with tab_overview:
        # ── FIFA Cards ────────────────────────────────────────────────
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown(f"**{p1_name}**")
            if score1 is not None and not pd.isna(score1):
                render_fifa_card(row1, primary_score_col="MainScore", band_col="MainBand", title=None)
                try:
                    png1 = generate_player_card_png(row1, score1, band1, fmt_market_value)
                    st.download_button(
                        "📤 Share Card",
                        data=png1,
                        file_name=f"{p1_name.replace(' ', '_')}_{season}_card.png",
                        mime="image/png",
                        key="comp_dl_p1",
                    )
                except Exception:
                    pass
        with col_c2:
            st.markdown(f"**{p2_name}**")
            if score2 is not None and not pd.isna(score2):
                render_fifa_card(row2, primary_score_col="MainScore", band_col="MainBand", title=None)
                try:
                    png2 = generate_player_card_png(row2, score2, band2, fmt_market_value)
                    st.download_button(
                        "📤 Share Card",
                        data=png2,
                        file_name=f"{p2_name.replace(' ', '_')}_{season}_card.png",
                        mime="image/png",
                        key="comp_dl_p2",
                    )
                except Exception:
                    pass

        # ── Score comparison bar chart ────────────────────────────────
        st.markdown("### Score Comparison")

        score_cols = {
            "Offense": ("OffScore_abs", "OffScore"),
            "Midfield": ("MidScore_abs", "MidScore"),
            "Defense": ("DefScore_abs", "DefScore"),
            "Overall": ("MainScore", "MainScore"),
        }

        score_records = []
        for label, (col_a, col_b) in score_cols.items():
            for row, name in [(row1, p1_name), (row2, p2_name)]:
                val = None
                for c in (col_a, col_b):
                    if c in row.index and pd.notna(row[c]):
                        try:
                            val = float(row[c])
                            break
                        except (ValueError, TypeError):
                            pass
                if val is not None:
                    score_records.append({"Score Type": label, "Player": name, "Score": val})

        if score_records:
            df_scores = pd.DataFrame(score_records)
            bar = (
                alt.Chart(df_scores)
                .mark_bar()
                .encode(
                    x=alt.X("Score Type:N", axis=alt.Axis(labelColor="#E5E7EB", titleColor="#E5E7EB")),
                    y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1000]), axis=alt.Axis(labelColor="#E5E7EB", titleColor="#E5E7EB")),
                    color=alt.Color(
                        "Player:N",
                        scale=alt.Scale(domain=[p1_name, p2_name], range=[VALUE_COLOR, "#fde68a"]),
                        legend=alt.Legend(labelColor="#E5E7EB", titleColor="#E5E7EB"),
                    ),
                    xOffset="Player:N",
                    tooltip=["Player:N", "Score Type:N", alt.Tooltip("Score:Q", format=".0f")],
                )
                .properties(height=300)
                .configure_view(strokeWidth=0)
                .configure_axis(grid=False, domain=True)
            )
            st.altair_chart(bar, use_container_width=True)

    with tab_deepdive:
        # ── Pizza charts ──────────────────────────────────────────────
        st.markdown("### Style Profiles")

        # Pizza charts need raw feature table (per-90 metrics), not processed scores
        try:
            df_features = load_feature_table_for_season(season)
        except (FileNotFoundError, Exception):
            df_features = pd.DataFrame()

        if df_features.empty:
            st.info("Raw feature data not available for pizza charts in this season.")
        else:
            df_feat_p1 = df_features[df_features["Player"] == p1_name]
            df_feat_p2 = df_features[df_features["Player"] == p2_name]

            role1 = df_feat_p1["Pos"].iloc[0] if not df_feat_p1.empty and "Pos" in df_feat_p1.columns else None
            role2 = df_feat_p2["Pos"].iloc[0] if not df_feat_p2.empty and "Pos" in df_feat_p2.columns else None

            col_pz1, col_pz2 = st.columns(2)
            with col_pz1:
                st.markdown(f"**{p1_name}**")
                fig1 = render_pizza_chart(df_features, df_feat_p1, role1, season)
                if fig1 is not None:
                    st.pyplot(fig1, use_container_width=True)
            with col_pz2:
                st.markdown(f"**{p2_name}**")
                fig2 = render_pizza_chart(df_features, df_feat_p2, role2, season)
                if fig2 is not None:
                    st.pyplot(fig2, use_container_width=True)

        # ── Metrics table ─────────────────────────────────────────────
        st.markdown("### Key Metrics")

        metric_candidates = [
            ("Goals/90",      ["G-PK_Per90", "G-PK/90"]),
            ("xG/90",         ["npxG_Per90", "npxG/90"]),
            ("Assists/90",    ["Ast_Per90", "Ast/90"]),
            ("Key Passes/90", ["KP_Per90", "KP/90"]),
            ("Prog Pass/90",  ["PrgP_Per90", "PrgP/90"]),
            ("Prog Carry/90", ["PrgC_Per90", "PrgC/90"]),
            ("Tackles W/90",  ["TklW_Per90", "TklW/90"]),
            ("Interc./90",    ["Int_Per90", "Int/90"]),
            ("Clearances/90", ["Clr_Per90", "Clr/90"]),
            ("Pass Cmp%",     ["Cmp%"]),
            ("Minutes",       ["Min", "90s"]),
        ]

        def _resolve_val(row: pd.Series, candidates: list[str]) -> str:
            for c in candidates:
                if c in row.index and pd.notna(row[c]):
                    try:
                        v = float(row[c])
                        return f"{v:.2f}" if c != "Min" else f"{v:.0f}"
                    except (ValueError, TypeError):
                        pass
            return "—"

        table_rows = []
        for label, candidates in metric_candidates:
            v1 = _resolve_val(row1, candidates)
            v2 = _resolve_val(row2, candidates)
            if v1 != "—" or v2 != "—":
                table_rows.append({"Metric": label, p1_name: v1, p2_name: v2})

        if table_rows:
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="PlayerScore",
        layout="wide",
        page_icon="📊",
    )

    # ---- Custom CSS ----
    st.markdown(
        """
        <style>
        html, body, [class*="css"]  {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .ps-title {
            font-size: 2.4rem;
            font-weight: 700;
            color: #F9FAFB;
            margin-bottom: 0.15rem;
        }
        .ps-subtitle {
            font-size: 0.95rem;
            color: #94A3B8;
            margin-bottom: 1.6rem;
        }
        .stDataFrame tbody td {
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    data_version = get_data_version()

    df_all, df_agg, df_squad, df_big5, df_valuations = load_data(data_version)

    st.caption(f"🕒 Data last updated: {data_version}")

    if "Pos_raw" not in df_all.columns:
        df_all["Pos_raw"] = df_all["Pos"]

    pos_map_display = {
        "Off_MF": "MF",
        "Def_MF": "DF",
    }
    df_all["Pos"] = df_all["Pos_raw"].map(pos_map_display).fillna(df_all["Pos_raw"])

    if df_all.empty:
        st.info("No processed data found yet. Run the pipeline locally and push the CSVs or Kaggle sync.")
        st.stop()

    # ── Deep-link: read URL params into session state (runs once on load) ──
    _qp = st.query_params
    if "mode" in _qp and "main_mode" not in st.session_state:
        st.session_state["main_mode"] = _qp["mode"]
    if "player" in _qp:
        st.session_state.setdefault("pp_selected_player", _qp["player"])

    # Handle pending navigation from Rankings (must run before radio renders)
    if "_nav_to_player" in st.session_state:
        st.session_state["pp_selected_player"] = st.session_state.pop("_nav_to_player")
        st.session_state["pp_source"] = "global"
        st.session_state["main_mode"] = "Player profile"

    st.sidebar.header("PlayerScore")
    st.sidebar.caption("📊 FBref · Big-5 Leagues · Updated weekly")
    mode = st.sidebar.radio(
        "Navigate",
        ["Home", "Player profile", "Player Rankings", "Team scores", "Hidden Gems", "Compare Players"],
        key="main_mode",
    )
    # ── Deep-link: keep URL in sync with current mode ──
    st.query_params["mode"] = mode

    _c_exc = BAND_COLORS["Exceptional"]
    _c_wc  = BAND_COLORS["World Class"]
    _c_ts  = BAND_COLORS["Top Starter"]
    _c_ss  = BAND_COLORS["Solid Squad Player"]
    _c_bb  = BAND_COLORS["Below Big-5 Level"]
    st.sidebar.markdown(
        f"<div style='font-size:0.72rem;color:#6B7280;margin-top:0.5rem;line-height:1.6;'>"
        f"Score 0\u20131000<br>"
        f"<span style='color:{_c_exc}'>&#9679;</span> Exceptional \u2265 900&nbsp;&nbsp;"
        f"<span style='color:{_c_wc}'>&#9679;</span> World Class \u2265 750<br>"
        f"<span style='color:{_c_ts}'>&#9679;</span> Top Starter \u2265 400&nbsp;&nbsp;"
        f"<span style='color:{_c_ss}'>&#9679;</span> Solid \u2265 200<br>"
        f"<span style='color:{_c_bb}'>&#9679;</span> Below Big-5 &lt; 200"
        f"</div>",
        unsafe_allow_html=True,
    )

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
    # MODE: HOME
    # ==================================================================
    if mode == "Home":
        # ── Dynamic KPIs from loaded data ──────────────────────────────
        n_players = int(df_all["Player"].nunique()) if "Player" in df_all.columns else 0
        n_seasons = int(df_all["Season"].nunique()) if "Season" in df_all.columns else 0
        n_leagues = int(df_all["Comp"].nunique()) if "Comp" in df_all.columns else 5

        # ── Hero ───────────────────────────────────────────────────────
        st.markdown(
            f"""
            <div style="padding: 2rem 0 1.5rem 0;">
                <h1 style="font-size:2.8rem; font-weight:800;
                           letter-spacing:-0.03em; margin-bottom:0.3rem;">
                    <span style="color:{VALUE_COLOR};">Player</span><span style="color:#FFFFFF;">Score</span>
                </h1>
                <p style="font-size:1.15rem; color:#94A3B8; margin:0; max-width:580px;">
                    Role-aware player performance analytics for Europe's Big-5 leagues —
                    transparent, benchmark-driven, updated weekly.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── KPI tiles ──────────────────────────────────────────────────
        k1, k2, k3 = st.columns(3)
        kpi_style = (
            "border-radius:0.75rem; padding:1.1rem 1.4rem; "
            "background:rgba(0,184,169,0.08); border:1px solid rgba(0,184,169,0.25); "
            "text-align:center;"
        )
        with k1:
            st.markdown(
                f"""<div style="{kpi_style}">
                    <div style="font-size:2.2rem;font-weight:800;color:{VALUE_COLOR};">{n_players:,}</div>
                    <div style="font-size:0.85rem;color:#94A3B8;margin-top:0.2rem;">Players tracked</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"""<div style="{kpi_style}">
                    <div style="font-size:2.2rem;font-weight:800;color:{VALUE_COLOR};">{n_seasons}</div>
                    <div style="font-size:0.85rem;color:#94A3B8;margin-top:0.2rem;">Seasons covered</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f"""<div style="{kpi_style}">
                    <div style="font-size:2.2rem;font-weight:800;color:{VALUE_COLOR};">{n_leagues}</div>
                    <div style="font-size:0.85rem;color:#94A3B8;margin-top:0.2rem;">Leagues</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

        # ── Global player search ───────────────────────────────────────
        st.markdown("### Player search")
        _all_players_home = sorted(df_all["Player"].dropna().unique().tolist())
        _home_pick = st.selectbox(
            "Search any player…",
            [""] + _all_players_home,
            key="home_search_box",
        )
        if _home_pick:
            st.session_state["_nav_to_player"] = _home_pick
            st.rerun()

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

        # ── Feature cards ─────────────────────────────────────────────
        card_style = (
            "border-radius:0.75rem; padding:1.2rem 1.4rem; height:100%; "
            "background:#161B22; border:1px solid #21262D;"
        )
        icon_style = f"font-size:1.6rem; margin-bottom:0.5rem;"
        title_style = f"font-size:1rem; font-weight:700; color:#F9FAFB; margin-bottom:0.3rem;"
        desc_style = "font-size:0.85rem; color:#94A3B8; line-height:1.5;"

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""<div style="{card_style}">
                    <div style="{icon_style}">👤</div>
                    <div style="{title_style}">Player Profiles</div>
                    <div style="{desc_style}">
                        Pizza charts, scatter plots, career trends and market value history
                        for any player across all seasons.
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""<div style="{card_style}">
                    <div style="{icon_style}">📊</div>
                    <div style="{title_style}">Player Rankings</div>
                    <div style="{desc_style}">
                        Filter by league, club, position and age. Click any player
                        to jump straight to their profile.
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown(
                f"""<div style="{card_style}">
                    <div style="{icon_style}">💎</div>
                    <div style="{title_style}">Hidden Gems</div>
                    <div style="{desc_style}">
                        Discover undervalued players with high scores relative
                        to their market value — built for scouts and analysts.
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"""<div style="{card_style}">
                    <div style="{icon_style}">⚖️</div>
                    <div style="{title_style}">Compare Players</div>
                    <div style="{desc_style}">
                        Side-by-side comparison of two players: scores,
                        radar profiles and key metrics at a glance.
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

        # ── Scoring system explainer (compact) ────────────────────────
        st.markdown(
            f"<div style='font-size:0.8rem;color:#6B7280;border-top:1px solid #21262D;"
            f"padding-top:0.8rem;'>Scores range 0–1000 and are computed per role "
            f"(FW · MF · DF) using per-90 benchmarks from FBref Big-5 data. "
            f"Updated automatically every Tuesday via GitHub Actions.</div>",
            unsafe_allow_html=True,
        )
        return

    # ==================================================================
    # MODE: PLAYER PROFILE
    # ==================================================================
    if mode == "Player profile":
        st.sidebar.subheader("Profile filters")

        # --- Required columns check ---
        required = {"Player", "Season", "Comp", "Squad"}
        missing = [c for c in required if c not in df_all.columns]
        if missing:
            st.warning(f"Missing required columns: {missing}")
            return

        comp_col = "Comp"
        squad_col = "Squad"

        # =========================
        # Current season (filters should ONLY show players currently at club)
        # =========================
        seasons_all = sorted(df_all["Season"].dropna().unique())
        if not seasons_all:
            st.warning("No seasons found in data.")
            return

        CURRENT_SEASON = seasons_all[-1]
        df_current = df_all[df_all["Season"] == CURRENT_SEASON].copy()

        # =========================
        # Session state (selection source)
        # =========================
        st.session_state.setdefault("pp_selected_player", None)
        st.session_state.setdefault("pp_source", None)  # "global" | "filtered"

        GLOBAL_PLACEHOLDER = "Search a player..."
        FILTER_PLACEHOLDER = "Select a player..."
        LEAGUE_PLACEHOLDER = "All leagues"
        CLUB_PLACEHOLDER = "All clubs"

        # --- Callbacks: last action wins (no illegal session_state writes) ---
        def _on_global_change():
            val = st.session_state.get("pp_global_player_selectbox", GLOBAL_PLACEHOLDER)
            if val != GLOBAL_PLACEHOLDER:
                st.session_state["pp_selected_player"] = val
                st.session_state["pp_source"] = "global"

        def _on_filtered_change():
            val = st.session_state.get("pp_filtered_player_selectbox", FILTER_PLACEHOLDER)
            if val != FILTER_PLACEHOLDER:
                st.session_state["pp_selected_player"] = val
                st.session_state["pp_source"] = "filtered"

        # =========================
        # 1) Global player search (ALL seasons)
        # =========================
        players_global = (
            df_all["Player"].dropna().astype(str).sort_values().unique().tolist()
        )
        global_options = [GLOBAL_PLACEHOLDER] + players_global

        global_default = GLOBAL_PLACEHOLDER
        if (
            st.session_state.get("pp_source") == "global"
            and st.session_state.get("pp_selected_player") in global_options
        ):
            global_default = st.session_state["pp_selected_player"]

        st.sidebar.selectbox(
            "Search player (all seasons)",
            options=global_options,
            index=global_options.index(global_default),
            key="pp_global_player_selectbox",
            on_change=_on_global_change,
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown("**Or browse by club**")
        st.sidebar.caption(f"Showing current season ({CURRENT_SEASON}) squads.")

        # =========================
        # 2) League filter (CURRENT SEASON)
        # =========================
        leagues = sorted(df_current[comp_col].dropna().unique().tolist())
        league_options = [LEAGUE_PLACEHOLDER] + leagues

        prev_league = st.session_state.get("pp_league_selectbox", LEAGUE_PLACEHOLDER)
        if prev_league not in league_options:
            prev_league = LEAGUE_PLACEHOLDER

        league_sel = st.sidebar.selectbox(
            "League",
            options=league_options,
            index=league_options.index(prev_league),
            key="pp_league_selectbox",
        )

        df_filt = df_current.copy()
        if league_sel != LEAGUE_PLACEHOLDER:
            df_filt = df_filt[df_filt[comp_col] == league_sel]

        # =========================
        # 3) Club filter (CURRENT SEASON, depends on league)
        # =========================
        clubs = sorted(df_filt[squad_col].dropna().unique().tolist())
        club_options = [CLUB_PLACEHOLDER] + clubs

        prev_club = st.session_state.get("pp_club_selectbox", CLUB_PLACEHOLDER)
        if prev_club not in club_options:
            prev_club = CLUB_PLACEHOLDER

        club_sel = st.sidebar.selectbox(
            "Club",
            options=club_options,
            index=club_options.index(prev_club),
            key="pp_club_selectbox",
        )

        if club_sel != CLUB_PLACEHOLDER:
            df_filt = df_filt[df_filt[squad_col] == club_sel]

        # =========================
        # 4) Player list (CURRENT SEASON only + filtered by league/club)
        # =========================
        players_filtered = (
            df_filt["Player"].dropna().astype(str).sort_values().unique().tolist()
        )
        filtered_options = [FILTER_PLACEHOLDER] + players_filtered

        filtered_default = FILTER_PLACEHOLDER
        if (
            st.session_state.get("pp_source") == "filtered"
            and st.session_state.get("pp_selected_player") in filtered_options
        ):
            filtered_default = st.session_state["pp_selected_player"]

        st.sidebar.selectbox(
            f"Players at club in {CURRENT_SEASON}",
            options=filtered_options,
            index=filtered_options.index(filtered_default),
            key="pp_filtered_player_selectbox",
            on_change=_on_filtered_change,
        )

        # =========================
        # 5) Final selected player (last action wins)
        # =========================
        player = st.session_state.get("pp_selected_player", None)
        if not player:
            st.subheader("Player profile")
            st.caption("Use the search in the sidebar to find any player — or start with one of these:")

            # Suggest top 6 players from current season by MainScore
            _df_suggest = df_current.copy()
            _df_suggest[["MainScore", "MainBand"]] = _df_suggest.apply(
                get_primary_score_and_band, axis=1, result_type="expand"
            )
            _df_suggest = (
                _df_suggest[_df_suggest["MainScore"].notna()]
                .sort_values("MainScore", ascending=False)
                .head(6)
            )
            _suggest_cols = st.columns(3)
            for _i, (_, _srow) in enumerate(_df_suggest.iterrows()):
                _pname = _srow["Player"]
                _squad = _srow.get("Squad", "")
                _score = f"{_srow['MainScore']:.0f}"
                with _suggest_cols[_i % 3]:
                    if st.button(
                        f"**{_pname}**\n{_squad} · {_score}",
                        key=f"suggest_{_i}",
                        use_container_width=True,
                    ):
                        st.session_state["pp_selected_player"] = _pname
                        st.session_state["pp_source"] = "global"
                        st.rerun()
            return

        st.session_state["selected_player_label"] = player

        # ── Deep-link: keep URL in sync with selected player ──
        if player:
            st.query_params["player"] = player
        else:
            st.query_params.pop("player", None)

        # alle Saisons & Clubs des Spielers
        player_squad = None
        df_player_all = df_all[df_all["Player"] == player].copy()

        if "Season" in df_player_all.columns:
            seasons = sorted(df_player_all["Season"].dropna().unique())
        else:
            seasons = sorted(df_all["Season"].dropna().unique())

        if not seasons:
            st.warning("No seasons found for this player.")
            return

        st.sidebar.markdown("---")
        profile_view = st.sidebar.radio(
            "Profile view",
            ["Per season", "Career"],
            key="profile_view",
        )

        season = None
        if profile_view == "Per season":
            default_season_idx = len(seasons) - 1 if seasons else 0
            season = st.sidebar.selectbox(
                "Season",
                seasons,
                index=default_season_idx,
                key=f"profile_season_{player}",
            )

        if profile_view == "Per season" and season is not None:
            df_player = df_player_all[df_player_all["Season"] == season].copy()
        else:
            df_player = df_player_all.copy()

        st.subheader(f"Player Profile – {player}")

        typical_pos = df_player_all["Pos"].dropna().mode()
        typical_pos = typical_pos.iloc[0] if not typical_pos.empty else None

        if profile_view == "Per season" and not df_player.empty:
            pos_txt = df_player["Pos"].iloc[0] if "Pos" in df_player.columns else None
            squad_txt = df_player["Squad"].iloc[0] if "Squad" in df_player.columns else None
            details = " | ".join(x for x in [pos_txt, squad_txt] if x)
            crest_path = get_crest_path(squad_txt or "")
            if crest_path:
                c_img, c_txt = st.columns([0.08, 0.92])
                with c_img:
                    st.image(str(crest_path), width=52)
                with c_txt:
                    if details:
                        st.caption(details)
            else:
                if details:
                    st.caption(details)
        else:
            st.caption(get_role_label(typical_pos))

        role = typical_pos or (df_player["Pos"].iloc[0] if not df_player.empty and "Pos" in df_player.columns else None)

        if profile_view == "Career":
            if "Season" in df_player_all.columns and not df_player_all.empty:
                first_season = df_player_all["Season"].min()
                last_season = df_player_all["Season"].max()
                st.caption(f"Career: {first_season} – {last_season}")
            else:
                st.caption("Seasons: n/a")

        if not df_player.empty:
            df_player = df_player.copy()
            df_player[["MainScore", "MainBand"]] = df_player.apply(
                get_primary_score_and_band,
                axis=1,
                result_type="expand",
            )

        # ----- FIFA Card + Score-Logik -----
        card_row = None
        main_score = None
        main_band = None
        pizza_fig_for_pdf    = None   # captured later for PDF export
        scatter_df_all_pdf   = None   # captured later for PDF scatter
        scatter_df_player_pdf = None

        if profile_view == "Per season" and not df_player.empty:
            card_row = df_player.iloc[0].copy()
            card_row = enrich_card_row_with_per90(card_row)
            main_score, main_band = get_primary_score_and_band(card_row)

        elif profile_view == "Career" and not df_player_all.empty:
            card_row = build_career_card_row(df_player_all, player)
            main_score, main_band = compute_career_main_score(df_player_all)

        elif not df_player_all.empty:
            if "Season" in df_player_all.columns:
                card_row = df_player_all.sort_values("Season").iloc[-1].copy()
            else:
                card_row = df_player_all.iloc[0].copy()
            card_row = enrich_card_row_with_per90(card_row)
            main_score, main_band = get_primary_score_and_band(card_row)

        if card_row is not None and main_score is not None and not pd.isna(main_score):
            card_row["MainScore"] = main_score
            card_row["MainBand"] = main_band

            render_fifa_card(
                card_row,
                primary_score_col="MainScore",
                band_col="MainBand",
                title=None,
            )

            try:
                png_bytes = generate_player_card_png(card_row, main_score, main_band, fmt_market_value)
                season_label = card_row.get("Season", "career") or "career"
                st.download_button(
                    "📤 Share Card",
                    data=png_bytes,
                    file_name=f"{player.replace(' ', '_')}_{season_label}.png",
                    mime="image/png",
                    key="dl_player_card",
                )
            except Exception:
                pass
        elif card_row is not None:
            st.info("No primary role score available for this player in the selected view.")

        # ===================== SUMMARY TILES =====================
        # FIFA card already shows: name, club, score, band, position, season.
        # Tiles add: score (prominent), band, minutes, league (per-season)
        #            or career score, band, total 90s, seasons played (career).

        _tile = (
            "border-radius:0.6rem;padding:0.65rem 0.9rem;"
            "background:#161B22;border:1px solid #21262D;height:100%;"
        )

        col1, col2, col3, col4 = st.columns(4)

        if profile_view == "Per season":
            main_score_tile = None
            main_band_tile = None
            min_val = None
            comp_val = None

            if not df_player.empty:
                if "MainScore" in df_player.columns:
                    main_score_tile = df_player["MainScore"].iloc[0]
                if "MainBand" in df_player.columns:
                    main_band_tile = df_player["MainBand"].iloc[0]
                if "Min" in df_player.columns:
                    min_val = df_player["Min"].iloc[0]
                if "Comp" in df_player.columns:
                    comp_val = df_player["Comp"].iloc[0]

            score_display = f"{main_score_tile:.0f}" if main_score_tile is not None and not pd.isna(main_score_tile) else "n/a"
            band_display = str(main_band_tile).split(" ")[0] if main_band_tile else "n/a"
            min_display = f"{int(min_val):,}'" if min_val is not None and not pd.isna(min_val) else "n/a"
            comp_display = str(comp_val).replace("eng ", "").replace("es ", "").replace("de ", "").replace("it ", "").replace("fr ", "") if comp_val else "n/a"

            with col1:
                st.markdown(f'<div style="{_tile}"><div style="font-size:0.78rem;color:#6B7280;margin-bottom:0.2rem;">Score</div><div style="font-size:2rem;font-weight:800;color:{VALUE_COLOR};">{score_display}</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div style="{_tile}"><div style="font-size:0.78rem;color:#6B7280;margin-bottom:0.2rem;">Band</div><div style="font-size:1rem;font-weight:700;color:#F9FAFB;">{band_display}</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div style="{_tile}"><div style="font-size:0.78rem;color:#6B7280;margin-bottom:0.2rem;">Minutes</div><div style="font-size:1.4rem;font-weight:700;color:#F9FAFB;">{min_display}</div></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div style="{_tile}"><div style="font-size:0.78rem;color:#6B7280;margin-bottom:0.2rem;">League</div><div style="font-size:0.95rem;font-weight:700;color:#F9FAFB;">{comp_display}</div></div>', unsafe_allow_html=True)

        else:
            career_score, career_band = compute_career_main_score(df_player_all)
            total_90s = float(df_player_all["90s"].sum()) if "90s" in df_player_all.columns else 0.0
            n_seasons = int(df_player_all["Season"].nunique()) if "Season" in df_player_all.columns else 0

            score_display = f"{career_score:.0f}" if career_score is not None and not pd.isna(career_score) else "n/a"
            band_display = str(career_band).split(" ")[0] if career_band else "n/a"
            n90s_display = f"{total_90s:.0f}" if total_90s > 0 else "n/a"

            with col1:
                st.markdown(f'<div style="{_tile}"><div style="font-size:0.78rem;color:#6B7280;margin-bottom:0.2rem;">Career Score</div><div style="font-size:2rem;font-weight:800;color:{VALUE_COLOR};">{score_display}</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div style="{_tile}"><div style="font-size:0.78rem;color:#6B7280;margin-bottom:0.2rem;">Band</div><div style="font-size:1rem;font-weight:700;color:#F9FAFB;">{band_display}</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div style="{_tile}"><div style="font-size:0.78rem;color:#6B7280;margin-bottom:0.2rem;">Career 90s</div><div style="font-size:1.4rem;font-weight:700;color:#F9FAFB;">{n90s_display}</div></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div style="{_tile}"><div style="font-size:0.78rem;color:#6B7280;margin-bottom:0.2rem;">Seasons</div><div style="font-size:1.4rem;font-weight:700;color:#F9FAFB;">{n_seasons}</div></div>', unsafe_allow_html=True)

        # ===================== ROLE METRICS =====================
        st.markdown("### Performance profile vs. peers")

        if profile_view == "Per season":
            try:
                df_features_season = load_feature_table_for_season(season)
            except FileNotFoundError:
                st.info("No raw feature data found for this season.")
                df_features_season = pd.DataFrame()

            if not df_features_season.empty:
                if "Squad" in df_features_season.columns and player_squad is not None:
                    df_feat_player = df_features_season[
                        (df_features_season["Player"] == player) &
                        (df_features_season["Squad"] == player_squad)
                    ].copy()
                else:
                    df_feat_player = df_features_season[
                        df_features_season["Player"] == player
                    ].copy()
            else:
                df_feat_player = pd.DataFrame()

            if role is not None and not df_feat_player.empty:
                # Check if player is in a Big-5 league for the pizza chart
                player_comp = df_feat_player["Comp"].iloc[0] if "Comp" in df_feat_player.columns and not df_feat_player.empty else None
                if player_comp not in BIG5_COMPS:
                    st.info("ℹ️ The pizza chart is only available for Big-5 league players. The scatter plot is still shown.")

                col_pizza, col_scatter = st.columns(2)

                with col_pizza:
                    fig = render_pizza_chart(df_features_season, df_feat_player, role, season)
                    if fig is not None:
                        pizza_fig_for_pdf = fig
                        st.pyplot(fig, use_container_width=True)

                with col_scatter:
                    scatter_chart = render_role_scatter(df_features_season, df_feat_player, role)
                    if scatter_chart is not None:
                        st.altair_chart(scatter_chart, use_container_width=True)
                        scatter_df_all_pdf    = df_features_season
                        scatter_df_player_pdf = df_feat_player

            # ── Similar players ───────────────────────────────────────────
            with st.expander("Similar players — by playing style", expanded=True):
                try:
                    df_feat_sim = load_feature_table_for_season(season)
                    df_season_scores = df_all[df_all["Season"] == season].copy()
                    if "MainScore" not in df_season_scores.columns or "MainBand" not in df_season_scores.columns:
                        df_season_scores[["MainScore", "MainBand"]] = df_season_scores.apply(
                            get_primary_score_and_band, axis=1, result_type="expand"
                        )
                    sim = find_similar_players(
                        df_feat_sim, df_season_scores, player, role or "MF", n=5
                    )
                    if not sim.empty:
                        for _, r in sim.iterrows():
                            c1, c2, c3 = st.columns([0.5, 0.25, 0.25])
                            with c1:
                                if st.button(str(r["Player"]), key=f"sim_{r['Player']}_{season}"):
                                    st.session_state["pp_selected_player"] = r["Player"]
                                    st.session_state["pp_source"] = "global"
                                    st.rerun()
                            with c2:
                                st.caption(str(r.get("Squad", "")))
                            with c3:
                                score_val = r.get("MainScore")
                                st.caption(f"{int(score_val)}" if pd.notna(score_val) else "–")
                    else:
                        st.caption("Not enough data for this season/role.")
                except Exception:
                    st.caption("Similar players not available for this season.")

        else:
            if "Season" in df_player_all.columns:
                player_seasons = sorted(df_player_all["Season"].dropna().unique())
            else:
                player_seasons = []

            if not player_seasons or role is None:
                st.info("Not enough career data available to build a pizza chart for this player.")
            else:
                col_pizza, col_career_scatter = st.columns(2)
                with col_pizza:
                    fig = render_career_pizza_chart(player, role, player_seasons)
                    if fig is not None:
                        pizza_fig_for_pdf = fig
                        st.pyplot(fig, use_container_width=True)
                with col_career_scatter:
                    # Career scatter: show score distribution vs peers (all seasons)
                    if "Season" in df_player_all.columns:
                        df_career_feat_list = []
                        for s in player_seasons:
                            try:
                                df_s = load_feature_table_for_season(str(s))
                                if not df_s.empty:
                                    df_s["Season"] = s
                                    df_career_feat_list.append(df_s)
                            except Exception:
                                pass
                        if df_career_feat_list:
                            df_career_feat = pd.concat(df_career_feat_list, ignore_index=True)
                            df_career_player = df_career_feat[df_career_feat["Player"] == player].copy()
                            scatter_chart = render_role_scatter(df_career_feat, df_career_player, role)
                            if scatter_chart is not None:
                                st.altair_chart(scatter_chart, use_container_width=True)
                                scatter_df_all_pdf    = df_career_feat
                                scatter_df_player_pdf = df_career_player

        # ===================== PDF EXPORT =====================
        if card_row is not None and main_score is not None:
            try:
                # ── Peer percentile ───────────────────────────────────────────
                _peer_pct = None
                try:
                    _role_pdf = str(card_row.get("Pos", "") or "")
                    _score_col_map = {
                        "FW": "OffScore_abs", "Off_MF": "OffScore_abs",
                        "MF": "MidScore_abs",
                        "DF": "DefScore_abs", "Def_MF": "DefScore_abs",
                    }
                    _role_group = {
                        "FW": ["FW", "Off_MF"], "Off_MF": ["FW", "Off_MF"],
                        "MF": ["MF"],
                        "DF": ["DF", "Def_MF"], "Def_MF": ["DF", "Def_MF"],
                    }
                    _sc = _score_col_map.get(_role_pdf)
                    if _sc and _sc in df_all.columns:
                        _df_p = df_all.copy()
                        _season_pdf = card_row.get("Season")
                        if _season_pdf and "Season" in _df_p.columns:
                            _df_p = _df_p[_df_p["Season"] == _season_pdf]
                        _pos_filter = _role_group.get(_role_pdf, [_role_pdf])
                        if "Pos" in _df_p.columns:
                            _df_p = _df_p[_df_p["Pos"].isin(_pos_filter)]
                        _scores_pct = pd.to_numeric(_df_p[_sc], errors="coerce").dropna()
                        if len(_scores_pct) > 0:
                            _pct_val = float((_scores_pct < main_score).mean())
                            _peer_pct = {
                                "percentile": _pct_val,
                                "n_peers": int(len(_scores_pct)),
                                "role": _role_pdf,
                                "season": str(_season_pdf) if _season_pdf else None,
                            }
                except Exception:
                    pass

                _png_for_pdf = generate_player_card_png(card_row, main_score, main_band, fmt_market_value)
                _pdf_bytes = generate_player_report_pdf(
                    card_row, main_score, main_band,
                    card_png_bytes=_png_for_pdf,
                    pizza_fig=pizza_fig_for_pdf,
                    scatter_df_all=scatter_df_all_pdf,
                    scatter_df_player=scatter_df_player_pdf,
                    peer_percentile=_peer_pct,
                    fmt_market_value_fn=fmt_market_value,
                )
                _season_label = str(card_row.get("Season", "career") or "career")
                st.download_button(
                    "📄 Export PDF Report",
                    data=_pdf_bytes,
                    file_name=f"{player.replace(' ', '_')}_{_season_label}_report.pdf",
                    mime="application/pdf",
                    key="dl_player_report_pdf",
                )
            except Exception:
                pass

        # ===================== SCORE TREND (Career) =====================
        if profile_view == "Career":
            st.markdown("### Career Score Trend")

            if "MainScore" not in df_player_all.columns or "MainBand" not in df_player_all.columns:
                df_player_all = df_player_all.copy()
                df_player_all[["MainScore", "MainBand"]] = df_player_all.apply(
                    get_primary_score_and_band,
                    axis=1,
                    result_type="expand",
                )

            score_col = "MainScore"

            plot_df = (
                df_player_all[["Season", "Squad", score_col]]
                .dropna(subset=[score_col])
                .sort_values("Season")
            )

            y_enc = alt.Y(
                f"{score_col}:Q",
                title="Score",
                scale=alt.Scale(domain=[0, 1100]),
                axis=alt.Axis(values=[0, 500, 1000]),
            )

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
                    "x": [last_season] * 4,
                }
            )

            band_lines = (
                alt.Chart(band_data)
                .mark_rule(
                    strokeDash=[4, 4],
                    strokeWidth=0.6,
                    opacity=0.4,
                    color="#6b7280",
                )
                .encode(
                    y="y:Q",
                )
            )

            band_labels = (
                alt.Chart(band_data)
                .mark_text(
                    align="left",
                    baseline="middle",
                    dx=30,
                    color="#e5e7eb",
                    fontSize=9,
                )
                .encode(
                    x="x:O",
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

            # ── Age curve ─────────────────────────────────────────────────
            render_age_curve(df_all, player, role or "MF", get_primary_score_and_band, VALUE_COLOR)

            # ===================== MARKET VALUE HISTORY (Career) =====================
            tm_id = None
            if "tm_player_id" in df_all.columns:
                ids = df_all[df_all["Player"] == player]["tm_player_id"].dropna()
                if not ids.empty:
                    tm_id = ids.iloc[0]

            if (
                tm_id is not None
                and not df_valuations.empty
                and "player_id" in df_valuations.columns
            ):
                df_pv = df_valuations[df_valuations["player_id"] == tm_id].copy()
                if not df_pv.empty and "date" in df_pv.columns and "market_value_in_eur" in df_pv.columns:
                    df_pv["date"] = pd.to_datetime(df_pv["date"], errors="coerce")
                    df_pv = df_pv.dropna(subset=["date", "market_value_in_eur"])
                    df_pv["MarketValue_M"] = df_pv["market_value_in_eur"] / 1_000_000
                    df_pv = df_pv.sort_values("date")

                    if not df_pv.empty:
                        st.markdown("### Market Value History")

                        val_chart = (
                            alt.Chart(df_pv)
                            .mark_line(strokeWidth=2, color=VALUE_COLOR)
                            .encode(
                                x=alt.X("date:T", title="Date"),
                                y=alt.Y("MarketValue_M:Q", title="Market Value (€M)"),
                                tooltip=[
                                    alt.Tooltip("date:T", title="Date", format="%b %Y"),
                                    alt.Tooltip("MarketValue_M:Q", title="Value (€M)", format=".1f"),
                                ],
                            )
                            .properties(height=220)
                            .configure_axis(
                                grid=False, domain=True,
                                labelColor="#e5e7eb", titleColor="#e5e7eb",
                            )
                            .configure_view(strokeWidth=0)
                        )
                        st.altair_chart(val_chart, use_container_width=True)

        return

    # ==================================================================
    # MODE: Player Rankings
    # ==================================================================
    if mode == "Player Rankings":
        st.sidebar.subheader("Ranking filters")

        seasons = sorted(df_all["Season"].dropna().unique())
        default_season_idx = len(seasons) - 1 if seasons else 0
        season = st.sidebar.selectbox(
            "Season",
            seasons,
            index=default_season_idx,
            key="toplists_season",
        )

        df_view = df_all[df_all["Season"] == season].copy()

        # League filter
        if "Comp" in df_view.columns:
            nations = sorted(df_view["Comp"].dropna().unique())
            nation_options = ["All"] + nations

            if "toplists_nation" not in st.session_state:
                st.session_state["toplists_nation"] = "All"

            if st.session_state["toplists_nation"] not in nation_options:
                st.session_state["toplists_nation"] = "All"

            nation_sel = st.sidebar.selectbox(
                "League",
                nation_options,
                index=nation_options.index(st.session_state["toplists_nation"]),
                key="toplists_nation",
            )

            if nation_sel != "All":
                df_view = df_view[df_view["Comp"] == nation_sel]

        # Club filter
        if "Squad" in df_view.columns:
            clubs = sorted(df_view["Squad"].dropna().unique())
            club_options = ["All"] + clubs

            if "toplists_club" not in st.session_state:
                st.session_state["toplists_club"] = "All"

            if st.session_state["toplists_club"] not in club_options:
                st.session_state["toplists_club"] = "All"

            club_sel = st.sidebar.selectbox(
                "Club",
                club_options,
                index=club_options.index(st.session_state["toplists_club"]),
                key="toplists_club",
            )

            if club_sel != "All":
                df_view = df_view[df_view["Squad"] == club_sel]

        # Position filter
        if "Pos" in df_view.columns:
            pos_values = sorted(df_view["Pos"].dropna().unique())
            selected_positions = st.sidebar.multiselect(
                "Positions",
                options=pos_values,
                default=pos_values,
                key="toplists_positions",
            )
            if selected_positions:
                df_view = df_view[df_view["Pos"].isin(selected_positions)]

        top_n = st.sidebar.slider(
            "Top N players",
            3,
            50,
            10,
            1,
            key="top_topn",
        )

        has_vfm_pre = "MarketValue_EUR" in df_view.columns
        sort_by_options_pre = ["Score"]
        if has_vfm_pre:
            sort_by_options_pre.append("Value for money (Score/€M)")

        sort_by = st.sidebar.radio(
            "Sort by",
            sort_by_options_pre,
            index=0,
            key="top_sort_mode",
        )

        # Advanced filters in expander
        with st.sidebar.expander("Advanced filters"):
            min_90s = st.slider(
                "Minimum 90s played",
                1.0,
                40.0,
                5.0,
                0.5,
                key="top_min_90s",
            )
            if "90s" in df_view.columns:
                df_view = df_view[df_view["90s"] >= min_90s]

            if "Age" in df_view.columns:
                age_numeric = pd.to_numeric(df_view["Age"], errors="coerce")
                if age_numeric.notna().any():
                    min_age = int(age_numeric.min())
                    max_age = int(age_numeric.max())
                    age_range = st.slider(
                        "Age range",
                        min_value=min_age,
                        max_value=max_age,
                        value=(min_age, max_age),
                        step=1,
                        key="top_age_range",
                    )
                    df_view = df_view[
                        (age_numeric >= age_range[0]) & (age_numeric <= age_range[1])
                    ]

            sort_mode = st.radio(
                "Sort order",
                ("Highest first", "Lowest first"),
                index=0,
                key="top_sort_order",
            )

        ascending = sort_mode == "Lowest first"

        if df_view.empty:
            st.warning("No players found for the selected filters.")
            return

        df_view[["MainScore", "MainBand"]] = df_view.apply(
            get_primary_score_and_band,
            axis=1,
            result_type="expand",
        )

        df_view = df_view[df_view["MainScore"].notna()].copy()

        # Value for money: Score / MarketValue (€M)
        if "MarketValue_EUR" in df_view.columns:
            mv_m = pd.to_numeric(df_view["MarketValue_EUR"], errors="coerce") / 1_000_000
            df_view["ValueForMoney"] = df_view["MainScore"] / mv_m.replace(0, np.nan)

        if df_view.empty:
            st.warning("No players with a primary score found for the selected filters.")
            return

        if sort_by == "Value for money (Score/€M)":
            sort_col = "ValueForMoney"
            metric_label = "Score per €1M"
        else:
            sort_col = "MainScore"
            metric_label = "Primary role score"

        df_top = (
            df_view
            .sort_values(sort_col, ascending=ascending)
            .head(top_n)
            .copy()
        )

        chart_main = render_toplist_bar(
            df=df_top,
            metric_col=sort_col,
            metric_label=metric_label,
            base_title=f"Top {len(df_top)} Players",
            top_n=len(df_top),
            season=season,
            ascending=ascending,
        )

        # Selectable table — FIRST, most prominent navigation element
        st.markdown("#### 👇 Click a row to open the player profile")
        show_cols = [c for c in ["Player", "Squad", "Comp", "Pos", "Age", "MainScore", "MainBand", "MarketValue_EUR"] if c in df_top.columns]
        df_show = df_top[show_cols].copy()
        if "MainScore" in df_show.columns:
            df_show["MainScore"] = df_show["MainScore"].round(0).astype("Int64")
        if "MarketValue_EUR" in df_show.columns:
            df_show["MarketValue_EUR"] = df_show["MarketValue_EUR"].apply(
                lambda v: fmt_market_value(v) if pd.notna(v) else "—"
            )
            df_show = df_show.rename(columns={"MarketValue_EUR": "Market Value"})

        sel_event = st.dataframe(
            df_show,
            selection_mode="single-row",
            on_select="rerun",
            key="rankings_df_select",
            use_container_width=True,
            hide_index=True,
        )
        if sel_event.selection.rows:
            idx = sel_event.selection.rows[0]
            st.session_state["_nav_to_player"] = df_top.iloc[idx]["Player"]
            st.rerun()

        if chart_main is not None:
            bar_event = st.plotly_chart(
                chart_main,
                use_container_width=True,
                on_select="rerun",
                key="rankings_bar_chart",
            )
            try:
                pts = bar_event.selection.points
                if pts:
                    cd = pts[0].get("customdata")
                    player = cd[0] if isinstance(cd, (list, tuple)) and cd else (cd if isinstance(cd, str) else None)
                    if player:
                        st.session_state["_nav_to_player"] = str(player)
                        st.rerun()
            except Exception:
                pass

        export_base = ["Player", "Squad", "Comp", "Pos", "Age", "Min", "90s", "MainScore", "MainBand"]
        if sort_by == "Value for money (Score/€M)":
            export_base += ["MarketValue_EUR", "ValueForMoney"]
        export_cols = [c for c in export_base if c in df_top.columns]
        csv_bytes = df_top[export_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download top list as CSV",
            data=csv_bytes,
            file_name=f"playerscore_top{len(df_top)}_{season}.csv",
            mime="text/csv",
        )

        # Beeswarm as Plotly — supports text labels AND click selection
        beeswarm_fig = render_score_age_beeswarm(
            df_all_filtered=df_view,
            df_top=df_top,
        )

        if beeswarm_fig is not None:
            beeswarm_event = st.plotly_chart(
                beeswarm_fig,
                use_container_width=True,
                on_select="rerun",
                key="rankings_beeswarm_chart",
            )
            st.caption("Hollow circles = all players · Filled teal + name = top N · Click any dot to open profile")
            try:
                pts = beeswarm_event.selection.points
                if pts:
                    cd = pts[0].get("customdata")
                    if isinstance(cd, (list, tuple)) and cd:
                        player = cd[0]
                    elif isinstance(cd, str):
                        player = cd
                    else:
                        player = None
                    if player:
                        st.session_state["_nav_to_player"] = str(player)
                        st.rerun()
            except Exception:
                pass

        band_hist = render_band_histogram(df_view, season=season)
        if band_hist is not None:
            st.altair_chart(band_hist, use_container_width=True)

        return

    # ==================================================================
    # MODE: TEAM SCORES
    # ==================================================================
    elif mode == "Team scores":

        # ------------------------------------------------------------------
        # Daten laden (robust gegen alte / neue load_data-Versionen)
        # ------------------------------------------------------------------
        try:
            data_version = get_data_version()
            df_all, df_agg, df_squad, df_big5, _df_val = load_data(data_version)
        except Exception:
            df_all = df_agg = df_squad = pd.DataFrame()
            df_big5 = pd.DataFrame()

        # ------------------------------------------------------------------
        # Guardrails
        # ------------------------------------------------------------------
        if df_squad is None or df_squad.empty:
            st.info("No squad score data available. Run the pipeline first.")
            return

        # ------------------------------------------------------------------
        # View rendern
        # ------------------------------------------------------------------
        render_team_scores_view(
            df_all=df_all,
            df_squad=df_squad,
            df_big5=df_big5,
        )

    # ==================================================================
    # MODE: HIDDEN GEMS
    # ==================================================================
    elif mode == "Hidden Gems":
        render_hidden_gems(df_all, df_valuations)

    # ==================================================================
    # MODE: COMPARE PLAYERS
    # ==================================================================
    elif mode == "Compare Players":
        render_player_comparison(df_all, df_valuations)


if __name__ == "__main__":
    main()