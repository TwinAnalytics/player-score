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

import matplotlib
matplotlib.use("Agg")

# -------------------------------------------------------------------
# Global Matplotlib + Theme
# -------------------------------------------------------------------
plt.rcParams["figure.dpi"] = 200      # höhere Render-Auflösung
plt.rcParams["savefig.dpi"] = 200

PRIMARY_COLOR = "#1f77b4"  # main brand color
APP_BG = "#000000"
GRID_COLOR = "#374151"
TEXT_COLOR = "#e5e7eb"
SLICE_COLOR = "cornflowerblue"
VALUE_COLOR = "#00B8A9"  #f57f17

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

# ---------- kleine Helper statt externem utils-Modul ----------

# ---------- kleine Helper statt externem utils-Modul ----------

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
def load_data():
    try:
        from src.multi_season import load_all_seasons, aggregate_player_scores
        from src.squad import compute_squad_scores
    except Exception as e:
        st.error("Error importing data loading functions (src.multi_season / src.squad).")
        st.exception(e)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    try:
        root = Path(__file__).resolve().parent
        processed_dir = root / "Data" / "Processed"

        df_all = load_all_seasons(processed_dir)
        df_agg = aggregate_player_scores(df_all)
        df_squad = compute_squad_scores(df_all)

        # ---------- BIG5 TABLE ----------
        df_big5 = pd.DataFrame()
        big5_path = processed_dir / "big5_table_all_seasons.csv"
        if big5_path.exists():
            df_big5 = pd.read_csv(big5_path)

            # Country -> Comp (matcht deine df_all Comp Werte)
            country2comp = {
                "eng": "eng Premier League",
                "es":  "es La Liga",
                "fr":  "fr Ligue 1",
                "it":  "it Serie A",
                "de":  "de Bundesliga",
            }
            if "Country" in df_big5.columns and "Comp" not in df_big5.columns:
                df_big5["Comp"] = (
                    df_big5["Country"].astype(str).str.split().str[0].map(country2comp)
                )

            # Keys normalisieren (gegen Whitespace)
            for c in ("Season", "Squad", "Comp"):
                if c in df_big5.columns:
                    df_big5[c] = df_big5[c].astype(str).str.strip()
                if c in df_all.columns:
                    df_all[c] = df_all[c].astype(str).str.strip()
                if c in df_squad.columns:
                    df_squad[c] = df_squad[c].astype(str).str.strip()

            # Nur sinnvolle Team-Kontext-Spalten mergen (du kannst erweitern)
            big5_cols = [c for c in [
                "Season","Squad","Comp","LgRk","Pts","Pts/MP","xGD/90","xG","xGA","xGD","Attendance"
            ] if c in df_big5.columns]

            df_all = df_all.merge(df_big5[big5_cols], on=["Season","Squad","Comp"], how="left")
            if not df_squad.empty and all(c in df_squad.columns for c in ["Season","Squad","Comp"]):
                df_squad = df_squad.merge(df_big5[big5_cols], on=["Season","Squad","Comp"], how="left")

        return df_all, df_agg, df_squad, df_big5

    except Exception as e:
        st.error("Error loading processed score files from Data/Processed.")
        st.exception(e)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

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
    Wählt den passenden Score + Band basierend auf der Roh-Position.
    - FW und Off_MF -> OffScore_abs / OffBand
    - MF           -> MidScore_abs / MidBand
    - DF und Def_MF -> DefScore_abs / DefBand
    """
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
    if metric_col not in df.columns:
        st.info(f"Keine Spalte '{metric_col}' gefunden.")
        return None

    base_cols = ["Player", "Squad", metric_col, "Pos", "Comp"]
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
    df_plot["Player_order"] = df_plot["Player"]
    y_order = df_plot["Player_order"].tolist()

    max_val = float(df_plot[metric_col].max())
    x_scale = alt.Scale(domain=(0, max_val * 1.05))

    pos_label = None
    comp_label = None
    squad_label = None

    if "Pos" in df_plot.columns:
        pos_vals = sorted(df_plot["Pos"].dropna().unique())
        if len(pos_vals) == 1:
            pos_map = {
                "FW": "Forwards",
                "Off_MF": "Offensive midfielders",
                "MF": "Midfielders",
                "DF": "Defenders",
                "Def_MF": "Defensive midfielders",
            }
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
    if pos_label is not None:
        parts.append(pos_label)
    if squad_label is not None:
        parts.append(squad_label)
    elif comp_label is not None:
        parts.append(comp_label)

    if not parts:
        context = "All Big-5 leagues"
    else:
        context = " – ".join(parts)

    full_title = f"{base_title} – {context}"
    if season is not None:
        full_title += f" (Season {season})"

    bars = (
        alt.Chart(df_plot)
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
        .encode(
            x=alt.X(
                f"{metric_col}:Q",
                scale=x_scale,
                axis=None,
            ),
            y=alt.Y(
                "Player_order:N",
                sort=y_order,
                axis=alt.Axis(
                    labels=False,
                    ticks=False,
                    title=None,
                    domain=False,
                    grid=False,
                ),
            ),
            tooltip=[
                "Player",
                "Squad",
                alt.Tooltip(f"{metric_col}:Q", title=metric_label, format=".0f"),
            ],
            color=alt.value(VALUE_COLOR),
        )
    )

    score_text = (
        alt.Chart(df_plot)
        .mark_text(
            align="right",
            baseline="middle",
            dx=-6,
            fontSize=15,
            color="#0f172a",
            fontWeight="bold",
        )
        .encode(
            x=alt.X(f"{metric_col}:Q", scale=x_scale),
            y=alt.Y("Player_order:N", sort=y_order),
            text=alt.Text(f"{metric_col}:Q", format=".0f"),
        )
    )

    name_text = (
        alt.Chart(df_plot)
        .mark_text(
            align="left",
            baseline="middle",
            dx=6,
            fontSize=12,
            color="#E5E7EB",
        )
        .encode(
            x=alt.X(f"{metric_col}:Q", scale=x_scale),
            y=alt.Y("Player_order:N", sort=y_order),
            text="Player",
        )
    )

    chart = (
        (bars + score_text + name_text)
        .properties(
            height=26 * len(df_plot) + 20,
            title=full_title,
        )
        .configure_axis(
            labelColor="#E5E7EB",
            titleColor="#E5E7EB",
        )
        .configure_title(
            color="#E5E7EB",
            fontSize=20,
            anchor="start",
        )
        .configure_view(strokeWidth=0)
    )

    return chart

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

    score_domain = (0, 1000)
    age_min = float(df_plot["Age_num"].min())
    age_max = float(df_plot["Age_num"].max())
    age_domain = (max(15, age_min - 1), age_max + 1)

    base = alt.Chart(df_plot)

    peers = (
        base.transform_filter(alt.datum.is_top == False)
        .mark_circle(
            size=40,
            opacity=1.0,
            fillOpacity=0,
            stroke="#F9FAFB",
            strokeWidth=1.0,
        )
        .encode(
            x=alt.X(
                "MainScore:Q",
                title="Score",
                scale=alt.Scale(domain=score_domain),
                axis=alt.Axis(format=".0f"),
            ),
            y=alt.Y(
                "Age_jitter:Q",
                title="Age",
                scale=alt.Scale(domain=age_domain),
            ),
            tooltip=[
                "Player",
                "Squad",
                "Pos",
                alt.Tooltip("Age_num:Q", title="Age"),
                alt.Tooltip("MainScore:Q", title="Score", format=".0f"),
                alt.Tooltip("MinutesFactor:Q", title="90s played", format=".1f"),
            ],
        )
    )

    tops = (
        base.transform_filter(alt.datum.is_top == True)
        .mark_circle(
            size=200,
            opacity=1.0,
            fillOpacity=1.0,
            stroke="#F9FAFB",
            strokeWidth=1.2,
        )
        .encode(
            x=alt.X(
                "MainScore:Q",
                scale=alt.Scale(domain=score_domain),
                axis=alt.Axis(format=".0f"),
            ),
            y=alt.Y(
                "Age_jitter:Q",
                scale=alt.Scale(domain=age_domain),
            ),
            color=alt.value(VALUE_COLOR),
            tooltip=[
                "Player",
                "Squad",
                "Pos",
                alt.Tooltip("Age_num:Q", title="Age"),
                alt.Tooltip("MainScore:Q", title="Score", format=".0f"),
                alt.Tooltip("MinutesFactor:Q", title="90s played", format=".1f"),
            ],
        )
    )

    labels = (
        base.transform_filter(alt.datum.is_top == True)
        .mark_text(
            align="left",
            baseline="middle",
            dx=15,
            dy=-2,
            fontSize=10,
            color="#E5E7EB",
        )
        .encode(
            x=alt.X(
                "MainScore:Q",
                scale=alt.Scale(domain=score_domain),
            ),
            y=alt.Y(
                "Age_jitter:Q",
                scale=alt.Scale(domain=age_domain),
            ),
            text="Player:N",
        )
    )

    chart = (
        (peers + tops + labels)
        .properties(
            height=450,
            title="Score vs Age",
        )
        .configure_axis(
            grid=True,
            gridOpacity=0.1,
            gridColor="#4b5563",
            labelColor="#E5E7EB",
            titleColor="#E5E7EB",
        )
        .configure_title(
            color="#E5E7EB",
            fontSize=20,
            anchor="start",
        )
        .configure_view(strokeWidth=0)
    )

    return chart

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

    try:
        overall_raw = float(row.get(primary_score_col, 0.0))
        overall = int(round(overall_raw))
    except Exception:
        overall = 0

    band_label = str(row.get(band_col, "") or "")
    comp = str(row.get("Comp", ""))

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
    </style>

    <div class="fifa-card-container">
      <div class="fifa-card">
        <div class="fifa-card-top-row">
          <div class="fifa-card-overall">
            <div class="fifa-card-overall-value">{overall}</div>
            <div class="fifa-card-pos">{pos_display}</div>
            <div class="fifa-card-band-pill">{band_label}</div>
          </div>
          <div class="fifa-card-meta">
            {"<div>Age: " + str(int(age)) + "</div>" if age is not None else ""}
            {"<div>" + comp + "</div>" if comp else ""}
          </div>
        </div>

        <div class="fifa-card-player-name">{player_name}</div>
        <div class="fifa-card-club">{club}</div>

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
        # erste Fläche gleich „single team“, zweite leicht transparenter
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

    # Default metric (sorting)
    metric_col = "OverallScore_squad"
    metric_name = "Squad Score"

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
        .sort_values(metric_col, ascending=False)
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

    # Existing extras
    if "Age_squad_mean" in df_rank.columns:
        df_rank["Age (avg)"] = pd.to_numeric(df_rank["Age_squad_mean"], errors="coerce").round(1)

    if "Min_squad" in df_rank.columns:
        df_rank["Minutes (total)"] = pd.to_numeric(df_rank["Min_squad"], errors="coerce").round(0).astype("Int64")

    if "NumPlayers_squad" in df_rank.columns:
        df_rank["Players"] = pd.to_numeric(df_rank["NumPlayers_squad"], errors="coerce").astype("Int64")

    if "Pts" in df_rank.columns:
        df_rank["Pts"] = pd.to_numeric(df_rank["Pts"], errors="coerce").round(0).astype("Int64")

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

    st.dataframe(df_rank[cols_show], use_container_width=True, hide_index=True)

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
        <div style="font-size:0.8rem; opacity:0.8; color:#E5E7EB;">League rank</div>
        <div style="font-size:1.6rem; font-weight:600; color:{VALUE_COLOR};">
          {fmt_int(rank_val)}
        </div>
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
        color="black",
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
    color_range = [VALUE_COLOR, "#61abd2", "#214642", "#ffffff"]

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

def merge_big5_into_squad(
    df_squad: pd.DataFrame,
    df_big5: pd.DataFrame,
) -> pd.DataFrame:
    if df_squad.empty or df_big5.empty:
        return df_squad

    needed = {"Season", "Squad"}
    if not needed.issubset(df_squad.columns) or not needed.issubset(df_big5.columns):
        return df_squad

    keep_cols = [
        "Season",
        "Squad",
        "LgRk",
        "MP",
        "Pts",
        "Pts/MP",
        "GD",
        "xGD",
        "xGD/90",
    ]
    keep_cols = [c for c in keep_cols if c in df_big5.columns]

    big5_small = df_big5[keep_cols].copy()

    # numeric safety
    for c in keep_cols:
        if c not in ("Season", "Squad"):
            big5_small[c] = pd.to_numeric(big5_small[c], errors="coerce")

    df_out = df_squad.merge(
        big5_small,
        on=["Season", "Squad"],
        how="left",
    )

    return df_out

# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="PlayerScore",
        layout="wide",
        page_icon="⚽",
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

    df_all, df_agg, df_squad, df_big5 = load_data()

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

    st.sidebar.header("View")
    mode = st.sidebar.radio(
        "Select mode",
        ["Home", "Player profile", "Top lists", "Team scores"],
    )

    if mode in ("Player profile", "Top lists", "Team scores"):
        st.markdown(
            "Score scale (0–1000): 🟣 Exceptional ≥ 900  ·  🟢 World Class ≥ 750  ·  🔵 Top Starter ≥ 400  ·  🟡 Solid Squad Player ≥ 200  ·  ⚪️ Below Big-5 Level < 200"
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
        st.markdown(
            f"""
            <h1 style="color:{VALUE_COLOR}; margin-bottom:0.25rem;">PlayerScore</h1>
            <p style="font-size:1.05rem; margin-top:0;">
                Advanced football player analytics across leagues and seasons
            </p>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
             f"""
             <h3 style="color:{VALUE_COLOR};">What is PlayerScore?</h3>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            PlayerScore is built on a self-developed, data-driven scoring framework that makes it possible 
            to compare football players across different leagues — regardless of country, competition level, 
            or data availability.

            The analysis engine continuously processes the latest publicly available performance data from FBref 
            and combines it with a custom scoring logic that uses comprehensive Big-5 metrics.
            """
        )

        st.markdown(
             f"""
             <h3 style="color:{VALUE_COLOR};">Why does PlayerScore matter?</h3>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
           Modern football generates more data than ever — but turning that data into actionable insight remains difficult. Raw stats alone rarely answer crucial questions such as:

            •	Is this player truly performing above league average?
            •	How does he compare to similar profiles in other countries?
            •	Would he be an upgrade for our current squad?
            •	How stable is his performance across seasons?

            PlayerScore bridges this gap by transforming scattered match statistics into a unified analytical model that reveals the actual impact and consistency of a player.
            """
        )

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
             f"""
            <h3 style="color:{VALUE_COLOR};">How PlayerScore works?</h3>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            The engine behind PlayerScore integrates:

            🔹 Per-90 standardization
            Normalizes all performance metrics, enabling fair comparisons regardless of playing time.

            🔹 Role-specific scoring frameworks
            Attackers, midfielders and defenders are evaluated using metrics that truly matter for their respective roles.

            🔹 Multi-season perspective
            Performance is measured across seasons, offering deeper insight into development, peaks, and declines.

            🔹 Squad-level analytics
            Beyond individuals, PlayerScore assesses entire squads to identify strengths, weaknesses, and dependencies on top performers.

            🔹 Cross-league comparability
            Scores are calibrated to allow interpretation across league boundaries — essential for scouting and recruitment.  
            """
        )

        st.markdown(
             f"""
            <h3 style="color:{VALUE_COLOR};">What can you do with PlayerScore?</h3>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            With PlayerScore, you can:

            📊 Explore interactive player profiles with role-based radars, trendlines and detailed metrics

            📈 Track career trajectories and instantly spot breakthroughs or regressions

            🔎 Identify squad leaders with the “Impact Share” model, showing who truly drives team performance

            ⚖️ Compare players across leagues using a consistent scoring scale

            🏗 Evaluate team quality with squad-level scores and historical development charts

            🛠 Build data-driven scouting shortlists and find undervalued profiles

            Whether you’re doing recruitment, tactical analysis, data scouting or performance tracking — PlayerScore gives you an intuitive, modern and transparent framework to understand football through data.  
            """
        )

        st.markdown(
             f"""
            <h3 style="color:{VALUE_COLOR};">A living system — continuously improving</h3>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            The PlayerScore database already contains several thousand players from Europe’s top-5 leagues and multiple historical seasons — and it grows with every pipeline run.

            Upcoming developments include:

            📊 Advanced clustering & similarity models
            🌍 Broader multi-league data coverage
            ⚽ Expanded radar profiles for team and player roles  
            """
        )

        st.info(
            "Use the sidebar to switch to **Player profile** to explore an individual player, "
            "or to **Top lists** to see ranked players by role and season."
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
            "Player (global search)",
            options=global_options,
            index=global_options.index(global_default),
            key="pp_global_player_selectbox",
            on_change=_on_global_change,
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Search by League & Club**")

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
            st.info("Select a player via global search or via current league/club filters.")
            return

        st.session_state["selected_player_label"] = player

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
            if details:
                st.caption(details)
        else:
            st.caption(get_role_label(typical_pos))

        role = typical_pos or (df_player["Pos"].iloc[0] if not df_player.empty and "Pos" in df_player.columns else None)

        if profile_view == "Per season" and season is not None:
            st.markdown(f"Season: {season}")
        else:
            if "Season" in df_player_all.columns and not df_player_all.empty:
                first_season = df_player_all["Season"].min()
                last_season = df_player_all["Season"].max()
                st.markdown(f"Seasons: {first_season} – {last_season}")
            else:
                st.markdown("Seasons: n/a")

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
        elif card_row is not None:
            st.info("No primary role score available for this player in the selected view.")

        # ===================== SUMMARY =====================
        st.markdown("### Summary")

        col1, col2, col3, col4 = st.columns(4)

        if profile_view == "Per season":
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
            if "Season" in df_player_all.columns and "Age" in df_player_all.columns:
                df_sorted = df_player_all.sort_values("Season")
                age_career = df_sorted["Age"].iloc[-1]
            else:
                age_career = None

            total_90s = float(df_player_all["90s"].sum()) if "90s" in df_player_all.columns else 0.0

            career_score, career_band = compute_career_main_score(df_player_all)
            avg_score = career_score
            avg_band_label = career_band

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
                col_pizza, col_scatter = st.columns(2)

                with col_pizza:
                    fig = render_pizza_chart(df_features_season, df_feat_player, role, season)
                    if fig is not None:
                        st.pyplot(fig, use_container_width=True)

                with col_scatter:
                    scatter_chart = render_role_scatter(df_features_season, df_feat_player, role)
                    if scatter_chart is not None:
                        st.altair_chart(scatter_chart, use_container_width=True)

        else:
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

        return

    # ==================================================================
    # MODE: TOP LISTS
    # ==================================================================
    if mode == "Top lists":
        st.sidebar.subheader("Top list filters")

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
            st.sidebar.markdown("**Positions**")

            selected_positions = []
            for pos in pos_values:
                checked = st.sidebar.checkbox(
                    pos,
                    value=True,
                    key=f"top_pos_{pos}",
                )
                if checked:
                    selected_positions.append(pos)

            if selected_positions:
                df_view = df_view[df_view["Pos"].isin(selected_positions)]

        # Minutes filter
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

        # Age filter
        if "Age" in df_view.columns:
            st.sidebar.markdown("**Age filter**")
            use_age_filter = st.sidebar.checkbox(
                "Enable Age Filter",
                value=False,
                key="top_use_age_filter",
            )

            if use_age_filter:
                age_numeric = pd.to_numeric(df_view["Age"], errors="coerce")
                if age_numeric.notna().any():
                    min_age = int(age_numeric.min())
                    max_age = int(age_numeric.max())

                    if "top_age_max" not in st.session_state:
                        st.session_state["top_age_max"] = min(20, max_age)
                    else:
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

        if df_view.empty:
            st.warning("No players found for the selected filters.")
            return

        df_view[["MainScore", "MainBand"]] = df_view.apply(
            get_primary_score_and_band,
            axis=1,
            result_type="expand",
        )

        df_view = df_view[df_view["MainScore"].notna()].copy()

        if df_view.empty:
            st.warning("No players with a primary score found for the selected filters.")
            return

        top_n = st.sidebar.slider(
            "Top N players",
            3,
            50,
            5,
            1,
            key="top_topn",
        )

        sort_mode = st.sidebar.radio(
            "Sort order",
            ("Highest score first", "Lowest score first"),
            index=0,
            key="top_sort_order",
        )
        ascending = sort_mode == "Lowest score first"

        df_top = (
            df_view
            .sort_values("MainScore", ascending=ascending)
            .head(top_n)
            .copy()
        )

        chart_main = render_toplist_bar(
            df=df_top,
            metric_col="MainScore",
            metric_label="Primary role score",
            base_title=f"Top {len(df_top)} Players",
            top_n=len(df_top),
            season=season,
            ascending=ascending,
        )

        if chart_main is not None:
            st.altair_chart(chart_main, use_container_width=True)

        beeswarm_chart = render_score_age_beeswarm(
            df_all_filtered=df_view,
            df_top=df_top,
        )

        if beeswarm_chart is not None:
            st.altair_chart(beeswarm_chart, use_container_width=True)

        band_hist = render_band_histogram(df_view, season=season)
        if band_hist is not None:
            st.altair_chart(band_hist, use_container_width=False)

        return

    # ==================================================================
    # MODE: TEAM SCORES
    # ==================================================================

    elif mode == "Team scores":

        # ------------------------------------------------------------------
        # Daten laden (robust gegen alte / neue load_data-Versionen)
        # ------------------------------------------------------------------
        try:
            # neue Version (mit Big5)
            df_all, df_agg, df_squad, df_big5 = load_data()
        except ValueError:
            # Fallback: alte Version ohne Big5
            df_all, df_agg, df_squad = load_data()
            df_big5 = pd.DataFrame()


        if df_squad is None or df_squad.empty:
            st.info("No squad score data available. Run the pipeline first.")
            return
        
        render_team_scores_view(
            df_all=df_all,
            df_squad=df_squad,
            df_big5=df_big5,
        )

if __name__ == "__main__":
    main()