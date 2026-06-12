"""
Generate three LinkedIn visuals:
  1. Scatter — Team Score vs Market Value (Big-5, 2024-2025)
  2. Bar chart — Efficient vs Wasteful squads (residual from trend)
  3. Bar chart — Squad Efficiency Ranking (Score per €100M)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "Data" / "Processed"
OUT_DIR   = Path(__file__).resolve().parent / "visuals"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Style constants  (matches app dark theme)
# ---------------------------------------------------------------------------
BG        = "#0D0D0D"
SURFACE   = "#161616"
TEXT      = "#E5E7EB"
MUTED     = "#6B7280"
GRID      = "#1F2937"
GREEN     = "#22C55E"
RED       = "#EF4444"
TRENDLINE = "#94A3B8"

LEAGUE_COLORS = {
    "eng Premier League": "#38BDF8",   # sky blue
    "es La Liga":         "#FACC15",   # yellow
    "de Bundesliga":      "#FB923C",   # orange
    "it Serie A":         "#4ADE80",   # green
    "fr Ligue 1":         "#C084FC",   # purple
}
LEAGUE_SHORT = {
    "eng Premier League": "Premier League",
    "es La Liga":         "La Liga",
    "de Bundesliga":      "Bundesliga",
    "it Serie A":         "Serie A",
    "fr Ligue 1":         "Ligue 1",
}

SEASON    = "2024-2025"
FOOTNOTE  = "PlayerScore by Stefan Hofmann  •  Data: FBref & Transfermarkt  •  Season 2024-2025"

# ---------------------------------------------------------------------------
# Load & merge data
# ---------------------------------------------------------------------------
sq = pd.read_csv(DATA_DIR / "squad_scores_all_seasons.csv")
mv = pd.read_csv(DATA_DIR / "player_market_values.csv")

latest   = sq[sq["Season"] == SEASON].copy()
team_mv  = mv.groupby("Squad")["MarketValue_EUR"].sum().reset_index()
df       = latest.merge(team_mv, on="Squad", how="inner")
df["MarketValue_M"]  = df["MarketValue_EUR"] / 1e6
df["log_mv"]         = np.log10(df["MarketValue_M"])
df["Score"]          = pd.to_numeric(df["OverallScore_squad"], errors="coerce")
df = df.dropna(subset=["Score", "MarketValue_M"])

# OLS on log scale
coeffs           = np.polyfit(df["log_mv"], df["Score"], 1)
df["Score_exp"]  = np.polyval(coeffs, df["log_mv"])
df["residual"]   = df["Score"] - df["Score_exp"]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def set_dark_bg(fig, ax):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.6)


def add_footnote(fig, text=FOOTNOTE):
    fig.text(0.98, 0.01, text, ha="right", va="bottom",
             fontsize=7, color=MUTED, style="italic")


# ===========================================================================
# VISUAL 1 — Scatter: Market Value vs Team Score
# ===========================================================================
fig1, ax1 = plt.subplots(figsize=(12, 7.5))
set_dark_bg(fig1, ax1)

# Trendline curve
mv_range = np.linspace(df["log_mv"].min() - 0.05, df["log_mv"].max() + 0.05, 300)
trend_y  = np.polyval(coeffs, mv_range)
ax1.plot(10**mv_range, trend_y, color=TRENDLINE, linewidth=1.8,
         linestyle="--", alpha=0.75, zorder=2, label="Expected (trend)")

y_min_plot, y_max_plot = 350, 760

# Scatter points
for comp, grp in df.groupby("Comp"):
    color = LEAGUE_COLORS.get(comp, "#FFFFFF")
    ax1.scatter(grp["MarketValue_M"], grp["Score"],
                color=color, s=55, alpha=0.85, zorder=4,
                label=LEAGUE_SHORT.get(comp, comp))

# Labels for strong outliers (|residual| > 60 or notable clubs)
LABEL_ALWAYS = {"Bayern Munich", "Real Madrid", "Barcelona", "Paris S-G",
                "Liverpool", "Arsenal", "Manchester City", "Chelsea",
                "Stuttgart", "Holstein Kiel", "Monaco",
                "Genoa", "Southampton", "Everton", "Torino"}
label_mask = (df["residual"].abs() > 65) | df["Squad"].isin(LABEL_ALWAYS)
for _, row in df[label_mask].iterrows():
    is_over  = row["residual"] >= 0
    color    = GREEN if is_over else RED
    dy       = 9 if is_over else -13
    ax1.annotate(
        row["Squad"],
        xy=(row["MarketValue_M"], row["Score"]),
        xytext=(0, dy),
        textcoords="offset points",
        fontsize=7.5,
        color=color,
        fontweight="bold",
        ha="center",
        va="bottom" if is_over else "top",
        zorder=6,
    )

# Zone text
ax1.text(0.02, 0.97, "More quality than budget suggests", transform=ax1.transAxes,
         fontsize=9, color=GREEN, fontweight="bold", va="top", alpha=0.8)
ax1.text(0.02, 0.03, "Less quality than budget suggests", transform=ax1.transAxes,
         fontsize=9, color=RED, fontweight="bold", va="bottom", alpha=0.8)

# Axes
ax1.set_xscale("log")
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"€{int(x):,}M".replace(",", ".")))
ax1.xaxis.set_minor_formatter(mticker.NullFormatter())
ax1.set_xlabel("Squad Market Value (€M, log scale)", fontsize=10)
ax1.set_ylabel("PlayerScore  (0 – 1000)", fontsize=10)
ax1.set_ylim(y_min_plot, y_max_plot)
ax1.set_xlim(25, 2800)

# Title
fig1.suptitle(
    "Squad Quality vs Squad Budget  —  Big-5  ·  2024 / 2025",
    fontsize=16, fontweight="bold", color=TEXT, x=0.05, ha="left", y=0.97
)
ax1.set_title(
    "Above trendline = more squad quality per € spent  ·  below = less quality than budget implies",
    fontsize=9, color=MUTED, loc="left", pad=6
)

# Legend
handles, labels = ax1.get_legend_handles_labels()
# keep only league entries (skip trendline)
league_handles = [h for h, l in zip(handles, labels) if l in LEAGUE_SHORT.values()]
league_labels  = [l for l in labels if l in LEAGUE_SHORT.values()]
trend_handle   = Line2D([0], [0], color=TRENDLINE, linewidth=1.8,
                        linestyle="--", label="Expected (trend)")
ax1.legend(
    handles=league_handles + [trend_handle],
    labels=league_labels + ["Expected (trend)"],
    framealpha=0.15, facecolor=BG, edgecolor=GRID,
    labelcolor=TEXT, fontsize=8, loc="upper left"
)

add_footnote(fig1)
fig1.tight_layout(rect=[0, 0.02, 1, 0.97])

out1 = OUT_DIR / "team_mv_scatter.png"
fig1.savefig(out1, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out1}")
plt.close(fig1)


# ===========================================================================
# VISUAL 2 — Bar chart: Top 12 Over- & Underperformers
# ===========================================================================
TOP_N = 12

over  = df.nlargest(TOP_N, "residual")[["Squad", "Comp", "Score", "MarketValue_M", "residual"]].copy()
under = df.nsmallest(TOP_N, "residual")[["Squad", "Comp", "Score", "MarketValue_M", "residual"]].copy()

# Combine and sort for a diverging bar chart
combined = pd.concat([over, under]).sort_values("residual")
combined["color"]  = combined["residual"].apply(lambda v: GREEN if v >= 0 else RED)
combined["label"]  = combined.apply(
    lambda r: f"{r['Squad']}  ({LEAGUE_SHORT.get(r['Comp'], r['Comp'])})", axis=1
)

fig2, ax2 = plt.subplots(figsize=(11, 9))
set_dark_bg(fig2, ax2)

bars = ax2.barh(
    combined["label"], combined["residual"],
    color=combined["color"], alpha=0.85, height=0.68, zorder=3
)

# Value labels on bars
for bar, (_, row) in zip(bars, combined.iterrows()):
    val  = row["residual"]
    sign = "+" if val >= 0 else ""
    xpos = val + (1.2 if val >= 0 else -1.2)
    ha   = "left" if val >= 0 else "right"
    ax2.text(xpos, bar.get_y() + bar.get_height() / 2,
             f"{sign}{val:.0f} pts", va="center", ha=ha,
             fontsize=7.8, color=TEXT, fontweight="bold")

# Market value annotations (small, right side)
for bar, (_, row) in zip(bars, combined.iterrows()):
    mv_txt = f"€{row['MarketValue_M']:.0f}M"
    ax2.text(0.99, bar.get_y() + bar.get_height() / 2,
             mv_txt, transform=ax2.get_yaxis_transform(),
             va="center", ha="right", fontsize=7, color=MUTED)

ax2.axvline(0, color=MUTED, linewidth=0.8, zorder=2)
ax2.set_xlabel("Quality delta vs budget expectation  (positive = more quality per €)", fontsize=9)
ax2.tick_params(axis="y", labelsize=8.5)

# Divider between over / under
n_under = len(under)
ax2.axhline(n_under - 0.5, color=GRID, linewidth=1.2, linestyle=":", zorder=5)

# Section labels
ax2.text(0.98, n_under + 0.1, "Efficient spend", transform=ax2.get_yaxis_transform(),
         fontsize=9, color=GREEN, fontweight="bold", va="bottom", ha="right")
ax2.text(0.98, n_under - 1.0, "Wasteful spend", transform=ax2.get_yaxis_transform(),
         fontsize=9, color=RED, fontweight="bold", va="top", ha="right")

fig2.suptitle(
    "Efficient vs Wasteful Squad Builds  —  Big-5  ·  2024 / 2025",
    fontsize=15, fontweight="bold", color=TEXT, x=0.03, ha="left", y=0.99
)
ax2.set_title(
    "Delta = actual PlayerScore minus expected score given squad market value  ·  positive = more quality per €",
    fontsize=8.5, color=MUTED, loc="left", pad=6
)

ax2.grid(axis="x", color=GRID, linewidth=0.5, linestyle="--", alpha=0.5)
ax2.grid(axis="y", visible=False)

add_footnote(fig2)
fig2.tight_layout(rect=[0, 0.02, 1, 0.98])

out2 = OUT_DIR / "team_mv_overunder.png"
fig2.savefig(out2, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out2}")
plt.close(fig2)


# ===========================================================================
# VISUAL 3 — Score per €100M: Squad Efficiency Ranking
# ===========================================================================
df["score_per_100m"] = df["Score"] / df["MarketValue_M"] * 100

TOP_EFF = 20   # show top 20 most efficient squads

eff = (
    df.nlargest(TOP_EFF, "score_per_100m")
    .sort_values("score_per_100m")
    .copy()
)
eff["color"]  = eff["Comp"].map(LEAGUE_COLORS).fillna("#FFFFFF")
eff["label"]  = eff.apply(
    lambda r: f"{r['Squad']}  ({LEAGUE_SHORT.get(r['Comp'], r['Comp'])})", axis=1
)

fig3, ax3 = plt.subplots(figsize=(11, 8.5))
set_dark_bg(fig3, ax3)

bars3 = ax3.barh(
    eff["label"], eff["score_per_100m"],
    color=eff["color"], alpha=0.85, height=0.68, zorder=3
)

# Value + market value labels
for bar, (_, row) in zip(bars3, eff.iterrows()):
    val = row["score_per_100m"]
    ax3.text(
        val + 0.3, bar.get_y() + bar.get_height() / 2,
        f"{val:.1f} pts", va="center", ha="left",
        fontsize=8, color=TEXT, fontweight="bold"
    )
    ax3.text(
        0.99, bar.get_y() + bar.get_height() / 2,
        f"€{row['MarketValue_M']:.0f}M  ·  {row['Score']:.0f} pts",
        transform=ax3.get_yaxis_transform(),
        va="center", ha="right", fontsize=7, color=MUTED
    )

# League legend
legend_handles = [
    Line2D([0], [0], marker="o", color="none",
           markerfacecolor=c, markersize=9, label=LEAGUE_SHORT[k])
    for k, c in LEAGUE_COLORS.items()
    if LEAGUE_SHORT[k] in eff["Comp"].map(LEAGUE_SHORT).values
]
ax3.legend(handles=legend_handles, framealpha=0.15, facecolor=BG,
           edgecolor=GRID, labelcolor=TEXT, fontsize=8, loc="lower right")

ax3.set_xlabel("PlayerScore per €100M invested", fontsize=9)
ax3.tick_params(axis="y", labelsize=8.5)
ax3.grid(axis="x", color=GRID, linewidth=0.5, linestyle="--", alpha=0.5)
ax3.grid(axis="y", visible=False)

fig3.suptitle(
    "Squad Efficiency Ranking  —  Big-5  ·  2024 / 2025",
    fontsize=15, fontweight="bold", color=TEXT, x=0.03, ha="left", y=0.99
)
ax3.set_title(
    "PlayerScore per €100M squad market value  ·  top 20 most efficient squads",
    fontsize=8.5, color=MUTED, loc="left", pad=6
)

add_footnote(fig3)
fig3.tight_layout(rect=[0, 0.02, 1, 0.98])

out3 = OUT_DIR / "team_mv_efficiency.png"
fig3.savefig(out3, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out3}")
plt.close(fig3)

print("Done.")
