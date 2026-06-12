"""
LinkedIn Visuals: PlayerScore vs League Table Performance
=========================================================

Visual 1 — Scatter: PlayerScore vs League Position (all seasons, Big-5)
Visual 2 — Overachievers & Underachievers (score vs expected position)
Visual 3 — Score vs Points per season (normalized within each league)
Visual 4 — Champions vs Rest: PlayerScore distribution per season
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

matplotlib.rcParams["font.family"] = "DejaVu Sans"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data" / "Processed"
OUT_DIR  = Path(__file__).resolve().parent / "visuals"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Style — matches existing dark theme
# ---------------------------------------------------------------------------
BG      = "#0D0D0D"
SURFACE = "#161616"
TEXT    = "#E5E7EB"
MUTED   = "#6B7280"
GRID    = "#1F2937"
GREEN   = "#22C55E"
RED     = "#EF4444"
ACCENT  = "#38BDF8"
GOLD    = "#FACC15"

LEAGUE_COLORS = {
    "eng Premier League": "#38BDF8",
    "es La Liga":         "#FACC15",
    "de Bundesliga":      "#FB923C",
    "it Serie A":         "#4ADE80",
    "fr Ligue 1":         "#C084FC",
}
LEAGUE_SHORT = {
    "eng Premier League": "Premier League",
    "es La Liga":         "La Liga",
    "de Bundesliga":      "Bundesliga",
    "it Serie A":         "Serie A",
    "fr Ligue 1":         "Ligue 1",
}

FOOTNOTE = "PlayerScore by Stefan Hofmann  •  Data: FBref  •  Seasons 2017–2026"


# ---------------------------------------------------------------------------
# Load & merge
# ---------------------------------------------------------------------------
big5   = pd.read_csv(DATA_DIR / "big5_table_all_seasons.csv")
squads = pd.read_csv(DATA_DIR / "squad_scores_all_seasons.csv")

df = big5.merge(squads, on=["Season", "Squad"], how="inner")
df["OverallScore_squad"] = pd.to_numeric(df["OverallScore_squad"], errors="coerce")
df["LgRk"]               = pd.to_numeric(df["LgRk"],               errors="coerce")
df["Pts"]                = pd.to_numeric(df["Pts"],                 errors="coerce")
df = df.dropna(subset=["OverallScore_squad", "LgRk", "Pts", "Comp"])

# Number of teams per league-season (for normalised rank)
league_sizes = (
    df.groupby(["Season", "Comp"])["LgRk"]
    .max()
    .rename("LeagueSize")
    .reset_index()
)
df = df.merge(league_sizes, on=["Season", "Comp"])
# Normalised rank: 1.0 = champion, 0.0 = last
df["NormRank"] = 1.0 - (df["LgRk"] - 1) / (df["LeagueSize"] - 1)

print(f"Merged dataset: {len(df)} team-seasons across {df['Season'].nunique()} seasons")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def dark_axes(fig, ax):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.6)


def footnote(fig, text=FOOTNOTE):
    fig.text(0.98, 0.01, text, ha="right", va="bottom",
             fontsize=7, color=MUTED, style="italic")


# ===========================================================================
# VISUAL 1 — Scatter: PlayerScore vs Normalised League Rank (all seasons)
# ===========================================================================
fig1, ax1 = plt.subplots(figsize=(12, 7.5))
dark_axes(fig1, ax1)

for comp, grp in df.groupby("Comp"):
    color = LEAGUE_COLORS.get(comp, "#FFFFFF")
    ax1.scatter(
        grp["OverallScore_squad"], grp["NormRank"],
        color=color, s=28, alpha=0.55, zorder=4,
        label=LEAGUE_SHORT.get(comp, comp),
    )

# OLS trendline across all leagues
x, y = df["OverallScore_squad"].values, df["NormRank"].values
coeffs = np.polyfit(x, y, 1)
xr = np.linspace(x.min(), x.max(), 200)
ax1.plot(xr, np.polyval(coeffs, xr), color="#94A3B8",
         linewidth=2, linestyle="--", zorder=5, label="Trend (OLS)")

r, p = pearsonr(x, y)
ax1.text(0.98, 0.04,
         f"Pearson r = {r:.2f}  (p {'< 0.001' if p < 0.001 else f'= {p:.3f}'})",
         transform=ax1.transAxes, ha="right", va="bottom",
         fontsize=9, color=MUTED, style="italic")

# highlight champions
champs = df[df["LgRk"] == 1]
ax1.scatter(champs["OverallScore_squad"], champs["NormRank"],
            color=GOLD, s=70, zorder=6, marker="*", label="Champions")

ax1.set_xlabel("PlayerScore  (0 – 1 000)", fontsize=10)
ax1.set_ylabel("Normalised League Rank  (1 = champion, 0 = last)", fontsize=10)
ax1.set_xlim(300, 850)
ax1.set_ylim(-0.05, 1.08)

fig1.suptitle(
    "Does Squad Quality Predict League Success?  —  Big-5  ·  2017 – 2026",
    fontsize=15, fontweight="bold", color=TEXT, x=0.04, ha="left", y=0.98,
)
ax1.set_title(
    "Each dot = one team-season  ·  normalised rank: 1 = title, 0 = relegation zone",
    fontsize=9, color=MUTED, loc="left", pad=6,
)
ax1.legend(framealpha=0.15, facecolor=BG, edgecolor=GRID,
           labelcolor=TEXT, fontsize=8.5, loc="upper left")

footnote(fig1)
fig1.tight_layout(rect=[0, 0.02, 1, 0.97])
out1 = OUT_DIR / "league_score_vs_rank.png"
fig1.savefig(out1, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out1}")
plt.close(fig1)


# ===========================================================================
# VISUAL 2 — Overachievers & Underachievers  (expected vs actual position)
# ===========================================================================
# Fit PlayerScore -> NormRank per league to get expected rank
residuals = []
for (season, comp), grp in df.groupby(["Season", "Comp"]):
    if len(grp) < 5:
        continue
    c = np.polyfit(grp["OverallScore_squad"], grp["NormRank"], 1)
    grp = grp.copy()
    grp["exp_rank"]  = np.polyval(c, grp["OverallScore_squad"])
    grp["rank_diff"] = grp["NormRank"] - grp["exp_rank"]   # positive = outperformed
    residuals.append(grp)

res_df = pd.concat(residuals, ignore_index=True)

TOP_N = 14
over  = res_df.nlargest(TOP_N, "rank_diff").copy()
under = res_df.nsmallest(TOP_N, "rank_diff").copy()

combined = pd.concat([under, over]).sort_values("rank_diff")
combined["color"] = combined["rank_diff"].apply(lambda v: GREEN if v >= 0 else RED)
combined["label"] = combined.apply(
    lambda r: f"{r['Squad']}  {r['Season'][-4:]}  ({LEAGUE_SHORT.get(r['Comp'], r['Comp'])})",
    axis=1,
)

fig2, ax2 = plt.subplots(figsize=(12, 10))
dark_axes(fig2, ax2)

bars = ax2.barh(
    combined["label"], combined["rank_diff"] * 100,
    color=combined["color"], alpha=0.85, height=0.72, zorder=3,
)

for bar, (_, row) in zip(bars, combined.iterrows()):
    v    = row["rank_diff"] * 100
    sign = "+" if v >= 0 else ""
    xpos = v + (0.5 if v >= 0 else -0.5)
    ha   = "left" if v >= 0 else "right"
    ax2.text(xpos, bar.get_y() + bar.get_height() / 2,
             f"{sign}{v:.1f}", va="center", ha=ha,
             fontsize=7.5, color=TEXT, fontweight="bold")

ax2.axvline(0, color=MUTED, linewidth=0.8, zorder=2)
n_under = len(under)
ax2.axhline(n_under - 0.5, color=GRID, linewidth=1.2, linestyle=":", zorder=5)

ax2.text(0.98, n_under + 0.15, "Overachievers",
         transform=ax2.get_yaxis_transform(),
         fontsize=9, color=GREEN, fontweight="bold", va="bottom", ha="right")
ax2.text(0.98, n_under - 1.05, "Underachievers",
         transform=ax2.get_yaxis_transform(),
         fontsize=9, color=RED, fontweight="bold", va="top", ha="right")

ax2.set_xlabel(
    "Rank delta vs expected  (positive = finished higher than PlayerScore suggested)",
    fontsize=9,
)
ax2.tick_params(axis="y", labelsize=8)
ax2.grid(axis="x", color=GRID, linewidth=0.5, linestyle="--", alpha=0.5)
ax2.grid(axis="y", visible=False)

fig2.suptitle(
    "League Overachievers & Underachievers  —  Big-5  ·  2017 – 2026",
    fontsize=15, fontweight="bold", color=TEXT, x=0.03, ha="left", y=0.99,
)
ax2.set_title(
    "Delta = actual normalised rank minus expected rank given PlayerScore  ·  across all seasons",
    fontsize=8.5, color=MUTED, loc="left", pad=6,
)

footnote(fig2)
fig2.tight_layout(rect=[0, 0.02, 1, 0.98])
out2 = OUT_DIR / "league_over_under_achievers.png"
fig2.savefig(out2, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out2}")
plt.close(fig2)


# ===========================================================================
# VISUAL 3 — PlayerScore vs Points (normalized within league-season)
# ===========================================================================
# Normalise Pts within each league-season (0–100)
df["PtsNorm"] = df.groupby(["Season", "Comp"])["Pts"].transform(
    lambda s: (s - s.min()) / (s.max() - s.min()) * 100
)

fig3, ax3 = plt.subplots(figsize=(12, 7.5))
dark_axes(fig3, ax3)

for comp, grp in df.groupby("Comp"):
    color = LEAGUE_COLORS.get(comp, "#FFFFFF")
    ax3.scatter(
        grp["OverallScore_squad"], grp["PtsNorm"],
        color=color, s=28, alpha=0.55, zorder=4,
        label=LEAGUE_SHORT.get(comp, comp),
    )

# Trendline
x3, y3 = df["OverallScore_squad"].values, df["PtsNorm"].values
c3     = np.polyfit(x3, y3, 1)
xr3    = np.linspace(x3.min(), x3.max(), 200)
ax3.plot(xr3, np.polyval(c3, xr3), color="#94A3B8",
         linewidth=2, linestyle="--", zorder=5, label="Trend (OLS)")

r3, p3 = pearsonr(x3, y3)
ax3.text(0.98, 0.04,
         f"Pearson r = {r3:.2f}  (p {'< 0.001' if p3 < 0.001 else f'= {p3:.3f}'})",
         transform=ax3.transAxes, ha="right", va="bottom",
         fontsize=9, color=MUTED, style="italic")

ax3.set_xlabel("PlayerScore  (0 – 1 000)", fontsize=10)
ax3.set_ylabel("Points (normalised within league-season, 0 = last · 100 = 1st)", fontsize=10)
ax3.set_xlim(300, 850)
ax3.set_ylim(-5, 108)

fig3.suptitle(
    "PlayerScore vs Points Earned  —  Big-5  ·  2017 – 2026",
    fontsize=15, fontweight="bold", color=TEXT, x=0.04, ha="left", y=0.98,
)
ax3.set_title(
    "Each dot = one team-season  ·  points normalised within each league-season",
    fontsize=9, color=MUTED, loc="left", pad=6,
)
ax3.legend(framealpha=0.15, facecolor=BG, edgecolor=GRID,
           labelcolor=TEXT, fontsize=8.5, loc="upper left")

footnote(fig3)
fig3.tight_layout(rect=[0, 0.02, 1, 0.97])
out3 = OUT_DIR / "league_score_vs_points.png"
fig3.savefig(out3, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out3}")
plt.close(fig3)


# ===========================================================================
# VISUAL 4 — Champions vs Rest: PlayerScore distribution per season
# ===========================================================================
# Per-league champions
champ_df   = df[df["LgRk"] == 1].copy()
nonchamp   = df[df["LgRk"] != 1].copy()

seasons = sorted(df["Season"].unique())

champ_scores   = champ_df.groupby("Season")["OverallScore_squad"].mean()
nonchamp_mean  = nonchamp.groupby("Season")["OverallScore_squad"].mean()
nonchamp_p75   = nonchamp.groupby("Season")["OverallScore_squad"].quantile(0.75)
nonchamp_p25   = nonchamp.groupby("Season")["OverallScore_squad"].quantile(0.25)

x_idx = np.arange(len(seasons))

fig4, ax4 = plt.subplots(figsize=(12, 6.5))
dark_axes(fig4, ax4)

ax4.fill_between(x_idx, nonchamp_p25.loc[seasons], nonchamp_p75.loc[seasons],
                 color=ACCENT, alpha=0.15, label="Rest — IQR (25–75th pct)")
ax4.plot(x_idx, nonchamp_mean.loc[seasons], color=ACCENT,
         linewidth=2, marker="o", markersize=5, label="Rest — mean")
ax4.plot(x_idx, champ_scores.loc[seasons],  color=GOLD,
         linewidth=2.5, marker="*", markersize=10, label="Champions — mean")

# gap annotation on latest season
latest_s  = seasons[-1]
gap       = champ_scores[latest_s] - nonchamp_mean[latest_s]
ax4.annotate(
    f"+{gap:.0f} pts gap",
    xy=(len(seasons) - 1, champ_scores[latest_s]),
    xytext=(-30, 18), textcoords="offset points",
    fontsize=9, color=GOLD, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2),
)

ax4.set_xticks(x_idx)
ax4.set_xticklabels([s[-4:] for s in seasons], fontsize=9)
ax4.set_xlabel("Season (end year)", fontsize=10)
ax4.set_ylabel("PlayerScore  (0 – 1 000)", fontsize=10)
ax4.set_xlim(-0.3, len(seasons) - 0.7)
ax4.legend(framealpha=0.15, facecolor=BG, edgecolor=GRID,
           labelcolor=TEXT, fontsize=9, loc="lower right")

fig4.suptitle(
    "Champions vs Rest — How Much Better Are Title Winners?  —  Big-5  ·  2017 – 2026",
    fontsize=14, fontweight="bold", color=TEXT, x=0.04, ha="left", y=0.98,
)
ax4.set_title(
    "Average PlayerScore of league champions vs all other teams per season",
    fontsize=9, color=MUTED, loc="left", pad=6,
)

footnote(fig4)
fig4.tight_layout(rect=[0, 0.02, 1, 0.97])
out4 = OUT_DIR / "league_champions_vs_rest.png"
fig4.savefig(out4, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out4}")
plt.close(fig4)

print("\nAll 4 visuals done.")
