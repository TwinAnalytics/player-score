"""
LinkedIn Carousel: IntensityScore — No GPS. No Problem.
Generates 6 PNG slides (1080x1080) in LinkedIn Posts/visuals/intensity_carousel/
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Colors ──────────────────────────────────────────────────────────────────
BG        = "#0D1117"
BG2       = "#161B22"
TEAL      = "#3DD9CC"
TEAL_DIM  = "#1A6B65"
WHITE     = "#F9FAFB"
GRAY      = "#6B7280"
GRAY2     = "#374151"
GRAY3     = "#1F2937"
RED       = "#EF4444"

# ── Output dir ───────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "visuals" / "intensity_carousel"
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 10.8, 10.8   # inches @ 100 dpi → 1080x1080 px
DPI  = 100

def new_fig():
    fig = plt.figure(figsize=(W, H), facecolor=BG)
    ax  = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax

def add_logo_bar(ax, slide_num: int, total: int = 6):
    """Bottom bar with slide counter."""
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 0.055, color=BG2, zorder=3))
    ax.text(0.05, 0.027, "PlayerScore", color=TEAL, fontsize=11,
            fontweight="bold", va="center", zorder=4)
    ax.text(0.95, 0.027, f"{slide_num} / {total}", color=GRAY, fontsize=10,
            va="center", ha="right", zorder=4)

def add_top_accent(ax):
    ax.add_patch(mpatches.Rectangle((0, 0.96), 1, 0.04, color=TEAL, zorder=3))

def save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Slide 1 — Hook ───────────────────────────────────────────────────────────
def slide1():
    fig, ax = new_fig()
    add_top_accent(ax)
    add_logo_bar(ax, 1)

    # Big headline
    ax.text(0.5, 0.68, "No GPS?", color=WHITE, fontsize=58,
            fontweight="bold", ha="center", va="center")
    ax.text(0.5, 0.54, "Event data still tells", color=TEAL, fontsize=36,
            fontweight="bold", ha="center", va="center")
    ax.text(0.5, 0.45, "a physical story.", color=TEAL, fontsize=36,
            fontweight="bold", ha="center", va="center")

    # Divider
    ax.plot([0.2, 0.8], [0.37, 0.37], color=GRAY2, linewidth=1.5)

    ax.text(0.5, 0.27, "How I built a physical intensity score", color=GRAY,
            fontsize=16, ha="center", va="center")
    ax.text(0.5, 0.21, "using only event data.", color=GRAY,
            fontsize=16, ha="center", va="center")

    # Tag pills
    for i, tag in enumerate(["#FootballAnalytics", "#DataScience", "#FBref"]):
        x = 0.22 + i * 0.28
        ax.add_patch(FancyBboxPatch((x - 0.095, 0.105), 0.19, 0.045,
                                    boxstyle="round,pad=0.01",
                                    facecolor=GRAY3, edgecolor=GRAY2, linewidth=1))
        ax.text(x, 0.128, tag, color=TEAL, fontsize=10, ha="center", va="center")

    save(fig, "slide1_hook.png")


# ── Slide 2 — The Problem ────────────────────────────────────────────────────
def slide2():
    fig, ax = new_fig()
    add_top_accent(ax)
    add_logo_bar(ax, 2)

    ax.text(0.5, 0.87, "The Problem", color=TEAL, fontsize=18,
            fontweight="bold", ha="center")

    ax.text(0.5, 0.78, "FBref has no tracking data.", color=WHITE,
            fontsize=28, fontweight="bold", ha="center")

    items = [
        ("✗", "No distance covered"),
        ("✗", "No sprint count"),
        ("✗", "No high-intensity runs"),
    ]
    for i, (sym, text) in enumerate(items):
        y = 0.64 - i * 0.10
        ax.text(0.22, y, sym, color=RED, fontsize=22, fontweight="bold", va="center")
        ax.text(0.29, y, text, color=GRAY, fontsize=18, va="center")

    ax.plot([0.1, 0.9], [0.32, 0.32], color=GRAY2, linewidth=1)

    ax.text(0.5, 0.25, "But event data still tells a physical story",
            color=WHITE, fontsize=15, ha="center", style="italic")
    ax.text(0.5, 0.18, "if you know where to look.",
            color=TEAL, fontsize=15, ha="center", fontweight="bold", style="italic")

    save(fig, "slide2_problem.png")


# ── Slide 3 — The 5 Proxies ──────────────────────────────────────────────────
def slide3():
    fig, ax = new_fig()
    add_top_accent(ax)
    add_logo_bar(ax, 3)

    ax.text(0.5, 0.87, "5 FBref Metrics as Physical Proxies", color=TEAL,
            fontsize=17, fontweight="bold", ha="center")

    metrics = [
        ("Ball Recoveries / 90",             "Off-ball workrate",           0.30),
        ("Ball Carries / 90",                "Ball engagement volume",       0.20),
        ("Progressive Carry Distance / 90",  "Forward drive",               0.25),
        ("Aerial Duels Won / 90",            "Physical contest",            0.10),
        ("Tackles + Interceptions / 90",     "Pressing intensity",          0.15),
    ]

    bar_x0, bar_w_max = 0.60, 0.28
    y_start = 0.76

    for i, (metric, desc, w) in enumerate(metrics):
        y = y_start - i * 0.127

        # Row bg
        ax.add_patch(FancyBboxPatch((0.06, y - 0.042), 0.88, 0.075,
                                    boxstyle="round,pad=0.01",
                                    facecolor=BG2, edgecolor=GRAY3, linewidth=1))

        # Metric name + description
        ax.text(0.12, y + 0.012, metric, color=WHITE, fontsize=13,
                fontweight="bold", va="center")
        ax.text(0.12, y - 0.018, desc, color=GRAY, fontsize=10, va="center")

        # Weight bar (background)
        ax.add_patch(mpatches.Rectangle((bar_x0, y - 0.012), bar_w_max, 0.020,
                                         color=GRAY3))
        # Weight bar (filled)
        ax.add_patch(mpatches.Rectangle((bar_x0, y - 0.012), bar_w_max * w / 0.30, 0.020,
                                         color=TEAL))
        # Weight label
        ax.text(bar_x0 + bar_w_max + 0.015, y - 0.001,
                f"{int(w*100)}%", color=TEAL, fontsize=11,
                fontweight="bold", va="center")

    save(fig, "slide3_proxies.png")


# ── Slide 4 — Position-Specific Weights ──────────────────────────────────────
def slide4():
    fig, ax = new_fig()
    add_top_accent(ax)
    add_logo_bar(ax, 4)

    ax.text(0.5, 0.88, "One score. Five different lenses.", color=TEAL,
            fontsize=17, fontweight="bold", ha="center")
    ax.text(0.5, 0.82, "A DM and a winger have very different intensity profiles.",
            color=GRAY, fontsize=13, ha="center")

    # Table
    cols   = ["Position", "Carries", "PrgDist", "Recoveries", "Tkl+Int", "Aerials"]
    rows = [
        ["FW",      "30%", "30%", "20%", "10%", "10%"],
        ["Off MF",  "27%", "27%", "25%", "14%",  "7%"],
        ["MF",      "20%", "20%", "25%", "25%", "10%"],
        ["Def MF",  "12%",  "8%", "30%", "35%", "15%"],
        ["DF",      "10%", "10%", "25%", "30%", "25%"],
    ]

    # Highlight which metric dominates per row
    row_highlights = [
        [0, 1],   # FW: Carries + PrgDist
        [1, 2],   # Off_MF: PrgDist + Recoveries
        [3, 4],   # MF: Tkl+Int + Recoveries (balanced)
        [3, 4],   # Def_MF: Tkl+Int + Recoveries
        [3, 4],   # DF: Tkl+Int + Aerials
    ]
    # Fix: highlight correct cols (1-based for data cols)
    row_highlights = [
        [1, 2],   # FW: Carries + PrgDist
        [1, 2],   # Off_MF: Carries + PrgDist
        [2, 3],   # MF: Recoveries + Tkl+Int
        [3, 4],   # Def_MF: Recoveries + Tkl+Int
        [3, 4],   # DF: Tkl+Int + Aerials → cols 3,4
    ]
    row_highlights = [
        {1, 2},   # FW
        {1, 2},   # Off_MF
        {2, 3},   # MF
        {2, 3},   # Def_MF
        {3, 4},   # DF
    ]

    col_x = [0.08, 0.26, 0.40, 0.54, 0.68, 0.82]
    row_y0 = 0.72
    row_h  = 0.090

    # Header
    for j, col in enumerate(cols):
        ax.text(col_x[j] + 0.05, row_y0 + 0.01, col,
                color=TEAL if j > 0 else WHITE,
                fontsize=11, fontweight="bold", ha="center", va="center")

    ax.plot([0.05, 0.95], [row_y0 - 0.012, row_y0 - 0.012], color=GRAY2, lw=1)

    for i, (row, highlights) in enumerate(zip(rows, row_highlights)):
        y = row_y0 - 0.025 - i * row_h
        bg = BG2 if i % 2 == 0 else BG
        ax.add_patch(mpatches.Rectangle((0.05, y - 0.028), 0.90, row_h * 0.92,
                                         color=bg))
        for j, val in enumerate(row):
            is_highlight = (j - 1) in highlights and j > 0
            color = TEAL if is_highlight else (WHITE if j == 0 else GRAY)
            weight = "bold" if is_highlight or j == 0 else "normal"
            ax.text(col_x[j] + 0.05, y - 0.001, val,
                    color=color, fontsize=12 if j == 0 else 11,
                    fontweight=weight, ha="center", va="center")

    ax.text(0.5, 0.10, "Teal = dominant metric for that position",
            color=GRAY, fontsize=10, ha="center", style="italic")

    save(fig, "slide4_weights.png")


# ── Slide 5 — Example Breakdown (real screenshots) ───────────────────────────
def slide5():
    import matplotlib.image as mpimg

    img1_path = Path(__file__).parent / "visuals" / "Intensity_Breakdown_Frankiedejong.png"
    img2_path = Path(__file__).parent / "visuals" / "Frenkie_2.png"

    fig, ax = new_fig()
    add_top_accent(ax)
    add_logo_bar(ax, 5)

    # Header
    ax.text(0.5, 0.89, "Not just a score. A breakdown.", color=TEAL,
            fontsize=17, fontweight="bold", ha="center")
    ax.text(0.5, 0.83, "Frenkie de Jong · MF · FC Barcelona · 2024/25",
            color=GRAY, fontsize=13, ha="center")

    # Divider label top image
    ax.text(0.05, 0.78, "Metric breakdown, percentile vs. MF peers", color=GRAY,
            fontsize=9, style="italic")

    # Top image: breakdown bar
    img1 = mpimg.imread(img1_path)
    ax1 = fig.add_axes([0.04, 0.52, 0.92, 0.26])
    ax1.imshow(img1)
    ax1.axis("off")

    # Divider label bottom image
    ax.text(0.05, 0.49, "Intensity vs. Role Score, MF peers (Big-5)", color=GRAY,
            fontsize=9, style="italic")

    # Bottom image: scatter
    img2 = mpimg.imread(img2_path)
    ax2 = fig.add_axes([0.04, 0.20, 0.92, 0.29])
    ax2.imshow(img2)
    ax2.axis("off")

    # Caption
    ax.text(0.5, 0.13, "93rd pct. Ball Carries, top-right in the scatter. High intensity and high role score.",
            color=WHITE, fontsize=11, ha="center", style="italic")
    ax.text(0.5, 0.08, "Built from FBref event data only.",
            color=GRAY, fontsize=10, ha="center")

    save(fig, "slide5_example.png")


# ── Slide 6 — CTA ────────────────────────────────────────────────────────────
def slide6():
    fig, ax = new_fig()
    add_top_accent(ax)
    add_logo_bar(ax, 6)

    ax.text(0.5, 0.85, "PlayerScore", color=TEAL, fontsize=36,
            fontweight="bold", ha="center")
    ax.text(0.5, 0.77, "Open football analytics tool", color=WHITE,
            fontsize=16, ha="center")
    ax.text(0.5, 0.71, "covering all Big-5 leagues · 2017 – 2026", color=GRAY,
            fontsize=13, ha="center")

    ax.plot([0.2, 0.8], [0.65, 0.65], color=GRAY2, lw=1)

    features = [
        "Player profiles: Off / Mid / Def / Intensity scores",
        "Squad comparisons across 9 seasons",
        "Pizza charts · Scatter plots · Career trends",
        "Position-specific scoring for all outfield roles",
    ]
    for i, f in enumerate(features):
        y = 0.59 - i * 0.078
        ax.text(0.18, y, "✓", color=TEAL, fontsize=14, va="center", fontweight="bold")
        ax.text(0.23, y, f, color=WHITE, fontsize=13, va="center")

    ax.plot([0.2, 0.8], [0.27, 0.27], color=GRAY2, lw=1)

    ax.text(0.5, 0.17, "Built with Python, Streamlit and FBref data",
            color=GRAY, fontsize=12, ha="center")

    save(fig, "slide6_cta.png")


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating LinkedIn carousel slides...")
    slide1()
    slide2()
    slide3()
    slide4()
    slide5()
    slide6()
    print(f"\nDone! Slides saved to:\n  {OUT_DIR}")
