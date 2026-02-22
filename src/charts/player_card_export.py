"""
Matplotlib-based player card PNG — matches the in-app HTML FIFA card layout.
Gradient background (teal → dark), score top-left, player name, attributes grid.
"""
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

from src.club_crests import get_crest_bytes

# ── Colours (match HTML card exactly) ─────────────────────────────────────────
_GRAD_A  = (0x2E / 255, 0xF2 / 255, 0xE0 / 255)   # bright teal  (#2EF2E0)
_GRAD_B  = (0x00 / 255, 0x89 / 255, 0x7B / 255)   # mid teal     (#00897B)
_GRAD_C  = (0x0B / 255, 0x1F / 255, 0x1E / 255)   # almost black (#0B1F1E)
_WHITE   = "#FFFFFF"
_OFFWHITE = "#F0FFFE"
_AMBER   = "#fde68a"
_SEMI    = (0, 0, 0, 0.25)    # rgba semi-transparent black for pill

BAND_COLORS = {
    "Exceptional":        "#a855f7",
    "World Class":        "#22c55e",
    "Top Starter":        "#3b82f6",
    "Solid Squad Player": "#eab308",
    "Below Big-5 Level":  "#9ca3af",
}


def _build_attrs(row: pd.Series) -> list[tuple[str, str]]:
    """Build the 6 attribute (label, formatted_value) pairs — identical to render_fifa_card."""

    def _pick(candidates: list[str], decimals: int = 2, is_percent: bool = False) -> str:
        for c in candidates:
            if c in row.index:
                val = row.get(c)
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    continue
                try:
                    v = float(val)
                    if is_percent:
                        return f"{v:.0f}%"
                    return f"{v:.{decimals}f}"
                except Exception:
                    continue
        return "-"

    # Minutes (integer)
    minutes_raw = row.get("Min", row.get("Minutes", None))
    if (minutes_raw is None or (isinstance(minutes_raw, float) and pd.isna(minutes_raw))) \
            and "90s" in row.index:
        try:
            minutes_raw = float(row.get("90s")) * 90.0
        except Exception:
            minutes_raw = None
    if minutes_raw is not None:
        try:
            minutes_str = str(int(round(float(minutes_raw))))
        except Exception:
            minutes_str = "-"
    else:
        minutes_str = "-"

    pass_acc = _pick(["Cmp%", "Cmp_Pct", "PassCmp_Pct"], decimals=0, is_percent=True)

    base_attrs = [("Min", minutes_str), ("Pass Acc", pass_acc)]

    pos_upper = str(row.get("Pos", row.get("Pos_raw", "")) or "").upper()
    if "FW" in pos_upper:
        role_type = "FW"
    elif "DF" in pos_upper or "CB" in pos_upper or "FB" in pos_upper:
        role_type = "DF"
    else:
        role_type = "MF"

    if role_type == "FW":
        role_attrs = [
            ("G/90",      _pick(["Gls_Per90", "G_90", "Gls90"])),
            ("xG/90",     _pick(["xG_Per90", "xG_90", "xG90"])),
            ("npxG/90",   _pick(["npxG_Per90", "npxG_90", "npxG90"])),
            ("Shots/90",  _pick(["Sh_Per90", "SoT_Per90", "Sh_90"])),
        ]
    elif role_type == "DF":
        role_attrs = [
            ("Blocks/90",   _pick(["Blocks_Per90", "Blocks_stats_defense_Per90",
                                   "Blocks_stats_defense_90", "Blocks_90"])),
            ("Int/90",      _pick(["Int_Per90", "Tkl+Int_Per90", "Int_90"])),
            ("Clr/90",      _pick(["Clr_Per90", "Clr_90"])),
            ("Aerials Won", _pick(["Won%", "AerialWon_Pct",
                                   "AerialDuelsWon_Pct", "AerialDuels_Won%"],
                                  decimals=0, is_percent=True)),
        ]
    else:  # MF (incl. Off_MF, Def_MF)
        role_attrs = [
            ("Ast/90",  _pick(["xAG_Per90", "xA_Per90", "xA_90"])),
            ("xAG/90",  _pick(["Ast_Per90", "Ast_90", "Ast90"])),
            ("KP/90",   _pick(["KP_Per90", "KP_90"])),
            ("SCA/90",  _pick(["SCA_Per90", "SCA90", "SCA_90"])),
        ]

    return base_attrs + role_attrs


def _make_gradient(h: int, w: int) -> np.ndarray:
    """Radial gradient matching CSS: circle at 0% 0% (top-left)."""
    Y, X = np.mgrid[0:h, 0:w]
    xn = X / w
    yn = Y / h
    # distance from top-left corner
    dist = np.sqrt(xn ** 2 + yn ** 2) / np.sqrt(2)
    # ramp: 0→A, 0.35→B, 1→C
    t1 = np.clip(dist / 0.35, 0, 1)
    t2 = np.clip((dist - 0.35) / 0.65, 0, 1)
    r = _GRAD_A[0] * (1 - t1) + _GRAD_B[0] * t1 * (1 - t2) + _GRAD_C[0] * t2
    g = _GRAD_A[1] * (1 - t1) + _GRAD_B[1] * t1 * (1 - t2) + _GRAD_C[1] * t2
    b = _GRAD_A[2] * (1 - t1) + _GRAD_B[2] * t1 * (1 - t2) + _GRAD_C[2] * t2
    return np.stack([r, g, b], axis=2)


def generate_player_card_png(
    row: pd.Series,
    score: float,
    band: str,
    fmt_market_value_fn=None,
) -> bytes:
    # ── Figure setup ──────────────────────────────────────────────────────────
    DPI = 200
    W_IN, H_IN = 2.7, 3.8          # inches → 540×760 px at 200 DPI
    fig = plt.figure(figsize=(W_IN, H_IN), facecolor=_GRAD_C)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Gradient background ───────────────────────────────────────────────────
    H_PX = int(H_IN * DPI)
    W_PX = int(W_IN * DPI)
    grad = _make_gradient(H_PX, W_PX)
    ax.imshow(grad, extent=[0, 1, 0, 1], origin="upper", aspect="auto", zorder=0)

    # ── Rounded card border (decorative) ─────────────────────────────────────
    card_border = mpatches.FancyBboxPatch(
        (0.03, 0.02), 0.94, 0.96,
        boxstyle="round,pad=0.02",
        facecolor="none",
        edgecolor=(1, 1, 1, 0.12),
        linewidth=1.5,
        zorder=5,
    )
    ax.add_patch(card_border)

    PAD = 0.07   # left/right padding

    # ── TOP ROW: Score + Position + Band (left) / Age + League (right) ───────
    score_int = int(round(score)) if score is not None else 0
    pos_display = str(row.get("Pos", row.get("Pos_raw", "-")))
    age_val = row.get("Age")
    comp = str(row.get("Comp", "") or "")
    comp_clean = (comp.replace("eng ", "").replace("es ", "")
                  .replace("de ", "").replace("it ", "").replace("fr ", ""))

    # Score — large, white
    ax.text(
        PAD, 0.96, str(score_int),
        ha="left", va="top",
        fontsize=38, fontweight="bold", color=_WHITE, zorder=6,
        path_effects=[pe.withStroke(linewidth=2, foreground=(0, 0, 0, 0.25))],
    )
    # Position below score
    ax.text(
        PAD, 0.79, pos_display.upper(),
        ha="left", va="top",
        fontsize=11, fontweight="bold", color=_OFFWHITE,
        alpha=0.92, zorder=6,
    )

    # Band pill
    band_color = BAND_COLORS.get(band, "#9ca3af")
    band_pill = mpatches.FancyBboxPatch(
        (PAD - 0.01, 0.695), 0.38, 0.048,
        boxstyle="round,pad=0.008",
        facecolor=(0, 0, 0, 0.28),
        edgecolor=(1, 1, 1, 0.38),
        linewidth=0.7, zorder=6,
    )
    ax.add_patch(band_pill)
    ax.text(
        PAD - 0.01 + 0.18, 0.719, band or "",
        ha="center", va="center",
        fontsize=6.5, fontweight="bold", color=_WHITE,
        alpha=0.95, zorder=7,
    )

    # Market value (amber, below pill)
    mv = row.get("MarketValue_EUR")
    mv_y = 0.655
    if mv is not None and pd.notna(mv):
        mv_str = (fmt_market_value_fn(float(mv)) if fmt_market_value_fn
                  else f"€{float(mv)/1_000_000:.0f}M")
        ax.text(
            PAD, mv_y, mv_str,
            ha="left", va="top",
            fontsize=9, fontweight="bold", color=_AMBER, zorder=6,
        )

    # Age + League (right-aligned)
    if age_val is not None and pd.notna(age_val):
        ax.text(
            1 - PAD, 0.92, f"Age {int(age_val)}",
            ha="right", va="top",
            fontsize=9, color=_OFFWHITE, alpha=0.88, zorder=6,
        )
    if comp_clean:
        ax.text(
            1 - PAD, 0.87, comp_clean,
            ha="right", va="top",
            fontsize=8, color=_OFFWHITE, alpha=0.82, zorder=6,
        )

    # ── PLAYER NAME ───────────────────────────────────────────────────────────
    player_name = str(row.get("Player", "Unknown")).upper()
    # Truncate if too long
    if len(player_name) > 18:
        player_name = player_name[:17] + "…"
    ax.text(
        PAD, 0.615, player_name,
        ha="left", va="top",
        fontsize=14, fontweight="bold", color=_WHITE,
        zorder=6,
        path_effects=[pe.withStroke(linewidth=1.5, foreground=(0, 0, 0, 0.3))],
    )

    # Club
    squad = str(row.get("Squad", "") or "")
    ax.text(
        PAD, 0.562, squad,
        ha="left", va="top",
        fontsize=8.5, color=_OFFWHITE, alpha=0.9, zorder=6,
    )

    # Club crest — top-right corner
    crest_b = get_crest_bytes(squad)
    if crest_b:
        try:
            crest_img = np.array(Image.open(io.BytesIO(crest_b)).convert("RGBA"))
            oi = OffsetImage(crest_img, zoom=0.09)
            ab = AnnotationBbox(oi, (1 - PAD, 0.82), frameon=False, zorder=8, box_alignment=(1, 1))
            ax.add_artist(ab)
        except Exception:
            pass

    # ── DIVIDER ───────────────────────────────────────────────────────────────
    divider_y = 0.515
    ax.plot(
        [PAD, PAD + 0.55], [divider_y, divider_y],
        color="white", alpha=0.35, linewidth=0.8, zorder=6,
    )

    # ── ATTRIBUTES ───────────────────────────────────────────────────────────
    ax.text(
        PAD, 0.497, "KEY ATTRIBUTES",
        ha="left", va="top",
        fontsize=6, color=_OFFWHITE, alpha=0.75,
        fontfamily="monospace", zorder=6,
    )

    attrs = _build_attrs(row)   # same logic as render_fifa_card in app.py

    col_xs = [PAD, 0.52]           # two column x positions
    attr_y_start = 0.455
    attr_row_h = 0.092

    for i, (label, val_str) in enumerate(attrs):
        col = i % 2
        row_i = i // 2
        x = col_xs[col]
        y = attr_y_start - row_i * attr_row_h

        # Attribute box
        attr_box = mpatches.FancyBboxPatch(
            (x - 0.01, y - 0.058), 0.42, 0.068,
            boxstyle="round,pad=0.005",
            facecolor=(0, 0, 0, 0.22),
            edgecolor=(1, 1, 1, 0.18),
            linewidth=0.5, zorder=6,
        )
        ax.add_patch(attr_box)

        # Value
        ax.text(
            x + 0.07, y - 0.012, val_str,
            ha="left", va="center",
            fontsize=10, fontweight="bold", color=_WHITE, zorder=7,
        )
        # Label
        ax.text(
            x + 0.07, y - 0.040, label,
            ha="left", va="center",
            fontsize=6, color=_OFFWHITE, alpha=0.8, zorder=7,
        )

    # ── FOOTER ───────────────────────────────────────────────────────────────
    ax.text(
        PAD, 0.045, "PlayerScore",
        ha="left", va="bottom",
        fontsize=6.5, color=_OFFWHITE, alpha=0.7, zorder=6,
    )
    ax.text(
        1 - PAD, 0.045, "FBref · Big-5",
        ha="right", va="bottom",
        fontsize=6.5, color=_OFFWHITE, alpha=0.7, zorder=6,
    )

    # ── Export ────────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()
