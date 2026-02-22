"""
Generates a professional A4 PDF player report using fpdf2.
Light-themed (white background) for print readability.

Layout:
  Page 1 - Header | Name/meta | FIFA card + Score / Sub-scores / Season stats
           Summary | Pizza chart + Scatter plot side by side
"""
import io
from datetime import date

import pandas as pd
from fpdf import FPDF


def _t(text: str) -> str:
    """Sanitise text for fpdf2 Helvetica (Latin-1 encoding)."""
    return (str(text)
            .replace("\u2014", "-")
            .replace("\u2013", "-")
            .replace("\u00b7", "|")
            .replace("\u20ac", "EUR")
            .encode("latin-1", errors="replace").decode("latin-1"))


# ── Colour palette (light-themed for print) ───────────────────────────────────
_HDR_BG    = (11, 31, 30)
_HDR_FG    = (255, 255, 255)
_TEAL      = (0, 184, 169)
_BODY      = (20, 20, 20)
_GREY      = (100, 100, 100)
_DIVIDER   = (220, 220, 220)
_TILE_BG   = (244, 249, 249)
_SUMM_BG   = (237, 247, 246)
_GOLD      = (155, 105, 0)

_BAND_RGB = {
    "Exceptional":        (120, 40, 200),
    "World Class":        (22, 160, 80),
    "Top Starter":        (37, 99, 210),
    "Solid Squad Player": (170, 110, 0),
    "Below Big-5 Level":  (90, 95, 105),
}

_AREA_LABELS = {
    "OffScore_abs": "attacking output",
    "MidScore_abs": "midfield contribution",
    "DefScore_abs": "defensive work",
}

_ROLE_LABEL = {
    "FW": "Forwards", "Off_MF": "Forwards / Off. MF",
    "MF": "Central Midfielders",
    "DF": "Defenders", "Def_MF": "Defenders / Def. MF",
}


def _resolve(row: pd.Series, candidates: list[str]) -> float | None:
    for c in candidates:
        if c in row.index and pd.notna(row[c]):
            try:
                return float(row[c])
            except (ValueError, TypeError):
                pass
    return None


def _fig_to_png(fig, dpi: int = 160) -> bytes:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _build_role_scatter_png(
    df_comp: pd.DataFrame,
    df_player: pd.DataFrame,
    role: str,
) -> bytes | None:
    """Matplotlib scatter plot matching the in-app render_role_scatter logic."""
    import matplotlib.pyplot as plt

    if df_player is None or df_player.empty:
        return None

    row = df_player.iloc[0]
    player_name = str(row.get("Player", ""))

    if role in ("FW", "Off_MF"):
        x_candidates = ["xG_Per90", "xG/90"]
        y_candidates = ["Gls_Per90", "Gls/90"]
        x_label = "xG per 90"
        y_label = "Goals per 90"
        title    = "FW - xG vs Goals per 90"
    elif role == "MF":
        x_candidates = ["xAG_Per90", "xAG/90", "xA_Per90", "xA/90"]
        y_candidates = ["Ast_Per90", "Ast/90"]
        x_label = "xAG per 90"
        y_label = "Assists per 90"
        title    = "MF - xAG vs Assists per 90"
    else:  # DF, Def_MF
        x_candidates = ["Int_Per90", "Int/90"]
        y_candidates = ["TklW_Per90", "TklW/90"]
        x_label = "Interceptions per 90"
        y_label = "Tackles won per 90"
        title    = "DF - Interceptions vs Tackles per 90"

    def _resolve_col(candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df_comp.columns:
                return c
        return None

    x_col = _resolve_col(x_candidates)
    y_col = _resolve_col(y_candidates)

    if x_col is None or y_col is None:
        return None

    needed = [c for c in ["Player", x_col, y_col] if c in df_comp.columns]
    plot_df = df_comp[needed].copy()
    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_col, y_col])

    if plot_df.empty:
        return None

    # Player coords
    player_rows = plot_df[plot_df["Player"] == player_name] if "Player" in plot_df.columns else pd.DataFrame()
    if player_rows.empty:
        # Try from df_player directly
        px = _resolve(row, x_candidates)
        py = _resolve(row, y_candidates)
        if px is None or py is None:
            return None
    else:
        px = float(player_rows[x_col].iloc[0])
        py = float(player_rows[y_col].iloc[0])

    # Clip peers at 99th percentile (but always keep player)
    qx = plot_df[x_col].quantile(0.99)
    qy = plot_df[y_col].quantile(0.99)
    peers = plot_df[(plot_df[x_col] <= qx) & (plot_df[y_col] <= qy)]

    fig, ax = plt.subplots(figsize=(3.8, 3.0), facecolor="white")
    ax.set_facecolor("#F8FAFA")

    # Peers
    ax.scatter(peers[x_col], peers[y_col], s=10, alpha=0.15, color="#6B7280", zorder=2)

    # Player highlight
    ax.scatter([px], [py], s=70, color="#00B8A9", zorder=5,
               edgecolors="white", linewidths=0.8)
    ax.annotate(
        _t(player_name), (px, py),
        xytext=(5, 5), textcoords="offset points",
        fontsize=5.5, color="#0B1F1E", zorder=6,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                  edgecolor="#00B8A9", linewidth=0.4, alpha=0.85),
    )

    ax.set_xlabel(x_label, fontsize=6.5, color="#374151")
    ax.set_ylabel(y_label, fontsize=6.5, color="#374151")
    ax.set_title(title, fontsize=7.5, fontweight="bold", color="#111827", pad=3)
    ax.tick_params(labelsize=5.5, colors="#6B7280")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#D1D5DB")
    ax.grid(True, alpha=0.3, linewidth=0.4, color="#D1D5DB")

    fig.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _generate_summary(
    row: pd.Series,
    score: float,
    band: str,
    fmt_market_value_fn=None,
) -> str:
    player     = _t(str(row.get("Player", "Unknown")))
    pos        = _t(str(row.get("Pos", "") or "").upper())
    squad      = _t(str(row.get("Squad", "") or ""))
    comp_raw   = str(row.get("Comp", "") or "")
    comp       = _t(comp_raw.replace("eng ", "").replace("es ", "")
                    .replace("de ", "").replace("it ", "").replace("fr ", ""))
    season_val = _t(str(row.get("Season", "") or ""))
    age_val    = row.get("Age")
    score_int  = int(round(score))

    band_desc = {
        "Exceptional":        "among the elite players",
        "World Class":        "among the top performers",
        "Top Starter":        "a reliable starter",
        "Solid Squad Player": "a solid squad contributor",
        "Below Big-5 Level":  "still developing at this level",
    }.get(band, "a notable player")

    # Determine strongest area from sub-scores
    sub = {col: _resolve(row, [col]) for col in _AREA_LABELS}
    sub = {k: v for k, v in sub.items() if v is not None}
    strongest_col  = max(sub, key=sub.get) if sub else None
    strongest_val  = sub[strongest_col] if strongest_col else None
    strongest_name = _AREA_LABELS.get(strongest_col, "") if strongest_col else ""

    mins_90 = _resolve(row, ["90s"])
    age_str = f", {int(age_val)} years old," if age_val and pd.notna(age_val) else ""

    s1 = f"{player} is a {pos}{age_str} playing for {squad} in the {comp}."
    if season_val:
        s2 = (f"In the {season_val} season, he achieved a PlayerScore of {score_int} "
              f"({band}) - {band_desc} across the Big-5 leagues.")
    else:
        s2 = (f"Across his career, he achieved a PlayerScore of {score_int} "
              f"({band}) - {band_desc} across the Big-5 leagues.")

    s3 = ""
    if strongest_name and strongest_val is not None:
        s3 = (f"His standout area is his {strongest_name}, "
              f"with a sub-score of {int(round(strongest_val))} out of 1000.")

    s4 = ""
    if mins_90 is not None:
        s4 = f"He has accumulated {mins_90:.1f} 90s of playing time this season."

    s5 = ""
    mv = row.get("MarketValue_EUR")
    if mv is not None and pd.notna(mv):
        if fmt_market_value_fn:
            mv_str = _t(fmt_market_value_fn(float(mv)))
        else:
            mv_str = f"EUR {float(mv) / 1_000_000:.1f}M"
        s5 = f"His current market value is estimated at {mv_str}."

    return "  ".join(s for s in [s1, s2, s3, s4, s5] if s)


def _footer(pdf: FPDF) -> None:
    pdf.set_draw_color(*_DIVIDER)
    pdf.line(14, 284, 196, 284)
    pdf.set_xy(14, 286)
    pdf.set_font("Helvetica", size=7)
    pdf.set_text_color(*_GREY)
    pdf.cell(91, 4, "Data: FBref | Big-5 European Leagues | Transfermarkt")
    pdf.set_xy(105, 286)
    pdf.cell(91, 4, "PlayerScore", align="R")


def generate_player_report_pdf(
    row: pd.Series,
    score: float,
    band: str,
    card_png_bytes: bytes,
    pizza_fig=None,
    scatter_df_all: pd.DataFrame | None = None,
    scatter_df_player: pd.DataFrame | None = None,
    peer_percentile: dict | None = None,
    fmt_market_value_fn=None,
) -> bytes:
    """Return an A4 PDF player report as bytes."""
    today = date.today().strftime("%d %b %Y")
    role  = str(row.get("Pos_raw", row.get("Pos", "MF")))

    pdf = FPDF(format="A4")
    pdf.set_margins(14, 14, 14)
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    # ── Header bar ─────────────────────────────────────────────────────────────
    pdf.set_fill_color(*_HDR_BG)
    pdf.rect(0, 0, 210, 16, style="F")
    pdf.set_font("Helvetica", style="B", size=10)
    pdf.set_text_color(*_HDR_FG)
    pdf.set_xy(14, 4.5)
    pdf.cell(91, 7, "PlayerScore - Player Report")
    pdf.set_font("Helvetica", size=8)
    pdf.set_xy(105, 4.5)
    pdf.cell(91, 7, _t(f"Generated: {today}"), align="R")

    # ── Player name + meta ─────────────────────────────────────────────────────
    player_name = _t(str(row.get("Player", "Unknown")))
    squad       = _t(str(row.get("Squad", "") or ""))
    comp_raw    = str(row.get("Comp", "") or "")
    comp_clean  = _t(comp_raw.replace("eng ", "").replace("es ", "")
                     .replace("de ", "").replace("it ", "").replace("fr ", ""))
    pos         = _t(str(row.get("Pos", row.get("Pos_raw", "")) or "").upper())
    age_val     = row.get("Age")
    season_val  = _t(str(row.get("Season", "") or ""))

    meta_parts = [p for p in [
        squad, comp_clean, pos,
        f"Age {int(age_val)}" if age_val and pd.notna(age_val) else "",
        f"Season {season_val}" if season_val else "",
    ] if p]

    pdf.set_xy(14, 20)
    pdf.set_font("Helvetica", style="B", size=20)
    pdf.set_text_color(*_BODY)
    pdf.cell(182, 9, player_name)

    pdf.set_xy(14, 30)
    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(*_GREY)
    pdf.cell(182, 5, "  |  ".join(meta_parts))

    pdf.set_draw_color(*_DIVIDER)
    pdf.set_line_width(0.3)
    pdf.line(14, 38, 196, 38)

    # ── Left: FIFA card ────────────────────────────────────────────────────────
    CARD_X, CARD_Y = 14, 41
    CARD_W = 58
    CARD_H = round(CARD_W * 3.8 / 2.7)   # ~81 mm
    pdf.image(io.BytesIO(card_png_bytes), x=CARD_X, y=CARD_Y, w=CARD_W)

    # ── Right: Score + band + market value + sub-scores + season stats ─────────
    RX  = 78
    RW  = 118
    CY  = CARD_Y

    score_int = int(round(score))
    pdf.set_xy(RX, CY)
    pdf.set_font("Helvetica", style="B", size=44)
    pdf.set_text_color(*_TEAL)
    pdf.cell(60, 18, str(score_int))

    pdf.set_xy(RX + 62, CY + 4)
    pdf.set_font("Helvetica", size=7)
    pdf.set_text_color(*_GREY)
    pdf.cell(40, 4, "PlayerScore")
    pdf.set_xy(RX + 62, CY + 9)
    pdf.cell(40, 4, "Range: 0 - 1000")

    band_rgb = _BAND_RGB.get(band, (90, 95, 105))
    pdf.set_xy(RX, CY + 19)
    pdf.set_font("Helvetica", style="B", size=11)
    pdf.set_text_color(*band_rgb)
    pdf.cell(RW, 6, _t(band or "-"))

    cur_y = CY + 28
    mv = row.get("MarketValue_EUR")
    if mv is not None and pd.notna(mv):
        if fmt_market_value_fn:
            mv_str = _t(fmt_market_value_fn(float(mv)))
        else:
            mv_str = f"EUR {float(mv) / 1_000_000:.1f}M"
        pdf.set_xy(RX, cur_y)
        pdf.set_font("Helvetica", style="B", size=9)
        pdf.set_text_color(*_GOLD)
        pdf.cell(RW, 5, f"Market Value: {mv_str}")
        cur_y += 7

    # Sub-score breakdown in right col
    pdf.set_draw_color(*_DIVIDER)
    pdf.line(RX, cur_y, RX + RW, cur_y)
    cur_y += 3

    pdf.set_xy(RX, cur_y)
    pdf.set_font("Helvetica", style="B", size=6.5)
    pdf.set_text_color(*_GREY)
    pdf.cell(RW, 4, "SCORE BREAKDOWN")
    cur_y += 5

    breakdowns = [
        ("Offense",  ["OffScore_abs"]),
        ("Midfield", ["MidScore_abs"]),
        ("Defense",  ["DefScore_abs"]),
    ]
    BAR_MAX_W = RW - 36   # label(26) + value(10)
    for lbl, cols in breakdowns:
        val = _resolve(row, cols)
        if val is None:
            continue

        # Label
        pdf.set_xy(RX, cur_y)
        pdf.set_font("Helvetica", size=7.5)
        pdf.set_text_color(*_GREY)
        pdf.cell(26, 5, lbl)

        # Value
        pdf.set_font("Helvetica", style="B", size=8)
        pdf.set_text_color(*_BODY)
        pdf.set_xy(RX + 26, cur_y)
        pdf.cell(10, 5, str(int(round(val))), align="R")

        # Bar track + fill
        bar_x = RX + 36
        bar_y = cur_y + 1
        bar_w = min(float(val) / 1000 * BAR_MAX_W, BAR_MAX_W)

        pdf.set_fill_color(225, 235, 235)
        pdf.rect(bar_x, bar_y, BAR_MAX_W, 3.5, style="F")
        pdf.set_fill_color(*_TEAL)
        pdf.rect(bar_x, bar_y, max(bar_w, 0.5), 3.5, style="F")

        cur_y += 9

    # Peer ranking below sub-scores
    if peer_percentile is not None:
        pdf.set_draw_color(*_DIVIDER)
        pdf.line(RX, cur_y, RX + RW, cur_y)
        cur_y += 3

        pdf.set_xy(RX, cur_y)
        pdf.set_font("Helvetica", style="B", size=6.5)
        pdf.set_text_color(*_GREY)
        pdf.cell(RW, 4, "PEER RANKING")
        cur_y += 5

        _pct      = peer_percentile["percentile"]          # fraction below player
        _top_pct  = int(round((1 - _pct) * 100))          # e.g. 12 → "Top 12%"
        _top_pct  = max(1, min(99, _top_pct))
        _n        = peer_percentile["n_peers"]
        _rl       = _ROLE_LABEL.get(peer_percentile["role"], peer_percentile["role"])
        _season_s = peer_percentile.get("season") or ""

        # "Top X%" — large
        pdf.set_xy(RX, cur_y)
        pdf.set_font("Helvetica", style="B", size=18)
        pdf.set_text_color(*_TEAL)
        pdf.cell(RW, 8, f"Top {_top_pct}%")
        cur_y += 9

        # Context line
        ctx = f"of {_n} {_rl} in Big-5"
        if _season_s:
            ctx += f" ({_season_s})"
        pdf.set_xy(RX, cur_y)
        pdf.set_font("Helvetica", size=7)
        pdf.set_text_color(*_GREY)
        pdf.cell(RW, 4, _t(ctx))
        cur_y += 5

        # Percentile bar
        BAR_FULL = RW
        bar_fill = BAR_FULL * _pct        # percentile as filled fraction
        pdf.set_fill_color(225, 235, 235)
        pdf.rect(RX, cur_y, BAR_FULL, 4, style="F")
        pdf.set_fill_color(*_TEAL)
        pdf.rect(RX, cur_y, max(bar_fill, 1), 4, style="F")
        cur_y += 6

        # Percentile label below bar
        pdf.set_xy(RX, cur_y)
        pdf.set_font("Helvetica", size=6.5)
        pdf.set_text_color(*_GREY)
        pdf.cell(RW, 3, f"Ranked higher than {int(round(_pct * 100))}% of peers")

    # ── Auto-generated summary box ─────────────────────────────────────────────
    below_card_y = CARD_Y + CARD_H + 4

    summary_text = _generate_summary(row, score, band, fmt_market_value_fn)

    SUMM_PAD = 4
    SUMM_W   = 182
    LINE_H   = 5.2
    pdf.set_font("Helvetica", size=8.5)

    n_lines = len(pdf.multi_cell(SUMM_W - 2 * SUMM_PAD, LINE_H,
                                 summary_text, dry_run=True, output="LINES"))
    summ_h  = n_lines * LINE_H + 2 * SUMM_PAD + 4

    summ_y = below_card_y + 2
    pdf.set_fill_color(*_SUMM_BG)
    pdf.rect(14, summ_y, SUMM_W, summ_h, style="F")
    pdf.set_draw_color(0, 184, 169)
    pdf.set_line_width(0.6)
    pdf.line(14, summ_y, 14, summ_y + summ_h)
    pdf.set_line_width(0.3)

    pdf.set_xy(14 + SUMM_PAD, summ_y + 2)
    pdf.set_font("Helvetica", style="B", size=7)
    pdf.set_text_color(*_TEAL)
    pdf.cell(SUMM_W - 2 * SUMM_PAD, 4, "SCOUTING SUMMARY")

    pdf.set_xy(14 + SUMM_PAD, summ_y + 6)
    pdf.set_font("Helvetica", size=8.5)
    pdf.set_text_color(*_BODY)
    pdf.multi_cell(SUMM_W - 2 * SUMM_PAD, LINE_H, summary_text)

    next_y = summ_y + summ_h + 6

    # ── Charts: Pizza + Scatter side by side ───────────────────────────────────
    has_pizza   = pizza_fig is not None
    scatter_png = None
    if scatter_df_all is not None and scatter_df_player is not None:
        scatter_png = _build_role_scatter_png(scatter_df_all, scatter_df_player, role)

    if has_pizza or scatter_png is not None:
        # Section label
        if next_y + 80 > 282:
            _footer(pdf)
            pdf.add_page()
            next_y = 18

        pdf.set_draw_color(*_DIVIDER)
        pdf.line(14, next_y, 196, next_y)
        next_y += 2

        pdf.set_xy(14, next_y)
        pdf.set_font("Helvetica", style="B", size=10)
        pdf.set_text_color(*_BODY)
        pdf.cell(182, 5, "Performance Profile vs. Peers", align="C")
        next_y += 8

        if has_pizza and scatter_png is not None:
            # Side by side: each 88mm with 6mm gap
            CHART_W = 88
            GAP     = 6
            pizza_png_bytes = _fig_to_png(pizza_fig, dpi=150)
            pdf.image(io.BytesIO(pizza_png_bytes), x=14,               y=next_y, w=CHART_W)
            pdf.image(io.BytesIO(scatter_png),     x=14 + CHART_W + GAP, y=next_y, w=CHART_W)
        elif has_pizza:
            # Pizza centred, no scatter
            pizza_png_bytes = _fig_to_png(pizza_fig, dpi=150)
            PIZZA_W = 110
            PIZZA_X = (210 - PIZZA_W) / 2
            pdf.image(io.BytesIO(pizza_png_bytes), x=PIZZA_X, y=next_y, w=PIZZA_W)
        elif scatter_png is not None:
            SCATTER_W = 110
            SCATTER_X = (210 - SCATTER_W) / 2
            pdf.image(io.BytesIO(scatter_png), x=SCATTER_X, y=next_y, w=SCATTER_W)

    _footer(pdf)
    return bytes(pdf.output())
