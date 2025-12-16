# src/charts/team_scatter.py
from __future__ import annotations

import pandas as pd
import altair as alt
import streamlit as st


def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _pick_first(existing_cols: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in existing_cols:
            return c
    return None


def render_team_scatter_under_table(
    df_rank: pd.DataFrame,
    *,
    value_color: str = "#00B8A9",
    x_domain=(250, 600),
    y_domain=(0, 3),
):
    """
    Single scatter (under league table):
    - fixed x/y domains
    - top-3 labels by x (squad score)
    - dashed subtle regression line
    - footnote below chart
    """
    cols = df_rank.columns.tolist()

    x_col = _pick_first(cols, ["OverallScore_squad", "Squad Score"])
    y_col = _pick_first(cols, ["Pts/MP", "Pts", "xGD", "GD"])  # fallback supported

    if x_col is None or y_col is None:
        st.info("Scatter not available (need squad score + at least one of: Pts/MP, Pts, xGD, GD).")
        return

    keep = ["Squad", x_col, y_col] + [c for c in ["Comp", "LeagueRank", "Rank"] if c in cols]
    df = df_rank[keep].copy()
    df = _safe_numeric(df, [x_col, y_col, "LeagueRank", "Rank"])
    df = df.dropna(subset=[x_col, y_col])

    if df.empty:
        st.info("Not enough data to render the scatter.")
        return

    # highlight selected team (if exists in session_state)
    selected_team = st.session_state.get("team_scores_selected_team", None)
    df["is_selected"] = (df["Squad"] == selected_team) if selected_team else False

    # top-3 labels by x within current filtered df
    top3 = set(df.nlargest(3, x_col)["Squad"].astype(str).tolist())
    df["label_top3"] = df["Squad"].astype(str).apply(lambda s: s if s in top3 else "")

    # axis labels
    y_title_map = {"Pts/MP": "Points per match", "Pts": "Points", "xGD": "xG difference", "GD": "Goal difference"}
    x_title = "Squad score"
    y_title = y_title_map.get(y_col, y_col)

    tooltip = [
        alt.Tooltip("Squad:N", title="Team"),
        alt.Tooltip(f"{x_col}:Q", title="Squad score", format=".0f"),
        alt.Tooltip(f"{y_col}:Q", title=y_title, format=".2f" if y_col in ["Pts/MP", "xGD"] else ".0f"),
    ]
    if "Comp" in df.columns:
        tooltip.insert(1, alt.Tooltip("Comp:N", title="League"))
    if "LeagueRank" in df.columns:
        tooltip.append(alt.Tooltip("LeagueRank:Q", title="LgRk", format=".0f"))
    if "Rank" in df.columns:
        tooltip.append(alt.Tooltip("Rank:Q", title="App rank", format=".0f"))

    base = alt.Chart(df).encode(
        x=alt.X(
            f"{x_col}:Q",
            title=x_title,
            scale=alt.Scale(domain=list(x_domain)),
            axis=alt.Axis(format=".0f"),
        ),
        y=alt.Y(
            f"{y_col}:Q",
            title=y_title,
            scale=alt.Scale(domain=list(y_domain)),
            axis=alt.Axis(format=".2f" if y_col == "Pts/MP" else ".0f"),
        ),
        tooltip=tooltip,
    )

    trend = (
        base.transform_regression(x_col, y_col)
        .mark_line(strokeDash=[6, 6], strokeWidth=1.5, opacity=0.25, color="#E5E7EB")
    )

    points = base.mark_circle(size=120, opacity=0.9).encode(
        color=alt.condition(
            alt.datum.is_selected,
            alt.value(value_color),
            alt.value("#9CA3AF"),
        ),
        stroke=alt.condition(
            alt.datum.is_selected,
            alt.value("#FFFFFF"),
            alt.value("transparent"),
        ),
        strokeWidth=alt.condition(alt.datum.is_selected, alt.value(1.5), alt.value(0)),
    )

    labels = base.mark_text(
        align="left",
        dx=8,
        dy=-10,
        fontSize=12,
        fontWeight="bold",
        color=value_color,
    ).encode(text="label_top3:N")

    chart = (
        (trend + points + labels)
        .properties(height=320)
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor="#E5E7EB", titleColor="#E5E7EB", grid=False, domain=True)
    )

    st.markdown("### Score vs context")
    st.altair_chart(chart, use_container_width=True)

    st.markdown(
        "<div style='margin-top:-0.4rem; font-size:0.75rem; color:#9CA3AF;'>"
        "Creator: <b>TwinAnalytics</b> &nbsp;•&nbsp; Data: FBref / Big-5 Leagues"
        "</div>",
        unsafe_allow_html=True,
    )


def render_big5_facet_scatter(
    df_rank: pd.DataFrame,
    *,
    value_color: str = "#00B8A9",
    x_domain=(250, 600),
    y_domain=(0, 3),
    columns: int = 3,
):
    """
    Facet scatter: one panel per league (Comp).
    Fixed axes, dashed subtle regression line, top-3 labels per league.
    Requires: Comp + Pts/MP.
    """
    cols = df_rank.columns.tolist()

    if "Comp" not in cols:
        st.info("Facet scatter needs 'Comp' column.")
        return

    x_col = _pick_first(cols, ["OverallScore_squad", "Squad Score"])
    y_col = "Pts/MP" if "Pts/MP" in cols else None
    if x_col is None or y_col is None:
        st.info("Facet scatter needs squad score + 'Pts/MP'.")
        return

    df = df_rank[["Comp", "Squad", x_col, y_col]].copy()
    df = _safe_numeric(df, [x_col, y_col])
    df = df.dropna(subset=[x_col, y_col])

    if df.empty:
        st.info("Not enough data for facet scatter.")
        return

    # top-3 per league by x
    df["rank_in_comp"] = df.groupby("Comp")[x_col].rank(method="first", ascending=False)
    df["label_top3"] = df.apply(lambda r: r["Squad"] if r["rank_in_comp"] <= 3 else "", axis=1)

    base = alt.Chart(df).encode(
        x=alt.X(
            f"{x_col}:Q",
            scale=alt.Scale(domain=list(x_domain)),
            title="Squad score",
            axis=alt.Axis(format=".0f"),
        ),
        y=alt.Y(
            f"{y_col}:Q",
            scale=alt.Scale(domain=list(y_domain)),
            title="Pts/MP",
            axis=alt.Axis(format=".2f"),
        ),
        tooltip=[
            alt.Tooltip("Comp:N", title="League"),
            alt.Tooltip("Squad:N", title="Team"),
            alt.Tooltip(f"{x_col}:Q", title="Squad score", format=".0f"),
            alt.Tooltip(f"{y_col}:Q", title="Pts/MP", format=".2f"),
        ],
    )

    trend = (
        base.transform_regression(x_col, y_col)
        .mark_line(strokeDash=[6, 6], strokeWidth=1.5, opacity=0.25, color="#E5E7EB")
    )
    points = base.mark_circle(size=140, opacity=0.9, color="#9CA3AF")
    labels = base.mark_text(
        align="left",
        dx=8,
        dy=-10,
        fontSize=12,
        fontWeight="bold",
        color=value_color,
    ).encode(text="label_top3:N")

    facet = (
        alt.layer(trend, points, labels)
        .facet(facet=alt.Facet("Comp:N", title=None), columns=columns)
        .resolve_scale(x="shared", y="shared")
    )

    chart = (
        facet.properties(height=260)
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor="#E5E7EB", titleColor="#E5E7EB", grid=False, domain=True)
        .configure_header(labelColor="#E5E7EB", titleColor="#E5E7EB", labelFontSize=13)
    )

    st.markdown("### Big-5: Squad score vs Points per match")
    st.altair_chart(chart, use_container_width=True)

    st.markdown(
        "<div style='margin-top:-0.4rem; font-size:0.75rem; color:#9CA3AF;'>"
        "Creator: <b>TwinAnalytics</b> &nbsp;•&nbsp; Data: FBref / Big-5 Leagues"
        "</div>",
        unsafe_allow_html=True,
    )


def render_scatter_linkedin_optimized(
    df_rank: pd.DataFrame,
    *,
    value_color: str = "#00B8A9",
    x_domain=(250, 600),
    y_domain=(0, 3),
    show_axis_titles: bool = False,
):
    """
    Single scatter optimized for screenshot / LinkedIn:
    - bigger points
    - more whitespace
    - fixed axes
    - dashed subtle regression
    - top-3 labels
    Requires: Pts/MP.
    """
    cols = df_rank.columns.tolist()

    x_col = _pick_first(cols, ["OverallScore_squad", "Squad Score"])
    y_col = "Pts/MP" if "Pts/MP" in cols else None
    if x_col is None or y_col is None:
        st.info("LinkedIn scatter needs squad score + 'Pts/MP'.")
        return

    df = df_rank[["Squad", x_col, y_col] + (["Comp"] if "Comp" in cols else [])].copy()
    df = _safe_numeric(df, [x_col, y_col])
    df = df.dropna(subset=[x_col, y_col])

    if df.empty:
        st.info("Not enough data for scatter.")
        return

    top3 = set(df.nlargest(3, x_col)["Squad"].astype(str).tolist())
    df["label_top3"] = df["Squad"].astype(str).apply(lambda s: s if s in top3 else "")

    base = alt.Chart(df).encode(
        x=alt.X(
            f"{x_col}:Q",
            scale=alt.Scale(domain=list(x_domain)),
            title=("Squad score" if show_axis_titles else None),
            axis=alt.Axis(format=".0f"),
        ),
        y=alt.Y(
            f"{y_col}:Q",
            scale=alt.Scale(domain=list(y_domain)),
            title=("Pts/MP" if show_axis_titles else None),
            axis=alt.Axis(format=".2f"),
        ),
        tooltip=[
            alt.Tooltip("Squad:N", title="Team"),
            alt.Tooltip(f"{x_col}:Q", title="Squad score", format=".0f"),
            alt.Tooltip(f"{y_col}:Q", title="Pts/MP", format=".2f"),
        ],
    )

    trend = (
        base.transform_regression(x_col, y_col)
        .mark_line(strokeDash=[7, 7], strokeWidth=1.8, opacity=0.22, color="#E5E7EB")
    )
    points = base.mark_circle(size=260, opacity=0.85, color="#9CA3AF")
    labels = base.mark_text(
        align="left",
        dx=10,
        dy=-12,
        fontSize=14,
        fontWeight="bold",
        color=value_color,
    ).encode(text="label_top3:N")

    chart = (
        alt.layer(trend, points, labels)
        .properties(height=420)
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor="#E5E7EB", titleColor="#E5E7EB", grid=False, domain=True)
    )

    st.markdown("### Squad score vs Pts/MP")
    st.altair_chart(chart, use_container_width=True)

    st.markdown(
        "<div style='margin-top:-0.4rem; font-size:0.75rem; color:#9CA3AF;'>"
        "Creator: <b>TwinAnalytics</b> &nbsp;•&nbsp; Data: FBref / Big-5 Leagues"
        "</div>",
        unsafe_allow_html=True,
    )