"""
Age-curve chart: score development vs. role peers by age.

Grey dashed line  — median MainScore for role peers across all seasons.
Teal line + dots  — selected player's actual score at each age.
"""

from __future__ import annotations

from typing import Callable

import altair as alt
import pandas as pd
import streamlit as st


def render_age_curve(
    df_all: pd.DataFrame,
    player: str,
    role: str,
    get_score_fn: Callable,
    value_color: str = "#00B8A9",
) -> None:
    """Render the age-curve chart inside the current Streamlit container.

    Parameters
    ----------
    df_all:
        Full multi-season DataFrame (all players, all seasons).
    player:
        Name of the selected player.
    role:
        Positional role string used to filter peers (``"FW"``, ``"MF"``, …).
    get_score_fn:
        Row-wise callable that returns ``(MainScore, MainBand)`` for a
        given row — e.g. ``get_primary_score_and_band`` from app.py.
    value_color:
        Hex colour for the player trajectory line/dots.
    """
    df = df_all.copy()

    # Compute MainScore / MainBand if not already present
    if "MainScore" not in df.columns or "MainBand" not in df.columns:
        df[["MainScore", "MainBand"]] = df.apply(
            get_score_fn, axis=1, result_type="expand"
        )

    df["Age"] = pd.to_numeric(df.get("Age", pd.Series(dtype=float)), errors="coerce")
    df = df.dropna(subset=["Age", "MainScore"])
    df["Age"] = df["Age"].round().astype(int)

    # --- Peer median curve (same role) ---
    pos_col = "Pos_raw" if "Pos_raw" in df.columns else "Pos"
    peers = df[df[pos_col] == role] if pos_col in df.columns else df
    peer_curve = (
        peers.groupby("Age")["MainScore"]
        .median()
        .reset_index()
        .rename(columns={"MainScore": "MedianScore"})
    )

    # --- Player trajectory ---
    player_data = df[df["Player"] == player].sort_values("Age")

    if player_data.empty:
        st.caption("No age data available.")
        return

    peer_line = (
        alt.Chart(peer_curve)
        .mark_line(color="#6b7280", strokeDash=[5, 3], strokeWidth=1.5)
        .encode(
            x=alt.X("Age:Q", title="Age", scale=alt.Scale(zero=False)),
            y=alt.Y(
                "MedianScore:Q",
                title="Score",
                scale=alt.Scale(domain=[0, 1000]),
            ),
            tooltip=[
                alt.Tooltip("Age:Q"),
                alt.Tooltip("MedianScore:Q", format=".0f", title="Role median"),
            ],
        )
    )

    player_line = (
        alt.Chart(player_data)
        .mark_line(color=value_color, strokeWidth=2.5)
        .encode(x="Age:Q", y="MainScore:Q")
    )

    tooltip_fields = [
        alt.Tooltip("Age:Q"),
        alt.Tooltip("MainScore:Q", format=".0f", title="Score"),
    ]
    if "Season" in player_data.columns:
        tooltip_fields.append("Season:N")
    if "Squad" in player_data.columns:
        tooltip_fields.append("Squad:N")

    player_dots = (
        alt.Chart(player_data)
        .mark_circle(color=value_color, size=65)
        .encode(x="Age:Q", y="MainScore:Q", tooltip=tooltip_fields)
    )

    chart = (
        (peer_line + player_line + player_dots)
        .properties(
            height=230,
            title=alt.TitleParams(
                "Score development vs. role peers",
                fontSize=12,
                color="#e5e7eb",
            ),
        )
        .configure_view(strokeWidth=0, fill="#0D1117")
        .configure_axis(
            gridColor="#374151",
            labelColor="#e5e7eb",
            titleColor="#9ca3af",
            domainColor="#374151",
        )
        .configure_title(color="#e5e7eb")
        .interactive()
    )

    st.markdown("#### Age curve")
    st.altair_chart(chart, use_container_width=True)
    st.caption("— Grey dashed: median for role peers  ·  Teal: selected player")
