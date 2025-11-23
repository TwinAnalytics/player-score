import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

PRIMARY_COLOR = "#1f77b4"  # main brand color (used for headline & band chart)

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

# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="PlayerScore – Big-5 Player Rating",
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
        "Off_MF": "MF",
        "MF": "MF",
        "Def_MF": "MF",
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


    # ==================================================================
    # MODE 0: HOME / LANDING PAGE
    # ==================================================================
    if mode == "Home":
        # Hero section
        st.markdown(
            f"""
            <h1 style="color:{PRIMARY_COLOR}; margin-bottom:0.25rem;">PlayerScore</h1>
            <p style="font-size:1.05rem; margin-top:0;">
                Advanced football player analytics across leagues and seasons
            </p>
            """,
            unsafe_allow_html=True,
        )

        # What is PlayerScore?
        st.markdown(
             f"""
             <h3 style="color:{PRIMARY_COLOR};">What is PlayerScore?</h3>
            """,
            unsafe_allow_html=True,
        )
        # Intro sections
        st.markdown(
            """
            PlayerScore is built on a self-developed, data-driven scoring framework that makes it possible 
            to compare football players across different leagues — regardless of country, competition level, 
            or data availability.

            Our analysis engine continuously processes the latest publicly available performance data from FBref 
            and combines it with a custom scoring logic that uses comprehensive Big-5 metrics as well as reduced 
            “Light” metrics for leagues with fewer statistical features.
            """
        )

        # What does PlayerScore deliver?
        st.markdown(
             f"""
            <h3 style="color:{PRIMARY_COLOR};">What does PlayerScore deliver?</h3>
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
            The PlayerScore database already contains several thousand players from Europe’s top leagues, 
            the 2. Bundesliga, and multiple historical seasons — and it grows with every pipeline run.
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

        # ----- Player direkt unter View -----
        players_all = sorted(df_all["Player"].dropna().unique())
        if not players_all:
            st.warning("No players found in the dataset.")
            return

        # Placeholder/Leere Auswahl
        placeholder = "Select a player..."
        player_options = [placeholder] + players_all

        # aktuellen Wert aus Session-State holen (oder Placeholder)
        current_selection = st.session_state.get("selected_player", placeholder)
        if current_selection not in player_options:
            current_selection = placeholder

        player = st.sidebar.selectbox(
            "Player",
            player_options,
            index=player_options.index(current_selection),
        )
        st.session_state["selected_player"] = player

        # Solange kein Spieler ausgewählt ist -> Hinweis anzeigen und abbrechen
        if player == placeholder:
            st.subheader("Player profile")
            st.info("Bitte links im Sidebar einen Spieler auswählen, um das Profil zu sehen.")
            return

        # Ab hier ist 'player' ein echter Spielername
        df_player_all = df_all[df_all["Player"] == player].copy()


        if "Season" in df_player_all.columns:
            seasons = sorted(df_player_all["Season"].dropna().unique())
        else:
            seasons = sorted(df_all["Season"].dropna().unique())

        if not seasons:
            st.warning("No seasons found for this player.")
            return

        # ----- Season unterhalb Player -----
        season = st.sidebar.selectbox(
            "Season",
            seasons,
            key="profile_season",
        )

        # Aktuelle Ansicht (Per season vs Career) etwas weiter unten
        st.sidebar.markdown("---")
        profile_view = st.sidebar.radio(
            "Profile view",
            ["Per season", "Career"],
        )

        # ---------- Player profile ----------
        st.subheader(f"Player profile – {player}")

        # DataFrames für diese Ansicht
        # df_player_all bleibt alle Saisons des Spielers (für Rolle + Trend)
        if profile_view == "Per season":
            df_player = df_player_all[df_player_all["Season"] == season].copy()
        else:
            df_player = df_player_all.copy()

        # Alle Saisons dieses Spielers (für Rolle + Trend)
        df_player_all = df_all[df_all["Player"] == player].copy()

        # Aktuelle Ansicht
        if profile_view == "Per season":
            df_player = df_player_all[df_player_all["Season"] == season].copy()
        else:
            df_player = df_player_all.copy()

        # Typical role (based on all seasons)
        typical_pos = df_player_all["Pos"].dropna().mode()
        typical_pos = typical_pos.iloc[0] if not typical_pos.empty else None
        st.caption(get_role_label(typical_pos))

        # Decide primary score dimension based on role (for squad assessment)
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

        primary_squad_info = None
        if (
            profile_view == "Per season"
            and primary_dim is not None
            and score_col is not None
            and squad_col is not None
            and not df_player.empty
            and not df_squad.empty
            and "Squad" in df_player.columns
        ):
            squad_name = df_player["Squad"].iloc[0]
            df_squad_row = df_squad[
                (df_squad["Season"] == season) & (df_squad["Squad"] == squad_name)
            ]

            if not df_squad_row.empty and squad_col in df_squad_row.columns:
                player_score = df_player[score_col].iloc[0]
                squad_score = df_squad_row[squad_col].iloc[0]

                if pd.notna(player_score) and pd.notna(squad_score):
                    diff = float(player_score) - float(squad_score)
                    assessment = assess_diff(diff)
                    primary_squad_info = {
                        "dimension": primary_dim,
                        "player_score": float(player_score),
                        "squad_score": float(squad_score),
                        "diff": diff,
                        "assessment": assessment,
                        "squad_name": squad_name,
                    }

        # Scores for current view (primary role score, same logic as Top Lists)
        st.markdown("### Scores")

        # Ensure primary score + band exist for this view
        if not df_player.empty:
            df_player = df_player.copy()
            df_player[["MainScore", "MainBand"]] = df_player.apply(
                get_primary_score_and_band,
                axis=1,
                result_type="expand",
            )

        cols_show = [
            "Season", "Squad", "Comp", "Pos", "Age", "Min", "90s",
            "MainScore", "MainBand",
        ]
        existing_cols = [c for c in cols_show if c in df_player.columns]

        if existing_cols and not df_player.empty:
            df_scores = df_player[existing_cols].sort_values("Season")

            # Replace band with icons (like in Top Lists)
            display_cols = [c for c in cols_show if c not in ("MainBand",)]
            display_cols = [c for c in display_cols if c in df_scores.columns]

            if "MainBand" in df_scores.columns:
                df_scores["Band"] = df_scores["MainBand"].map(BAND_ICONS).fillna(df_scores["MainBand"])
                display_cols.append("Band")

            df_scores_display = df_scores[display_cols].rename(columns={"MainScore": "Score"})

            # 👉 Score als Integer ohne Nachkommastellen anzeigen
            if "Score" in df_scores_display.columns:
                df_scores_display["Score"] = (
                    df_scores_display["Score"]
                    .round()
                    .astype("Int64")  # nullable int, falls doch mal NaN auftaucht
                )

            st.dataframe(df_scores_display, use_container_width=True)

        else:
            st.info("No season-level data available for this player and view.")


        # Summary metrics
        st.markdown("### Summary")

        col1, col2, col3, col4 = st.columns(4)

        total_seasons = df_player["Season"].nunique() if "Season" in df_player.columns else 0
        total_minutes = int(df_player["Min"].sum()) if "Min" in df_player.columns else 0
        total_90s = float(df_player["90s"].sum()) if "90s" in df_player.columns else 0.0

        with col1:
            st.metric("Seasons in view", total_seasons)

        with col2:
            st.metric("Total minutes", total_minutes)

        with col3:
            st.metric("Total 90s", f"{total_90s:.1f}")

        # 👉 Average Score nur in der Career-Ansicht anzeigen
        if profile_view == "Career":
            avg_score = None
            if "MainScore" in df_player.columns and df_player["MainScore"].notna().any():
                avg_score = df_player["MainScore"].mean()

            with col4:
                if avg_score is not None:
                    # gerundet, ohne Nachkommastellen
                    st.metric("Average score (career)", f"{avg_score:.0f}")
                else:
                    st.metric("Average score (career)", "n/a")


        # Squad score + verbal assessment for the primary dimension
        if primary_squad_info is not None:
            dim = primary_squad_info["dimension"]
            st.markdown(
                f"**Squad fit ({dim}):** "
                f"{primary_squad_info['player_score']:.0f} vs squad average "
                f"{primary_squad_info['squad_score']:.0f} "
                f"({primary_squad_info['diff']:+.0f}) → "
                f"{primary_squad_info['assessment']}."
            )

        # Squad comparison (this season)
        if profile_view == "Per season" and not df_player.empty and not df_squad.empty:
            squad_name = df_player["Squad"].iloc[0] if "Squad" in df_player.columns else None

            if squad_name is not None:
                df_squad_row = df_squad[
                    (df_squad["Season"] == season) & (df_squad["Squad"] == squad_name)
                ]

                if not df_squad_row.empty:
                    srow = df_squad_row.iloc[0]

                    st.markdown(f"### Squad comparison – {squad_name} ({season})")
                    col_a, col_b, col_c = st.columns(3)

                    # Offensive comparison
                    if (
                        "OffScore_abs" in df_player.columns
                        and "OffScore_squad" in df_squad_row.columns
                    ):
                        player_off = df_player["OffScore_abs"].iloc[0]
                        squad_off = srow.get("OffScore_squad")
                        if pd.notna(player_off) and pd.notna(squad_off):
                            diff_off = player_off - squad_off
                            with col_a:
                                st.metric(
                                    "Offensive",
                                    value=f"{player_off:.0f}",
                                    delta=f"{diff_off:+.0f} vs squad",
                                )

                    # Midfield comparison
                    if (
                        "MidScore_abs" in df_player.columns
                        and "MidScore_squad" in df_squad_row.columns
                    ):
                        player_mid = df_player["MidScore_abs"].iloc[0]
                        squad_mid = srow.get("MidScore_squad")
                        if pd.notna(player_mid) and pd.notna(squad_mid):
                            diff_mid = player_mid - squad_mid
                            with col_b:
                                st.metric(
                                    "Midfield",
                                    value=f"{player_mid:.0f}",
                                    delta=f"{diff_mid:+.0f} vs squad",
                                )

                    # Defensive comparison
                    if (
                        "DefScore_abs" in df_player.columns
                        and "DefScore_squad" in df_squad_row.columns
                    ):
                        player_def = df_player["DefScore_abs"].iloc[0]
                        squad_def = srow.get("DefScore_squad")
                        if pd.notna(player_def) and pd.notna(squad_def):
                            diff_def = player_def - squad_def
                            with col_c:
                                st.metric(
                                    "Defensive",
                                    value=f"{player_def:.0f}",
                                    delta=f"{diff_def:+.0f} vs squad",
                                )


        # Score trend
        st.markdown("### Score trend (all seasons)")

        score_options = []
        label_to_col = {}

        if "OffScore_abs" in df_player_all.columns and df_player_all["OffScore_abs"].notna().any():
            score_options.append("Offensive score")
            label_to_col["Offensive score"] = "OffScore_abs"

        if "MidScore_abs" in df_player_all.columns and df_player_all["MidScore_abs"].notna().any():
            score_options.append("Midfield score")
            label_to_col["Midfield score"] = "MidScore_abs"

        if "DefScore_abs" in df_player_all.columns and df_player_all["DefScore_abs"].notna().any():
            score_options.append("Defensive score")
            label_to_col["Defensive score"] = "DefScore_abs"

        if not score_options:
            st.info("No scores available for this player.")
        else:
            selected_label = st.selectbox(
                "Select score for trend",
                score_options,
            )
            score_col = label_to_col[selected_label]
            score_trend_chart(df_player_all, score_col, selected_label)

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
