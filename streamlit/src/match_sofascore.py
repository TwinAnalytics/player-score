# src/match_sofascore.py
"""
Matches FBref players to Sofascore players per league-season and produces
Data/Processed/player_sofascore_stats.csv: one row per (season, Comp, Player,
Squad) with the full set of Sofascore season metrics (prefixed `sofa_`).

Matching runs per FBref row (player + squad) within each league-season:
  1. Exact match on normalized name (Unicode -> ASCII, lowercase,
     hyphens/apostrophes stripped), after applying NAME_ALIASES
  2. If the exact candidate looks wrong (implausible minutes AND wrong team),
     or there is none, fuzzy candidates via token_set_ratio >= FUZZY_THRESHOLD
     are added to the pool
  3. Best candidate by: team-name similarity, name score, minutes plausibility

Minutes are validated on player level (FBref minutes summed across squads,
since Sofascore has one row per player per league-season). Players who switch
clubs within a league therefore get the same Sofascore season totals attached
to both of their FBref squad rows.

Unmatched players are written to Data/Processed/sofascore_unmatched.csv.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process as rfprocess

FUZZY_THRESHOLD = 85
TEAM_SIM_OK = 70

# FBref Comp -> Sofascore league slug (as used in the scraped CSVs)
COMP_TO_LEAGUE = {
    "eng Premier League": "premier-league",
    "es La Liga": "laliga",
    "de Bundesliga": "bundesliga",
    "it Serie A": "serie-a",
    "fr Ligue 1": "ligue-1",
}

# Manual fixes: normalized FBref name -> normalized Sofascore name
# (nicknames / completely different registrations that no fuzzy match can find)
NAME_ALIASES = {
    "mathias jorgensen": "zanka",
    "obite ndicka": "evan ndicka",
    "alfonso espino": "pacha",
    "jonny castro": "jonny otto",
    "jose luis garcia vaya": "pepelu",
}

# FBref squad abbreviations that token matching cannot bridge
SQUAD_REPLACEMENTS = {
    " utd": " united",
    "paris s g": "paris saint germain",
}

ID_COLS = ["player_id", "player_name", "player_slug", "team_id", "team_name", "position_group"]


# Characters that NFKD cannot decompose to ASCII and would silently drop
_CHAR_MAP = str.maketrans({
    "ø": "o", "Ø": "O", "æ": "ae", "Æ": "Ae", "ð": "d", "Ð": "D",
    "þ": "th", "Þ": "Th", "ł": "l", "Ł": "L", "đ": "dj", "Đ": "Dj", "ß": "ss",
})


def _normalize(s: str) -> str:
    s = str(s).translate(_CHAR_MAP)
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", errors="ignore").decode("ascii")
    s = s.lower().replace("-", " ").replace("'", "").replace(".", "")
    return " ".join(s.split())


def _normalize_squad(s: str) -> str:
    s = _normalize(s)
    for old, new in SQUAD_REPLACEMENTS.items():
        s = s.replace(old, new)
    return s


def _minutes_plausible(fbref_min: float, sofa_min: float) -> bool:
    """Accept if player-level total minutes roughly agree."""
    if pd.isna(fbref_min) or pd.isna(sofa_min):
        return True
    return abs(fbref_min - sofa_min) <= max(450.0, 0.5 * fbref_min)


def _team_sim(sofa_row: pd.Series, squad_norm: str) -> int:
    return fuzz.token_set_ratio(_normalize_squad(sofa_row.team_name), squad_norm)


def _select(pool: list[tuple[pd.Series, int]], squad_norm: str, total_min: float) -> tuple[pd.Series, int] | None:
    """Pick best (sofa_row, name_score) by team similarity, name score, minutes."""
    if not pool:
        return None

    def keyfn(item):
        row, name_score = item
        sim = _team_sim(row, squad_norm)
        plaus = _minutes_plausible(total_min, row.minutesPlayed)
        if pd.notna(total_min) and pd.notna(row.minutesPlayed):
            min_diff = abs(row.minutesPlayed - total_min)
        else:
            min_diff = 0.0
        return (sim >= TEAM_SIM_OK, name_score, plaus, -min_diff)

    return max(pool, key=keyfn)


def match_league_season(df_fbref: pd.DataFrame, df_sofa: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    df_fbref: FBref rows of ONE league-season (one row per player+squad).
    df_sofa:  Sofascore rows of the same league-season (one row per player).
    Returns (matched, unmatched) on FBref-row level.
    """
    totals = df_fbref.groupby("Player")["Min"].sum()

    sofa = df_sofa.reset_index(drop=True)
    sofa_norms = [_normalize(n) for n in sofa["player_name"]]
    exact_index: dict[str, list[int]] = {}
    for i, n in enumerate(sofa_norms):
        exact_index.setdefault(n, []).append(i)

    matched_rows, unmatched_rows = [], []
    used_sofa: set[int] = set()
    for _, p in df_fbref.iterrows():
        norm = _normalize(p.Player)
        norm = NAME_ALIASES.get(norm, norm)
        squad_norm = _normalize_squad(p.Squad)
        total_min = totals[p.Player]

        pool = [(sofa.iloc[i], 100) for i in exact_index.get(norm, [])]
        best = _select(pool, squad_norm, total_min)
        method = "exact"

        # No exact hit, or the exact candidate is implausible on both minutes
        # and team -> widen the pool with fuzzy candidates.
        if best is None or (
            not _minutes_plausible(total_min, best[0].minutesPlayed)
            and _team_sim(best[0], squad_norm) < TEAM_SIM_OK
        ):
            hits = rfprocess.extract(
                norm, sofa_norms, scorer=fuzz.token_set_ratio,
                score_cutoff=FUZZY_THRESHOLD, limit=5,
            )
            pool += [(sofa.iloc[h[2]], int(h[1])) for h in hits
                     if h[2] not in exact_index.get(norm, [])]
            widened = _select(pool, squad_norm, total_min)
            if widened is not None and (best is None or widened[1] < 100 or widened[0].name != best[0].name):
                method = "fuzzy"
            best = widened

        if best is None:
            unmatched_rows.append({"Player": p.Player, "Squad": p.Squad, "Min": p.Min, "Pos": p.Pos})
            continue

        best_row, name_score = best
        if method == "fuzzy" and not _minutes_plausible(total_min, best_row.minutesPlayed) \
                and _team_sim(best_row, squad_norm) < TEAM_SIM_OK:
            unmatched_rows.append({"Player": p.Player, "Squad": p.Squad, "Min": p.Min, "Pos": p.Pos})
            continue

        matched_rows.append({"Player": p.Player, "Squad": p.Squad,
                             "Min_fbref_total": total_min,
                             "MatchMethod": method, "MatchScore": name_score,
                             "sofa_index": best_row.name})
        used_sofa.add(best_row.name)

    # Pass 3: nickname registrations ("Maxi Gómez" vs "Maximiliano Gómez",
    # "Simy", "Savinho", …) share no usable name tokens, but team + minutes
    # form a near-unique fingerprint. Accept an unmatched FBref row if exactly
    # ONE still-unmatched Sofascore player of the same team has closely
    # agreeing minutes (and GK/outfield status is consistent).
    still_unmatched = []
    for p in unmatched_rows:
        total_min = totals[p["Player"]]
        squad_norm = _normalize_squad(p["Squad"])
        is_gk = "GK" in str(p.get("Pos", ""))
        cands = []
        for i in range(len(sofa)):
            if i in used_sofa:
                continue
            row = sofa.iloc[i]
            if (row.position_group == "G") != is_gk:
                continue
            if _team_sim(row, squad_norm) < TEAM_SIM_OK:
                continue
            if pd.notna(total_min) and pd.notna(row.minutesPlayed) \
                    and abs(row.minutesPlayed - total_min) <= max(250.0, 0.15 * total_min):
                cands.append(i)
        if len(cands) == 1:
            row = sofa.iloc[cands[0]]
            matched_rows.append({"Player": p["Player"], "Squad": p["Squad"],
                                 "Min_fbref_total": total_min,
                                 "MatchMethod": "team-minutes",
                                 "MatchScore": fuzz.token_set_ratio(
                                     _normalize(p["Player"]), _normalize(row.player_name)),
                                 "sofa_index": cands[0]})
            used_sofa.add(cands[0])
        else:
            still_unmatched.append(p)

    return pd.DataFrame(matched_rows), pd.DataFrame(still_unmatched)


def build_sofascore_lookup(processed_dir: Path, sofascore_dir: Path) -> None:
    processed_dir = Path(processed_dir)
    sofascore_dir = Path(sofascore_dir)

    sofa_all = pd.read_csv(sofascore_dir / "sofascore_player_stats_all_seasons_long.csv")
    stat_cols = [c for c in sofa_all.columns if c not in ("league", "season", *ID_COLS)]

    merged_frames, unmatched_frames = [], []
    for path in sorted(processed_dir.glob("player_scores-????-????.csv")):
        season = path.stem.replace("player_scores-", "")
        df_fb = pd.read_csv(path, usecols=["Player", "Squad", "Comp", "Pos", "Min"])
        season_sofa = sofa_all[sofa_all["season"] == season]
        if season_sofa.empty:
            print(f"[SOFA] {season}: no Sofascore data, skipping")
            continue

        for comp, league in COMP_TO_LEAGUE.items():
            df_fb_lg = df_fb[df_fb["Comp"] == comp]
            df_so_lg = season_sofa[season_sofa["league"] == league]
            if df_fb_lg.empty or df_so_lg.empty:
                continue

            matched, unmatched = match_league_season(df_fb_lg, df_so_lg)
            if not matched.empty:
                df_so_indexed = df_so_lg.reset_index(drop=True)
                out = matched.join(
                    df_so_indexed.loc[matched["sofa_index"], ID_COLS + stat_cols].reset_index(drop=True)
                )
                out = out.drop(columns=["sofa_index"])
                out = out.rename(columns={c: f"sofa_{c}" for c in ID_COLS + stat_cols})
                out.insert(0, "season", season)
                out.insert(1, "Comp", comp)
                merged_frames.append(out)
            if not unmatched.empty:
                unmatched.insert(0, "season", season)
                unmatched.insert(1, "Comp", comp)
                unmatched_frames.append(unmatched)

            n_fb = len(df_fb_lg)
            print(f"[SOFA] {season} {comp}: {len(matched)}/{n_fb} rows matched "
                  f"({100 * len(matched) / max(n_fb, 1):.1f}%)")

    result = pd.concat(merged_frames, ignore_index=True)
    out_path = processed_dir / "player_sofascore_stats.csv"
    result.to_csv(out_path, index=False)
    print(f"[SOFA] Saved {len(result)} rows -> {out_path}")

    if unmatched_frames:
        un = pd.concat(unmatched_frames, ignore_index=True)
        un_path = processed_dir / "sofascore_unmatched.csv"
        un.to_csv(un_path, index=False)
        print(f"[SOFA] {len(un)} unmatched rows -> {un_path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    build_sofascore_lookup(root / "Data" / "Processed", root / "Data" / "Raw" / "Sofascore")
