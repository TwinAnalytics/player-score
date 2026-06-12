# src/match_transfermarkt.py
"""
Fuzzy-matches FBref players to Transfermarkt players and produces
Data/Processed/player_market_values.csv with columns:
  Player, Squad, tm_player_id, MarketValue_EUR, MatchScore

Matching priority:
  1. Exact match on normalized name (Unicode → ASCII, lowercase, strip)
  2. Fuzzy match via rapidfuzz.fuzz.token_sort_ratio ≥ FUZZY_THRESHOLD
     with club name used as tiebreaker among equal-scoring candidates
  3. Unmatched rows are retained with NaN values (dropped from final output)
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

try:
    from rapidfuzz import fuzz, process as rfprocess

    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False
    print(
        "[TM WARN] rapidfuzz not installed. "
        "Fuzzy matching disabled; only exact matches will be used."
    )

FUZZY_THRESHOLD = 80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """Unicode → ASCII, lowercase, strip."""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", errors="ignore").decode("ascii")
    return s.lower().strip()


def _pick_col(columns, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


# ---------------------------------------------------------------------------
# Core matching
# ---------------------------------------------------------------------------

def _build_tm_index(df_tm: pd.DataFrame) -> tuple[
    list[str],          # tm_norms: normalized names (parallel to tm_rows)
    list[dict],         # tm_rows: {mv, tm_id, club_norm}
]:
    """Build in-memory lookup structures from the TM players DataFrame."""
    tm_norms: list[str] = []
    tm_rows: list[dict] = []

    name_col = _pick_col(df_tm.columns, ["name", "player_name", "full_name"])
    val_col = _pick_col(df_tm.columns, ["market_value_in_eur", "market_value", "value"])
    id_col = _pick_col(df_tm.columns, ["player_id", "id"])
    club_col = _pick_col(
        df_tm.columns,
        ["current_club_name", "club_name", "club"],
    )

    if name_col is None or val_col is None:
        return tm_norms, tm_rows

    for _, row in df_tm.iterrows():
        name_val = row.get(name_col)
        mv_val = row.get(val_col)
        if pd.isna(name_val) or pd.isna(mv_val):
            continue
        tm_norms.append(_normalize(str(name_val)))
        tm_rows.append(
            {
                "mv": float(mv_val),
                "tm_id": row[id_col] if id_col else None,
                "club_norm": _normalize(str(row[club_col])) if club_col else "",
            }
        )

    return tm_norms, tm_rows


def _match_player(
    norm_player: str,
    norm_squad: str,
    tm_norms: list[str],
    tm_rows: list[dict],
    exact_index: dict[str, list[int]],
) -> tuple[float | None, object, int]:
    """
    Returns (MarketValue_EUR, tm_id, match_score).
    """
    # 1. Exact match
    if norm_player in exact_index:
        candidates = exact_index[norm_player]
        # Pick candidate whose club matches best; otherwise highest market value
        best_idx = max(
            candidates,
            key=lambda i: (
                1 if tm_rows[i]["club_norm"] == norm_squad else 0,
                tm_rows[i]["mv"],
            ),
        )
        r = tm_rows[best_idx]
        return r["mv"], r["tm_id"], 100

    # 2. Fuzzy match — token_set_ratio handles "Joel Fujita" ↔ "Joel Chima Fujita"
    if _HAS_RAPIDFUZZ and tm_norms:
        results = rfprocess.extract(
            norm_player,
            tm_norms,
            scorer=fuzz.token_set_ratio,
            score_cutoff=FUZZY_THRESHOLD,
            limit=5,
        )
        if results:
            # results: list of (match_str, score, index)
            best_score = results[0][1]
            # Among top candidates with same score, prefer club match
            top = [r for r in results if r[1] == best_score]
            best = max(
                top,
                key=lambda r: (
                    1 if tm_rows[r[2]]["club_norm"] == norm_squad else 0,
                    tm_rows[r[2]]["mv"],
                ),
            )
            idx = best[2]
            r = tm_rows[idx]
            return r["mv"], r["tm_id"], int(best_score)

    return None, None, 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_market_value_lookup(
    raw_dir: Path,
    processed_dir: Path,
    df_fbref: pd.DataFrame,
) -> None:
    """
    Reads Data/Raw/tm_players.csv, matches FBref players by name,
    and writes Data/Processed/player_market_values.csv.

    df_fbref must contain at least columns: Player, Squad.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    tm_path = raw_dir / "tm_players.csv"
    if not tm_path.exists():
        print("[TM] tm_players.csv not found, skipping market value matching.")
        return

    df_tm = pd.read_csv(tm_path)
    print(f"[TM] Loaded {len(df_tm)} Transfermarkt players.")

    tm_norms, tm_rows = _build_tm_index(df_tm)
    if not tm_norms:
        print("[TM WARN] Could not build TM index (missing name/value columns).")
        return

    # Build exact-match index: norm_name → list of row indices
    exact_index: dict[str, list[int]] = {}
    for i, n in enumerate(tm_norms):
        exact_index.setdefault(n, []).append(i)

    # Unique (Player, Squad) pairs from FBref
    if df_fbref.empty or "Player" not in df_fbref.columns:
        print("[TM WARN] df_fbref empty or missing Player column.")
        return

    pairs = (
        df_fbref[["Player", "Squad"]]
        .dropna(subset=["Player"])
        .drop_duplicates()
        .copy()
    )
    print(f"[TM] Matching {len(pairs)} unique player-squad pairs …")

    results = []
    for _, pr in pairs.iterrows():
        player = str(pr["Player"])
        squad = str(pr.get("Squad", ""))
        norm_player = _normalize(player)
        norm_squad = _normalize(squad)

        mv, tm_id, score = _match_player(
            norm_player, norm_squad, tm_norms, tm_rows, exact_index
        )
        results.append(
            {
                "Player": player,
                "Squad": squad,
                "tm_player_id": tm_id,
                "MarketValue_EUR": mv,
                "MatchScore": score,
            }
        )

    df_out = pd.DataFrame(results)

    matched = df_out["MarketValue_EUR"].notna().sum()
    total = len(df_out)
    print(
        f"[TM] Matched {matched}/{total} player-squad pairs "
        f"({100 * matched / max(total, 1):.1f}%)"
    )

    out_path = processed_dir / "player_market_values.csv"
    df_out.to_csv(out_path, index=False)
    print(f"[TM] Saved → {out_path}")
