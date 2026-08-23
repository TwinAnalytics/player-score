# src/league_strength.py
"""
League strength factors from bridge players (Sofascore ratings).

Idea: the same player rates lower in a harder league. Model every qualified
player-season rating as

    rating ~ player_ability + league_effect + noise

a two-way fixed-effects model. Players who appear in more than one league
connect the leagues and identify the league effects, controlling for
individual quality. Solved by alternating least squares (minute-weighted),
anchored so the Big-5 average league effect is 0.

The league effect is a difficulty offset in rating points (an easier league
has a POSITIVE effect: ratings are inflated). It is turned into a score
multiplier: Big-5 = 1.0, easier leagues < 1.0. Big-5 leagues always stay at
1.0 so the frozen Big-5 history never changes.

Output: src/league_strength.json  { comp_label: factor }

Run:
    python -m src.league_strength
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .processing_sofa import LEAGUE_TO_COMP, EXTRA_LEAGUE_TO_COMP

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "Data" / "Raw" / "Sofascore"
OUT_PATH = Path(__file__).resolve().parent / "league_strength.json"

ALL_LEAGUE_TO_COMP = {**LEAGUE_TO_COMP, **EXTRA_LEAGUE_TO_COMP}
BIG5_SLUGS = set(LEAGUE_TO_COMP)

MIN_MINUTES = 600          # ~7 full matches for a stable rating
DISCOUNT_ALPHA = 0.37      # rating-point -> score-discount slope (tunable)
FACTOR_FLOOR = 0.70        # never discount a league below this


def _load_ratings(min_minutes: int = MIN_MINUTES) -> pd.DataFrame:
    """(player_id, slug, season, rating, minutes) for all scored leagues."""
    rows = []
    for path in glob.glob(str(RAW / "sofascore_player_stats-*-????-????.csv")):
        m = re.search(r"stats-(.+)-(\d{4}-\d{4})\.csv$", path)
        if not m or m.group(1) not in ALL_LEAGUE_TO_COMP:
            continue
        df = pd.read_csv(path, usecols=["player_id", "minutesPlayed", "rating"])
        df = df[(df["minutesPlayed"] >= min_minutes) & df["rating"].notna()]
        df["slug"] = m.group(1)
        df["season"] = m.group(2)
        rows.append(df)
    data = pd.concat(rows, ignore_index=True)
    data = data.rename(columns={"minutesPlayed": "min"})
    return data


def _fit_two_way_fe(data: pd.DataFrame, iters: int = 100) -> dict[str, float]:
    """Alternating least squares for rating = player + league effect.
    Returns {slug: league_effect}, Big-5 minute-weighted mean anchored to 0."""
    players = data["player_id"].to_numpy()
    leagues = data["slug"].to_numpy()
    y = data["rating"].to_numpy(dtype=float)
    w = data["min"].to_numpy(dtype=float)

    uP, pidx = np.unique(players, return_inverse=True)
    uL, lidx = np.unique(leagues, return_inverse=True)

    def wmean_by(idx, n, vals, weights):
        num = np.bincount(idx, weights=vals * weights, minlength=n)
        den = np.bincount(idx, weights=weights, minlength=n)
        return num / np.where(den == 0, 1.0, den)

    a = np.zeros(len(uP))
    b = np.zeros(len(uL))
    big5_mask = np.array([s in BIG5_SLUGS for s in uL])
    league_min = np.bincount(lidx, weights=w, minlength=len(uL))

    for _ in range(iters):
        a = wmean_by(pidx, len(uP), y - b[lidx], w)
        b = wmean_by(lidx, len(uL), y - a[pidx], w)
        anchor = np.average(b[big5_mask], weights=league_min[big5_mask])
        b -= anchor

    return {uL[i]: float(b[i]) for i in range(len(uL))}


def _to_factor(effect: float, slug: str) -> float:
    """League effect (rating points, + = easier) -> score multiplier.
    Big-5 fixed at 1.0; easier leagues discounted, clipped at FACTOR_FLOOR."""
    if slug in BIG5_SLUGS:
        return 1.0
    factor = 1.0 - DISCOUNT_ALPHA * effect
    return round(float(np.clip(factor, FACTOR_FLOOR, 1.0)), 3)


def compute() -> tuple[dict[str, float], pd.DataFrame]:
    data = _load_ratings()
    effects = _fit_two_way_fe(data)

    # bridge counts: players of a league who also appear in another league
    multi = data.groupby("player_id")["slug"].nunique()
    multi_players = set(multi[multi >= 2].index)
    bridges = (data[data["player_id"].isin(multi_players)]
               .groupby("slug")["player_id"].nunique())

    report = []
    factors = {}
    for slug, comp in ALL_LEAGUE_TO_COMP.items():
        eff = effects.get(slug)
        if eff is None:
            continue
        f = _to_factor(eff, slug)
        factors[comp] = f
        report.append({"slug": slug, "comp": comp, "effect": round(eff, 3),
                       "factor": f, "bridges": int(bridges.get(slug, 0))})
    rep = pd.DataFrame(report).sort_values("factor", ascending=False)
    return factors, rep


def load_factors() -> dict[str, float]:
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text())
    return {}


if __name__ == "__main__":
    factors, rep = compute()
    OUT_PATH.write_text(json.dumps(factors, indent=2, ensure_ascii=False))
    pd.set_option("display.width", 200)
    print(rep.to_string(index=False))
    print(f"\n-> {OUT_PATH}")
