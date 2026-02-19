# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Football analytics platform that scrapes FBref (Football-Reference) data and computes role-aware player performance scores across European leagues (Big-5). Scores are transparent and benchmark-driven (no ML black-box), ranging 0–1000 with 5 qualitative tiers.

## Common Commands

### Run the Streamlit app
```bash
streamlit run app.py
```

### Run the full multi-season pipeline
```bash
python run_multi_season_pipeline.py
```

### Pipeline environment variables
| Variable | Purpose | Default |
|---|---|---|
| `DO_SCRAPE` | Enable web scraping | `true` |
| `DO_PROCESS` | Enable processing | `true` |
| `USE_AUTO_SEASONS` | Auto-generate season range | `true` |
| `AUTO_SEASON_START` | First season | `2017-2018` |
| `PIPELINE_SEASONS` | Explicit comma-separated seasons (overrides auto) | — |
| `SCRAPE_PLAYERS` / `SCRAPE_SQUADS` / `SCRAPE_BIG5_TABLE` | Granular scraping control | `true` |

### Install dependencies (fresh environment)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install playwright && python -m playwright install chromium
```

## Architecture

### Data Flow
```
FBref (Playwright scraping)
  → Data/Raw/players_data-YYYY-YYYY.csv
  → src/processing.py   (position classification, per-90 normalization)
  → src/scoring.py      (role-specific benchmark scoring 0–1000)
  → Data/Processed/player_scores-YYYY-YYYY.csv
  → src/multi_season.py (season aggregation)
  → src/squad.py        (minute-weighted squad scores)
  → Data/Processed/*_all_seasons*.csv
  → app.py              (Streamlit UI)
```

### Key Source Files
- **`run_multi_season_pipeline.py`** — CI/CD entry point; orchestrates scraping + processing across all seasons
- **`src/pipeline.py`** — single-season pipeline (load raw → features → scores → save)
- **`src/processing.py`** — position classification (`prepare_positions`, `refine_mf_with_zones`), per-90 normalization
- **`src/scoring.py`** — hardcoded weights & benchmarks per role (FW, Off_MF, MF, Def_MF, DF); `compute_score_absolute()` returns 0–1000
- **`src/scraping_fbref_player_stats.py`** — Playwright-based scraper for 11 FBref stat categories; merges into single CSV
- **`src/multi_season.py`** — loads all season CSVs, aggregates player career stats
- **`src/squad.py`** — minute-weighted squad-level scores
- **`app.py`** — 4,500-line Streamlit dashboard (player profiles, top lists, team scores, pizza charts, scatter plots)

### Scoring Logic (`src/scoring.py`)
- Each role has explicit feature **weights** and **benchmarks** (the value considered "full performance")
- Score formula: `sum(weight × min(feature/benchmark, 1.0)) / sum(weights) × 1000`
- Bands: Exceptional (≥900), World Class (750–899), Top Starter (400–749), Solid Squad Player (200–399), Below Big-5 Level (<200)
- Position roles: `FW`, `Off_MF`, `MF`, `Def_MF`, `DF` (GK excluded from outfield scoring)

### Position Classification (`src/processing.py`)
- `main_pos_from_string()` parses FBref multi-position strings (e.g., `"MF,DF"`)
- `refine_mf_with_zones()` splits midfielders into `Off_MF` vs `Def_MF` based on touch zones

### Processed Data Files
| File | Content |
|---|---|
| `player_scores-YYYY-YYYY.csv` | Per-player per-season scores + bands |
| `player_scores_all_seasons_long.csv` | All seasons concatenated |
| `player_scores_agg_by_player.csv` | Career averages per player |
| `squad_scores_all_seasons.csv` | Minute-weighted squad strength by season |
| `big5_table_all_seasons.csv` | Combined league standings |

## CI/CD

Weekly GitHub Actions workflow (`.github/workflows/playerscore-weekly-current-season.yml`) runs every Tuesday 06:00 UTC on a **self-hosted macOS runner**. It scrapes, processes, and commits updated data back to the repo with message `"Weekly run (YYYY-YYYY) YYYY-MM-DD HH:MM UTC"`.
