# PlayerScore

PlayerScore is an interpretable rating model for football players in the Big-5 European leagues (FBref data). It collects multi-season player stats, engineers role-specific features (per 90, positional logic), and computes offensive, midfield, and defensive scores (0–1000) with clear performance bands.

**Score scale & tiers (0–1000)**  

- **900–1000** → Exceptional  
- **750–899** → World Class  
- **400–749** → Top Starter (Big-5 regular)  
- **200–399** → Solid Squad Player  
- **0–199** → Below Big-5 Level


**Live demo:** https://twinanalytics-player-score.streamlit.app/

PlayerScore is an interpretable rating model for football players in the Big-5 European leagues (FBref data). It collects multi-season player stats, engineers role-specific features (per 90, positional logic), and computes offensive, midfield, and defensive scores (0–1000) with clear performance bands.

## Features

- Web scraping of FBref Big-5 player stats with Playwright
- Feature engineering (per-90 metrics, minutes filter, positional refinement)
- Role-specific scores:
  - Offensive (FW, Off_MF)
  - Midfield (MF)
  - Defensive (DF, Def_MF)
- Multi-season support (from 2017/2018 to 2025/2026)
- Streamlit app for interactive exploration (player profiles & toplists)

## Project Structure

```text
src/
  scraping_fbref_player_stats.py  # scraping & light CSV
  processing.py                   # per90 + positions + filters
  scoring.py                      # score functions, weights, bands
  pipeline.py                     # end-to-end pipeline
notebooks/
  01_feature_dev_2024_25.ipynb
  02_scoring_analysis_2024_2025.ipynb
Data/
  Raw/       # not in repo
  Processed/ # scored CSVs per season
app.py       # Streamlit app
run_pipeline.py

## Installation

```bash
git clone https://github.com/TwinAnalytics/player-score.git
cd player-score

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.txt

## Documentation

A more detailed project report is available in:

- `docs/PlayerScore_Documentation_EN.pdf`

