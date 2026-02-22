# PlayerScore

PlayerScore transforms raw FBref data into interpretable, role-aware performance
insights — making it easy to compare players across leagues, seasons, and clubs.

Scores are fully transparent and benchmark-driven (no ML black-box), ranging
0–1000 with 5 qualitative tiers.

---

## 📦 Data Acquisition
- Automated scraping of Big-5 player stats (FBref) via Playwright
- Multi-season dataset: 2017/18 – 2025/26
- Market value data integrated from Transfermarkt (142 clubs)
- Robust handling of missing or league-limited stats

---

## 🧠 Feature Engineering
- Per-90 normalization for all relevant metrics
- Minutes thresholds and data quality filters
- Unified positional logic — players classified into:
  - FW / Off_MF (offensive roles)
  - MF (central midfield)
  - DF / Def_MF (defensive roles)

---

## 📊 Role-Specific Scoring

Each player receives up to three interpretable scores:
- **Offensive Score** (FW, Off_MF)
- **Midfield Score** (MF)
- **Defensive Score** (DF, Def_MF)

Score formula: `Σ(weight × min(metric / benchmark, 1.0)) / Σweights × 1000`

**Tiers:**
| Score | Band |
|-------|------|
| ≥ 900 | Exceptional |
| 750–899 | World Class |
| 400–749 | Top Starter |
| 200–399 | Solid Squad Player |
| < 200 | Below Big-5 Level |

---

## 🖥️ App Modes (Streamlit UI)

### 👤 Player Profile
- Per-season and career views with club crest
- Pizza charts vs Big-5 role peers
- Role-based scatter plots (xG vs G, xAG vs A, …)
- Career score trend lines
- Summary tiles: age, minutes, score, band, market value
- **Export:** FIFA-style PNG card · Full PDF report

### 📋 Player Rankings
- Filters: season, league, club, position, age, minutes
- Top-N bar charts by primary role score
- Score vs age beeswarm plot
- Band distribution for filtered set
- Click any row → opens player profile directly

### ⚽ Team Scores
- Minute-weighted squad offense / midfield / defense rankings
- League table with club crests
- Budget efficiency scatter (squad score vs total market value)
- Top contributors per club

### 💎 Hidden Gems
- Score-to-market-value efficiency ranking
- Identifies undervalued players within any filtered set
- Gem Score ranked 1–10 within selection

### 🆚 Compare Players
- Side-by-side FIFA cards for two players
- Score comparison chart (offense / midfield / defense / overall)
- Pizza charts side by side
- Key metrics table

---

## 🎨 Visual Features
- **FIFA-style player cards** with club crest, score, band, market value, key metrics
- **142 club crests** — all Big-5 clubs, sourced from ESPN
- **PNG card export** — shareable player card
- **PDF player report** — full A4 report with pizza chart and scatter plot

---

## ⚙️ CI/CD
Weekly GitHub Actions pipeline (self-hosted macOS runner) scrapes, processes,
and commits updated data every Tuesday 06:00 UTC.

---

## ❓ Why PlayerScore?

Modern football recruitment needs transparent, reproducible metrics — not
black-box models.

PlayerScore is built around:
- **Consistency** across leagues and competitions
- **Role-aware evaluation** based on real positional behavior
- **Reproducible scoring** using open data (FBref)
- **Explorable analytics** for scouting, recruitment, and squad planning
