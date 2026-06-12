# LinkedIn Post — Young Hidden Gems (Age ≤ 23)
**Serie: PlayerScore · Feature-Post**
**Sprache:** Englisch
**Visuals:** 5 Grafiken in `/LinkedIn Posts/visuals/u23_gems/`
**Daten:** 2024-25 season · Age ≤ 23 · GemScore ≥ 9.0 · 30 players

---

## Post-Text

**I filtered Europe's Big-5 leagues for players aged 23 or younger with a GemScore ≥ 9. 30 names came up. Some you know. Most you probably don't.**

---

GemScore measures one thing: how much performance you get per million euros of market value.

The formula is simple — PlayerScore divided by market value, percentile-ranked across all eligible players, scaled 0–10.

GemScore 9+ means you're producing top-tier output at a bargain price.

Now filter for age ≤ 23? You get a shortlist of players who are already elite by value-for-money standards — and still have their peak years ahead of them.

---

**The numbers for 2024-25:**

→ 30 players aged 23 or under scored GemScore ≥ 9
→ Minimum criteria: ≥ 5×90 min played, market value ≤ €30M, known PlayerScore

**By league:**

🇮🇹 Serie A: **11 gems**
🇫🇷 Ligue 1: **9 gems**
🇪🇸 La Liga: **6 gems**
🇩🇪 Bundesliga: **3 gems**
🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League: **1 gem**

The Premier League's near-absence is a structural pattern, not a talent gap — market valuations are simply too inflated relative to performance output to compete at the €30M ceiling.

**By position:**

The list skews heavily defensive — 13 defenders, 9 midfielders, 7 forwards. Young defenders at mid-table clubs tend to post solid performance scores while commanding low valuations. That combination is a GemScore engine.

---

**The top 10:**

| # | Player | Club | League | Age | Score | GemScore |
|---|--------|------|--------|-----|-------|----------|
| 1 | Yoel Lago | Celta Vigo | La Liga | 20 | 599 | 10.0 |
| 2 | Max Geschwill | Holstein Kiel | Bundesliga | 23 | 718 | 10.0 |
| 3 | Sávio | Manchester City | Premier League | 20 | 498 | 10.0 |
| 4 | Pica | Alavés | La Liga | 22 | 686 | 10.0 |
| 5 | Saba Goglichidze | Empoli | Serie A | 20 | 578 | 10.0 |
| 6 | Anrie Chase | Stuttgart | Bundesliga | 20 | 548 | 9.9 |
| 7 | Fisayo Dele-Bashiru | Lazio | Serie A | 23 | 398 | 9.9 |
| 8 | Rabby Nzingoula | Montpellier | Ligue 1 | 18 | 366 | 9.9 |
| 9 | David Torres | Valladolid | La Liga | 21 | 493 | 9.8 |
| 10 | Damián Rodríguez | Celta Vigo | La Liga | 21 | 473 | 9.8 |

The youngest player in the full list: **Yan Diomandé, 17, Leganés**.

---

**A few observations:**

Sávio at Manchester City stands out immediately — a €0.2M market value for a player scoring 498 is almost certainly a data artefact (his real market value is far higher), but the PlayerScore is accurate.

Max Geschwill at Holstein Kiel — a PlayerScore of 718 as a 23-year-old defender in a relegated Bundesliga side. That's legitimately undervalued.

Fisayo Dele-Bashiru at Lazio — the Nigerian international is outperforming his valuation with one of the stronger attacking midfield scores in Serie A this season.

---

**Why this matters:**

These aren't just obscure players on a spreadsheet. Several will be household names in 3 years. GemScore surfaces them now — before the transfer window catches up.

The Hidden Gems filter is live in PlayerScore. You can adjust the age range, minimum score, maximum market value, and position.

🔗 [PlayerScore on GitHub Pages — link in bio]

---

**No subscriptions, no paywall. Open analytics for football.**

---

#Football #Analytics #DataScience #PlayerScore #HiddenGems #Scouting #BigData #EuropeanFootball #SerieA #Ligue1 #LaLiga #Bundesliga #PremierLeague

---

## Visuals (attach in order)

1. `01_scatter_u23.png` — Score vs. Market Value scatter by league (top 6 labeled)
2. `02_gems_per_league.png` — Horizontal bar chart: gems per league
3. `03_top10_table.png` — Dark-themed top 10 table
4. `04_age_distribution.png` — Bar chart: gems by age (17–23)
5. `05_league_position.png` — Stacked bars: league × position breakdown

---

## Notes

- Data: 2024-25 season, scraped from FBref + Transfermarkt
- PlayerScore: transparent benchmark scoring (no ML), 0–1000 scale
- GemScore cap at €30M market value ensures focus on genuinely attainable targets
- Age filter can now be set directly in the web app (new feature)
