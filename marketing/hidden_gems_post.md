# LinkedIn Post — Hidden Gems in European Football
**Serie: PlayerScore · Woche 2 (alternativ: eigenständiger Feature-Post)**
**Sprache:** Englisch
**Visuals:** 5 Grafiken in `/LinkedIn Posts/visuals/`

---

## Post-Text

**I ran every Big-5 player through a value-for-money algorithm. 91 out of 870 scored a perfect GemScore of 9 or above. Here's what I found.**

---

In football analytics, the most interesting players aren't the ones ranked highest. They're the ones ranked highest *relative to what they cost*.

That's the core idea behind the Hidden Gems feature in PlayerScore — my open football analytics platform covering every outfield player across Europe's Big-5 leagues since 2017.

---

**Here's how GemScore works:**

1. Filter all players with a PlayerScore ≥ 400, at least 5×90 minutes played, and a known market value ≤ €30M.
2. Compute a value-for-money ratio: **PlayerScore / Market Value (€M)**
3. Percentile-rank that ratio across all eligible players in the same season.
4. Divide by 10 → **GemScore 0–10**.

GemScore 10 = you're producing elite-level output at the lowest cost in the dataset.
GemScore 9+ = top 10% by value for money.

No ML, no black box. Transparent math.

---

**What the data says for 2025/26:**

→ 870 players met the minimum criteria (score, minutes, market data)
→ **91 players scored GemScore ≥ 9** — the top 10% by value for money

The league breakdown is striking:

🇪🇸 La Liga: **26 gems**
🇩🇪 Bundesliga: **24 gems**
🇫🇷 Ligue 1: **23 gems**
🇮🇹 Serie A: **13 gems**
🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League: **5 gems**

The Premier League result isn't surprising — it's a direct consequence of inflated transfer fees and wages distorting the score-to-value ratio. World-class players at PL clubs often still outperform their price tags, but far fewer do so at the €30M ceiling.

---

**Three names that stand out:**

**Nicolás Paz** (Como, Serie A) — PlayerScore 741 ("World Class"), market value €0.10M. A 21-year-old central midfielder producing at a level typically seen at top-6 clubs. The market simply hasn't caught up yet.

**Gerard Martín** (Barcelona, La Liga) — PlayerScore 737, market value €0.05M. A 24-year-old left back whose defensive and progressive output ranks him above the Big-5 median for his role — playing at *Barcelona*.

**Julian Chabot** (Stuttgart, Bundesliga) — PlayerScore 686, market value €0.05M. Quietly one of the most efficient defenders in the Bundesliga this season.

---

**One structural insight:**

59 of 91 gems (65%) are **central defenders or defensive midfielders**. The market systematically underprices defensive output — perhaps because goals and assists are more visible than clean sheets and interception chains. The model doesn't care. It scores what it measures.

The median gem is **26 years old** — neither a teenager nor a veteran. The sweet spot where experience and low market value briefly coexist before a club decides to reprice them.

---

**The scatter plot tells the whole story.**

Everything in the bottom-right quadrant is a Hidden Gem: high score, low market value. The teal dots are where I'd start every scouting conversation.

PlayerScore is live, free, and open source.

🔗 Explore the Hidden Gems filter: [App link]
💻 GitHub: [Repo link]

---

*All data: FBref Big-5, 2025/26 season. Market values: Transfermarkt. Minimum 5×90 minutes.*

---

#FootballAnalytics #DataScience #HiddenGems #Python #Scouting #OpenSource #FootballData

---

## Visuals (in order of posting)

| # | File | Beschreibung |
|---|---|---|
| 1 | `01_scatter_hidden_gems.png` | Hauptvisual: Score vs. Marktwert, Quadranten, Gems hervorgehoben, Top-8 gelabelt |
| 2 | `02_gems_per_league.png` | Gems pro Liga (horizontale Balken, ligaspezifische Farben) |
| 3 | `03_gemscore_distribution.png` | GemScore-Verteilung Histogramm (zeigt wo die 9+ Spieler liegen) |
| 4 | `04_top10_gems_table.png` | Top 10 Spieler als dunkle Tabelle |
| 5 | `05_age_distribution.png` | Altersverteilung: Gems vs. alle Spieler |

**Empfehlung:** Post das Scatter-Plot als Titelbild (erstes Bild). Die restlichen 4 als Carousel.

---

## Key Numbers Cheat Sheet

| Metrik | Wert |
|---|---|
| Saison | 2025-2026 |
| Spieler mit Marktdaten | 870 |
| GemScore ≥ 9 | 91 |
| Meiste Gems | La Liga (26) |
| Wenigste Gems | Premier League (5) |
| Medianalter der Gems | 26 Jahre |
| Top GemScore 10.0 | Chabot, Martín, Paz, Yegbe |
