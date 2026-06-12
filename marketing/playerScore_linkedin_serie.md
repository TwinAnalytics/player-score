# PlayerScore — LinkedIn Post-Serie

**Ziel:** Recruiter, Data-Science-Jobs & Football-Analytics-Community erreichen
**Format:** 6 Posts über 6 Wochen, abwechselnd Feature- und Inhaltsposts
**Sprache:** Englisch
**Empfohlene Posting-Zeiten:** Di–Do, 8–10 Uhr oder 17–19 Uhr
**Hinweis:** Screenshots der App zu jedem Post anhängen — erhöht die Reichweite deutlich

---

## Übersicht

| Woche | Typ | Thema | Primäre Zielgruppe |
|---|---|---|---|
| 1 | Intro | Was ist PlayerScore, warum hab ich's gebaut | Alle |
| 2 | Inhalt | Hidden Gem #1 — Pedro Bigas | Football Analytics |
| 3 | Feature | Das Scoring-System — kein Black Box | DS Recruiter |
| 4 | Inhalt | Hidden Gem #2 — Fisnik Asllani | Football Analytics |
| 5 | Feature | Similar Players + Compare | DS + Football |
| 6 | Feature | Age Curve — Entwicklung vs. Peers | Alle |

---

## Post 1 — Intro (Woche 1)

**I spent the last months building a football analytics platform. Here's what I learned.**

Scouting databases cost thousands per year. So I built my own.

PlayerScore is an open analytics platform covering Europe's Big-5 leagues — Premier League, La Liga, Bundesliga, Serie A, Ligue 1. Every player. Every season since 2017.

The core idea: **role-aware performance scores from 0 to 1000.**

A striker and a defensive midfielder shouldn't be judged on the same metrics. So they aren't. Each position group has its own benchmarks, its own weights, its own scale.

Five tiers:
→ Exceptional (900+)
→ World Class (750–899)
→ Top Starter (400–749)
→ Solid Squad Player (200–399)
→ Below Big-5 Level (<200)

No black box. No ML. Just transparent, benchmark-driven math — so you always know *why* a player scores what they score.

Built with Python, Streamlit, and FBref data. Scraped weekly via Playwright, scored via a custom pipeline, deployed automatically via GitHub Actions.

The app lets you:
↳ Explore any player's full career profile
↳ Filter rankings by league, club and position
↳ Find undervalued players (hidden gems)
↳ Compare two players side by side
↳ Discover stylistically similar players

I'll be sharing specific findings and features over the next few weeks.

If you're into football analytics or data engineering — follow along.

🔗 [App link]

#FootballAnalytics #DataScience #Python #Streamlit #Scouting

---

## Post 2 — Hidden Gem #1: Pedro Bigas (Woche 2)

> ⚠️ Vor dem Posten ausfüllen: Score, Band, Peer-Percentile, Marktwert, Per-90-Werte, Club, Liga, Saison, Alter

**The Big-5 has defenders nobody talks about. The data does.**

Pedro Bigas isn't a name you'll find on transfer speculation lists. He doesn't play for a Champions League club. He doesn't have a highlight reel with millions of views.

But run him through a role-specific performance model and the number is hard to ignore.

PlayerScore — [CLUB], [LEAGUE], [SEASON]:
→ Defensive Score: **[SCORE] / 1000 — [BAND]**
→ Peer percentile (DF role, Big-5): **top [X]%**
→ Market value: **€[VALUE]M**

The metrics behind the score:
→ Tackles won per 90: [VALUE]
→ Interceptions per 90: [VALUE]
→ Clearances per 90: [VALUE]
→ Progressive passes per 90: [VALUE]

What stands out is the consistency. Bigas doesn't have boom-bust seasons. Season after season, the model rates him above the median for his role — which in a league as competitive as [LEAGUE] means something.

He's [AGE]. Experienced. Positionally smart. And priced like a backup.

This is the profile that gets overlooked at big clubs — and that gets quietly decisive at mid-table ones.

The model noticed. The market hasn't quite caught up.

🔗 Full profile: [App link]

#FootballAnalytics #HiddenGems #DataDrivenScouting #LaLiga

---

## Post 3 — Das Scoring-System (Woche 3)

**I deliberately didn't use machine learning for this. Here's why.**

When I started building PlayerScore, the obvious path was: collect data, train a model, output ratings.

I took a different route.

The problem with ML-based player ratings: **you can't explain them.** A scout asks "why is this player rated 7.2?" and the honest answer is "the model said so."

That's not useful. And in football, it's not trusted.

So I built a transparent scoring system instead.

Here's how it works:

**1. Role classification**
Players are grouped into 5 roles: FW, Off_MF, MF, Def_MF, DF. Each role has its own set of metrics that actually matter for that position.

**2. Per-90 normalization**
Raw stats penalise players with fewer minutes. Everything is normalized to per-90 minutes played — so a 60-minute-per-game player competes fairly with a starter.

**3. Benchmark scoring**
Each metric is compared to a benchmark — the value considered "full performance" for that role. A player hitting or exceeding that benchmark gets full weight on that dimension.

**4. Weighted sum → 0–1000**
Metrics are weighted by importance for the role and summed. The result is always interpretable: "this player scores 680 because they're elite at progressive carries but below average in pressing."

No black box. Every number has a reason.

The full pipeline runs automatically every Tuesday — scraping 11 FBref stat categories, processing, scoring, and committing updated data back to the repo via GitHub Actions.

Code is on GitHub: [Link]

#DataScience #FootballAnalytics #Python #TransparentAI #MLEngineering

---

## Post 4 — Hidden Gem #2: Fisnik Asllani (Woche 4)

> ⚠️ Vor dem Posten ausfüllen: Score, Band, Peer-Percentile, Marktwert, Per-90-Werte, Saison, Peer-Median-Wert laut Age Curve

**Inter Milan have a midfielder worth watching closely. The age curve makes it obvious.**

Fisnik Asllani is 22. He plays in Serie A. He controls tempo, progresses the ball, and presses with structure.

And according to PlayerScore, he's already performing well above what the market implies.

PlayerScore — [CLUB], [SEASON]:
→ Midfield Score: **[SCORE] / 1000 — [BAND]**
→ Peer percentile (MF role, Big-5): **top [X]%**
→ Market value: **€[VALUE]M**

Key metrics per 90:
→ Progressive carries: [VALUE]
→ Progressive passes: [VALUE]
→ Successful take-ons: [VALUE]
→ Interceptions: [VALUE]

The age curve is the real story here.

The median Big-5 midfielder at age 22 scores around [PEER MEDIAN]. Asllani is tracking at **[SCORE]** — [X] points above that baseline. And the trajectory is still pointing up.

Players who are above the peer median this early and still improving are rare. The model has seen [NUMBER]+ midfielder seasons across 8 years of Big-5 data. Profiles like this — young, efficient, positionally disciplined — tend to peak later and higher than the market anticipates.

Inter know what they have. The rest of Europe is starting to figure it out.

🔗 Explore his full profile and age curve: [App link]

#FootballAnalytics #HiddenGems #SerieA #Scouting #DataScience

---

## Post 5 — Similar Players + Compare (Woche 5)

**I asked my app: "Who plays like [WELL-KNOWN PLAYER]?" The answer was surprisingly good.**

One of the new features in PlayerScore: **style-based player similarity.**

The idea is simple. Every player has a pizza chart — 14 per-90 metrics covering possession, attacking output, and defensive contribution. Instead of comparing raw numbers, I percentile-rank each metric within their role group.

Then: Euclidean distance.

The player closest in that 14-dimensional space is the most stylistically similar — regardless of club, league, or market value.

I tested it on **[WELL-KNOWN PLAYER]**:

The top 5 similar players were:
1. [PLAYER 1] — [CLUB] · [SCORE]
2. [PLAYER 2] — [CLUB] · [SCORE]
3. [PLAYER 3] — [CLUB] · [SCORE]
4. [PLAYER 4] — [CLUB] · [SCORE]
5. [PLAYER 5] — [CLUB] · [SCORE]

Some familiar names. Some surprises.

The Compare Players feature lets you then put any two of them side by side — scores, pizza profiles, key metrics — to see exactly where they converge and diverge.

For scouting, this is the core use case: you lose a player, you need a replacement with a similar profile. The model gives you a shortlist in seconds.

Technically: it's not a fancy embedding model. It's intentional simplicity — percentile distance on domain-relevant features. Explainable, fast, and in this case, surprisingly accurate.

🔗 Try it yourself: [App link]

#FootballAnalytics #DataScience #Scouting #Python #SimilarPlayers

---

## Post 6 — Age Curve (Woche 6)

**When do footballers actually peak? I built a chart to find out.**

The new Age Curve feature in PlayerScore answers a question I kept coming back to while building the platform:

*Is this player getting better, plateauing, or declining — relative to their peers?*

The chart overlays two things:
→ **Grey dashed line:** median PlayerScore for all Big-5 players at the same position, by age
→ **Teal line:** the selected player's actual score across every season in the database

The peer median gives you the baseline. A typical midfielder in the Big-5 peaks around their mid-to-late 20s and starts declining around 30. But individual trajectories diverge sharply from that curve.

**Late developers** — players who score below the median in their early 20s and overtake it after 25. The market undervalues them for years.

**Early peakers** — high scores at 21–22, then a plateau. Often overpriced at the exact moment they stop improving.

**Consistent outliers** — players who track above the median for a full decade. These are the genuinely rare ones.

The age curve doesn't predict the future. But it contextualises the present — and that's often enough to ask better questions.

The feature runs on all 8+ seasons of Big-5 data in the platform. Every player. Every age.

🔗 [App link] — look up any player's curve.

#FootballAnalytics #DataScience #PlayerDevelopment #Python #Scouting

---

## Offene Platzhalter — Checkliste

### Post 2 (Pedro Bigas)
- [ ] Club, Liga, Saison
- [ ] Score + Band
- [ ] Peer-Percentile
- [ ] Marktwert
- [ ] Tackles won / Interceptions / Clearances / Progressive passes per 90
- [ ] Alter
- [ ] App-Link + Screenshot

### Post 4 (Fisnik Asllani)
- [ ] Saison
- [ ] Score + Band
- [ ] Peer-Percentile
- [ ] Marktwert
- [ ] Progressive carries / passes / Take-ons / Interceptions per 90
- [ ] Peer-Median-Wert (aus Age Curve)
- [ ] Anzahl MF-Saisons in der Datenbank (aus df_all)
- [ ] App-Link + Screenshot

### Post 5 (Similar Players)
- [ ] Bekannten Spieler wählen und Similar Players aus App ablesen
- [ ] Top 5 mit Club + Score eintragen
- [ ] App-Link + Screenshot
