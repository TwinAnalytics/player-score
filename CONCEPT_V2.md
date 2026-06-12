# PlayerScore V2 — Konzept: Professional Scouting Platform

> Erstellt aus drei parallelen Agent-Analysen (Data, Design, Product).  
> Ziel: Umbau von einem Analytics-Portfolio-Projekt zu einem professionellen B2B Scouting-Werkzeug für Fussballclubs.

---

## 1. Positionierung & USP

PlayerScore besetzt eine Lücke die kein anderes Tool füllt:

| Dimension | PlayerScore V2 | Wyscout | Opta/StatsBomb |
|-----------|---------------|---------|----------------|
| **Preis** | Frei / Low-cost | 3.000–15.000€/Jahr | 20.000–100.000€/Jahr |
| **Transparenz** | Vollständig (Formeln, Gewichte) | Keine | Keine |
| **Score-System** | Rollen-adjustiert 0–1000 | Nur Rohdaten | Nur Rohdaten |
| **Historisch** | 2017–heute (9 Saisons) | Ab Kauf | Partiell |
| **Scouting Mission** | Eingebaut (geführter Workflow) | Nur Suche | Nur Suche |
| **Video** | Nicht vorhanden | Kernfeature | Teilweise |
| **Erklärbarkeit** | Jeder Score erklärbar in 1 Satz | Black Box | Black Box |

**Kernaussage:** Wyscout-Funktionalität mit Hudl-Ästhetik, zum Preis von Transfermarkt. Kein anderes Tool erklärt warum ein Spieler einen Score hat.

---

## 2. Zielgruppen

### Persona A — "Der Pragmatische Scout"
Markus, 38, Video Scout bei einem Zweitligaklub. Kein Wyscout-Budget. Muss Empfehlungen intern verteidigen.
- **Braucht:** Schnelle Kandidatenliste für offene Position → PDF-Report für das Coaching-Staff-Meeting
- **Frustration:** Tools zu teuer. Kein Tool hilft ihm das "Warum" eines Spielers zu erklären.

### Persona B — "Die Daten-affine Sporting Director"
Ana, 44, Sporting Director Primera División. Entscheidet über Transfer-Budgets.
- **Braucht:** Kader-Schwächen erkennen, 3 Spieler vergleichen die ihr Scout-Team empfohlen hat
- **Frustration:** Scout-PDFs nicht vergleichbar. Keine konsistente Metrik über alle Positionen.

### Persona C — "Der Talentscout"
Ibra, 31, unabhängiger Agent. Fokus U23 aus Ligue 2 / Eredivisie.
- **Braucht:** Unterbewertete Talente finden bevor ein Klub sie entdeckt. Entwicklungskurve zeigen.
- **Frustration:** Wyscout zu teuer. Kein Tool zeigt "wann war der Wendepunkt" einer Karriere.

---

## 3. Design-Entscheidung: "Precision Dark" (Richtung A)

Evolution der bestehenden Apple Dark Basis. Kein Komplettbruch — gezieltes Upgrade.

### 3.1 Neue Navigationsstruktur: Collapsible Left Sidebar

**Warum Sidebar statt Top-Nav:**
Wyscout, InStat, Opta Vision nutzen alle Sidebar-Navigation. Sie kommuniziert "Werkzeug" statt "Website" und skaliert auf 20+ Items ohne Overflow.

```
┌─────────────────────────────────────────────────────────────────┐
│  [Logo]  [Global Search — Cmd+K]              [Scout Mode] [User]│  ← Top Bar (56px)
├──────────┬──────────────────────────────────┬───────────────────┤
│          │                                  │                   │
│ Sidebar  │     Main Content Area            │  Shortlist Panel  │
│ 72px /   │     (fluid)                      │  320px            │
│ 240px    │                                  │  (collapsible)    │
│          │                                  │                   │
└──────────┴──────────────────────────────────┴───────────────────┘
```

**Sidebar-Gruppen:**
```
ENTDECKEN
  Dashboard          /dashboard
  Scouting Mission   /mission       ← NEU
  Rankings           /rankings
  Hidden Gems        /hidden-gems
  Talent Tracker     /talents       ← NEU
  Transfer Intel     /transfers     ← NEU

ANALYSE
  Player Profile     /player/:id
  Compare Players    /compare
  Club Profiles      /club/:slug    ← NEU
  Team Scores        /teams

SCOUTING
  My Watchlist       /watchlist     ← NEU (Badge: count)
  My Reports         /reports       ← NEU

ÜBER
  Methodology        /methodology   ← NEU
  Settings           /settings      ← NEU
```

**Sidebar-Verhalten:**
- Default: Collapsed (72px, nur Icons + Tooltips)
- Hover / Pin: Expanded (240px mit Labels)
- Keyboard: `[` togglet Sidebar
- Active: Accent-farbiger linker Border

### 3.2 Farbpalette — Neue Token-Ergänzungen

Die bestehenden Apple Dark Tokens bleiben. Ergänzt werden:

```css
/* Neue Struktur-Tokens */
--color-bg-sidebar:    #111113;   /* Sidebar, dunkler als Content */
--color-bg-panel:      #161618;   /* Shortlist-Panel */
--color-bg-card:       #1C1C1E;   /* Karten, Tabellen */
--color-bg-hover:      #242426;   /* Row-Hover */

/* Signal-Farben (funktional, nicht dekorativ) */
--color-signal-up:     #2ECC71;   /* Score steigt, positiver Trend */
--color-signal-down:   #E74C3C;   /* Verletzung, negativer Trend */
--color-signal-warn:   #F39C12;   /* Alert: Vertrag, Warnung */

/* Score-Bands — eine Anpassung */
--color-band-quality:  #FF9F0A;   /* Orange statt Gelb — besser lesbar */
```

**Farbregel:** Jede Farbe hat genau eine Bedeutung. Grün/Rot ausschließlich für Trend. Amber ausschließlich für Alerts. Accent Blue ausschließlich für interaktive Elemente.

### 3.3 Typographie: Inter + JetBrains Mono

Inter ist der Standard für professionelle Data-Apps (Linear, Vercel, Pitch). JetBrains Mono für alle Zahlen und Scores — Monospace erhöht Scannbarkeit in Tabellen dramatisch.

```css
/* Typographie-Skala */
--text-2xl:  2rem,     weight 700, letter-spacing -0.03em  /* Page Titles */
--text-xl:   1.5rem,   weight 700, letter-spacing -0.02em  /* Section Titles */
--text-lg:   1.125rem, weight 600, letter-spacing -0.01em  /* Card Titles */
--text-base: 0.9375rem, weight 400                         /* Body */
--text-sm:   0.8125rem, weight 400                         /* Labels, Meta */
--text-xs:   0.6875rem, weight 500, uppercase, ls 0.08em   /* Column Header */

/* Score-Darstellung — Monospace */
--font-score: 'JetBrains Mono', monospace
--text-score-xl: 3rem,   weight 700   /* Haupt-Score in Profile */
--text-score-lg: 1.75rem, weight 600  /* Score in Rankings-Row */
--text-score-sm: 1rem,   weight 500   /* Score in Shortlist Card */
```

---

## 4. Dashboard — Neue Startseite (ersetzt Home)

Der Hero wird zur Auth-geschützten Dashboard-View. Eingeloggte User sehen sofort ihre Arbeit.

```
┌──────────────────────────────────────────────────────────┐
│  Good morning, Markus.  Season 2025/26  [League: All ▾]  │
├────────────────────┬──────────────────┬──────────────────┤
│  WATCHLIST         │  RECENTLY VIEWED │  ALERTS          │
│  ─────────────     │  ────────────    │  ────────        │
│  14 players        │  Balde     ↗     │  3 Score-Updates │
│  3 lists active    │  Bellingham →    │  2 Transfer-Hints│
│  Last: "CB Optns"  │  Diaz, L.  ↘    │  1 New Match     │
├────────────────────┴──────────────────┴──────────────────┤
│  TOP SCORE MOVERS — DIESE WOCHE                          │
│  ┌──────────┬──────────┬──────────┬──────────┐          │
│  │ +47 pts  │ +38 pts  │ -29 pts  │ +22 pts  │          │
│  │ Musiala  │ Bellngham│ Griezm.  │ Mbappé   │          │
│  │ MF · BL  │ MF · PL  │ FW · SA  │ FW · LA  │          │
│  └──────────┴──────────┴──────────┴──────────┘          │
│                                                          │
│  QUICK ACTIONS                                           │
│  [Neue Mission]  [Rankings öffnen]  [Report erstellen]  │
│                                                          │
│  MEINE LETZTEN REPORTS                    [Alle →]       │
│  LB Options · vor 2h    CB Sommer 2026 · gestern        │
└──────────────────────────────────────────────────────────┘
```

---

## 5. Neue Seiten — Vollständige Spezifikation

### 5.1 Scouting Mission `/mission` — P0

Geführter Workflow statt Free-Form-Filter. Scout definiert ein Anforderungsprofil in 5 Schritten.

**Step-Flow:**
1. Position wählen (klickbarer Fussballplatz als SVG, nicht Dropdown)
2. Alter-Range (Schieberegler + Presets: U21 / U23 / U25 / Erfahren)
3. Mindest-Score (mit Live-Preview "47 Spieler erfüllen das")
4. Budget / Marktwert-Ceiling (optional, "Freie Transfers"-Toggle)
5. Liga-Fokus (Multi-Select, Presets: "Big 5" / "Unter dem Radar")
→ Ergebnis: Priorisierte Trefferliste mit "Zur Watchlist" + "Profil" + "Report" pro Zeile

**Daten:** `player_scores_all_seasons_long.csv` + `player_market_values.csv`  
**Key feature:** Mission speicherbar → erscheint in Watchlist als aktives Board

---

### 5.2 Watchlist `/watchlist` — P0

Persistentes Scouting-Board. Das zentrale operative Feature.

**Layout: Drei-Spalten Kanban**
```
[ Beobachte ich ]    [ Shortlist ]    [ Empfohlen an SD ]
     15 Spieler           5 Spieler          3 Spieler

┌─ Spieler-Card ───────────────────────────────────┐
│ Alejandro Balde          FC Barcelona · LV · 21   │
│ Score: 762 ▲+18 seit Hinzufügen  [World Class]    │
│ Marktwert: 28M€ ▼-2M€  |  ★★★★☆                  │
│ "Gutes Dribbling/90, aber schwache Luftzweikämpfe"│
│ [Profil öffnen]  [Vergleichen]  [Aus Liste entf.] │
└───────────────────────────────────────────────────┘
```

**Datenmodell (localStorage Phase 1):**
```js
{
  player_name, season, score_snapshot, score_current,
  market_value_snapshot, market_value_current,
  added_date, board, note, tags, priority
}
```

**Actions:**
- Drag & Drop zwischen Boards
- Score-Delta seit Hinzufügen (grün/rot)
- `[Shortlist als PDF exportieren]` — Vergleichstabelle aller Spieler
- `[CSV Export]` für Excel
- `[Board teilen]` — URL-encodierter State (kein Login nötig)

**Einstiegspunkte:** Jede Rankings-Zeile, jeder Player-Card, Mission-Ergebnisse, Hidden Gems → konsistentes Bookmark-Icon (gefüllt wenn bereits auf Watchlist)

---

### 5.3 Club Profile `/club/:slug` — P0

Tiefes Kader-Portrait — nicht nur Team-Score sondern vollständiges Squad-Profil.

**5 Tabs:**

**Tab 1 — Kader:** Alle Spieler der Saison in sortierbarer Tabelle (Name, Pos, Alter, Score, Band, Marktwert, Spielzeit-%). Klick → Player Profile. "+ Watchlist" pro Zeile.

**Tab 2 — Team-DNA (Radar):** 6 Achsen: Offensive Output, Midfield Control, Defensive Solidity, Youth Index (Anteil U24 am Squad-Score), Depth Score (12.–20. Mann vs. Startelf), Consistency (Score-StdDev). Benchmark: Liga-Durchschnitt als zweite Linie.

**Tab 3 — Altersstruktur:** Dot-Plot: Spieler X-Achse nach Alter, Y = Score, Farbe = Position. Referenzlinien bei 23 und 29. Sofort erkennbar: "Kader zu alt", "Kader zu jung", "ausgewogen".

**Tab 4 — Score-History:** Linienchart Club-Average-Score pro Saison (2017/18–heute). "Hat der neue Trainer das Team besser gemacht?"

**Tab 5 — Positions-Tiefe:** Pro Positions-Gruppe: Spieleranzahl, Durchschnitt-Score, Starter/Backup. "LV: 1 Spieler, Score 680 — Single Point of Failure"

**Daten:** `squad_scores_all_seasons.csv` + `player_scores_all_seasons_long.csv` (gefiltert auf Club)

---

### 5.4 Talent Tracker `/talents` — P1

Spezialisierte Seite für Nachwuchstalente. Entwicklungsfokus statt Absolut-Ranking.

**Sektionen:**
- **Talents-Ranking:** Default U23 gefiltert, sortierbar nach Score-Anstieg (nicht Absolut-Score)
- **Age-Adjusted Percentile (AAP):** "Score 581 — Top 8% unter 21-Jährigen" — berechnet als Perzentil innerhalb Alters-Kohorte ±1 Jahr
- **Entwicklungstrend:** Mini-Sparkline pro Spieler (letzten 3 Saisons wenn vorhanden)
- **Historical Comparison:** "Wie war Pedri mit 20? Wie war Kimmich mit 22?" — Benchmarking gegen historische Stars im gleichen Alter
- **Breakout-Kandidaten:** Score-Anstieg > 100 Punkte gegenüber Vorjahr → "Im Durchbruch"

**Neue Berechnung — Wachstumsrate:**  
`+X Punkte/Jahr` als lineare Extrapolation. Ein 21-Jähriger mit +80/Jahr → mit 24 auf World-Class-Kurs.

**Daten:** Alle `player_scores-*.csv` (historisch), `player_scores_all_seasons_long.csv`

---

### 5.5 Transfer Intelligence `/transfers` — P1

"Why now"-Argumente aus vorhandenen Daten — keine Vertragsdaten nötig.

**Drei berechnete Scores:**

**Undervalue Signal (UV):**
```
UV = (Score-Rang in Liga) − (Marktwert-Rang in Liga)
Positiv = Score höher als Marktwert suggeriert → Kaufgelegenheit
```

**Exit-Risk (ER):**
```
Alter ≥ 30 AND Score-Trend negativ    → +30
Spielzeit < 50% letzte Saison         → +25
Marktwert-Rückgang > 20% in 1 Jahr   → +25
Peak-Performance vor ≥ 2 Saisons      → +20
Summe 0–100, über 60 = Transfer-Kandidat
```

**Breakthrough (BT):**
```
BT = Score-Anstieg letzte Saison / Age-adjusted Expected Improvement
> 1.0 = schneller als erwartet entwickelt
```

**Layout:** Drei Panels nebeneinander: "Kaufoptionen (UV hoch)", "Mögliche Abgänge (ER hoch)", "Durchbruch-Talente (BT hoch)". Top 10 pro Panel, gefiltert nach Liga/Position/Alter.

**Daten:** `player_scores_all_seasons_long.csv` + `player_market_values.csv`

---

### 5.6 Methodology `/methodology` — P1

Vollständige Transparenz der Scoring-Logik. Der USP des Produkts — aber bisher nirgendwo erklärt.

**Inhalte:**
- Warum kein ML (Kurztext: "Jedes Gewicht ist eine bewusste Entscheidung")
- Rollen-Tabelle (FW/Off_MF/MF/Def_MF/DF): welche Metriken, welche Gewichte, welche Benchmarks
- Score-Formel mit konkretem Rechenbeispiel (z.B. Bellingham)
- Datenquellen und Update-Rhythmus (wöchentlich, Dienstag 06:00 UTC)
- Band-Verteilung live (datenbankbasiert, nicht statisch)
- Bekannte Limitierungen (kein GK, Big-5 only, ab 2017)

---

### 5.7 Reports `/reports` + `/reports/new` — P1

**`/reports`:** Archiv aller generierten PDFs mit Erstellungsdatum, Download, erneut generieren.

**`/reports/new` — Report Builder:**
1. Template wählen: Kurzprofil (1 Seite) / Vollprofil (3 Seiten) / Shortlist-Report
2. Spieler auswählen (aus Watchlist oder Suche)
3. Metriken konfigurieren (Checkboxen)
4. Preview
5. PDF-Export

**PDF-Seitenstruktur (Vollprofil):**
- Seite 1: FIFACard-Snapshot + Score-Breakdown + Radar-Chart (via html2canvas)
- Seite 2: Career-Trajectory-Chart + Metriken-Tabelle (Stärken/Schwächen)
- Seite 3: Scouting-Notiz (Freitextfeld das Scout vor Export ausfüllt)
- Optional Seite 4: Vergleichstabelle mit 2 weiteren Kandidaten

---

### 5.8 Settings `/settings` — P2

User-Präferenzen: Default-Saison, Default-Liga, Default Scout-Mode Position, Compact/Comfortable Toggle, Club-Logo Upload (White-Label für Reports).

---

## 6. Überarbeitete bestehende Seiten

### Rankings `/rankings`
- **Table-first:** 1 Row = 1 Spieler, max 8 Spalten sichtbar + Column-Picker
- **Spalten:** # | Player | Club | League | Pos | Age | Score (Mono) | Band | Trend (60px Sparkline) | 90s | [★]
- **Filter-Pill-Bar** (persistent, über Tabelle): `[Season: 2024/25 ×] [League: PL ×] [+ Add Filter]`
- **Row-Expand:** Hover/Click öffnet Mini-Preview in-place. Nur bei tiefem Dive: Full Profile
- **Sticky:** Header + Filter-Pills scrollen mit

### Player Profile `/player/:id`
Wechsel zu **Tab-Layout** statt einer langen Scroll-Page:
- Tab 1 — Overview: FIFACard + Score-Breakdown + AAP
- Tab 2 — Metrics: Pizza-Chart + Percentile-Strip-Chart
- Tab 3 — History: Career-Trajectory + Pre/Post-Transfer
- Tab 4 — Notes: Scout-Notizen (Rich-Text, Tags, Export)

3-Column-Layout auf Desktop:
- Links (280px): Player-Meta (Name, Club, Position, Alter, Marktwert, Saison-Selector)
- Mitte (fluid): Score + Charts + Metrics (tabs)
- Rechts (320px): Watchlist-Panel + Vergleich-Shortcuts

### Compare Players `/compare`
- Cross-Season Compare: "Haaland 2022/23" vs. "Lewandowski 2019/20"
- Drei-Spieler-Vergleich (nicht nur 1vs1)
- Radar Chart überlagert beide/alle Spieler in einer Ansicht
- "Add to Report"-Button direkt aus Compare

### Hidden Gems `/hidden-gems`
- Scatter Plot (X: Marktwert log, Y: Score) als Haupt-View mit Zoom
- UV-Score (Undervalue Signal) als primäre Sortierung
- Quick-Add zur Watchlist direkt aus Scatter Dot

### Team Scores `/teams`
- Liga-Tabelle als Haupt-View (Row-Click expandiert Team-Detail in-place)
- Top-5 Spieler + Average Score + Score-Distribution direkt im Expander
- Deep-Link zu Club Profile

---

## 7. Neue UI-Komponenten

### A. Shortlist Panel (persistent rechts, 320px)
Bereits spezifiziert in 5.2. Erscheint auf allen Seiten außer Methodology/Settings.

### B. Score Sparkline (60px, 5-Season-Trend)
Inline in Rankings-Rows. Vital für Scouts — Trajektorie statt Snapshot.

### C. Global Command Palette (Cmd+K)
```
[ 🔍 Spieler, Teams, Reports suchen... ]
ZULETZT GESEHEN: Balde · Bellingham
AKTIONEN: Neue Mission | Shortlist öffnen | Rankings (PL · FW)
```

### D. Filter Pill Bar
`[Season: 2024/25 ×] [Liga: PL ×] [Pos: FW ×] [Min 90s: 10 ×] [+ Filter]`
Persistent über Sessions (localStorage).

### E. Scout Mode Toggle (NavBar rechts)
`[Scout Mode: AUS]` → `[Scout Mode: MF ▾]`  
Wenn aktiv: Rankings, Hidden Gems, Talent Tracker filtern automatisch auf diese Position. Scouting Report passt Metriken an.

### F. Age-Adjusted Percentile Badge
Auf FIFACard unter Haupt-Score (nur U24):  
`Score: 647   Top 11% unter 22-Jährigen`

### G. Positional Heat Map (auf Player Profile)
Statische SVG-Pitch-Map mit Spieler-Position. Standard in Wyscout — fehlt komplett in PlayerScore.

### H. Scouting Notes (Tab 4 in Player Profile)
Rich-Text-Editor, Timestamp, Kategorie-Tags (Technik / Athletik / Charakter / Potential / Fit), Export-fähig.

---

## 8. Neue Visualisierungen

### Viz 1: Score-Trajectory mit Kontext-Annotationen
X = Saison, Y = MainScore, Farbkodierung = Band.  
Annotations: Vereinswechsel (Squad ändert sich), Verletzungssaisons (Min < 500).  
**Daten:** `Season`, Score-Spalten, `Age`, `Squad`, `Min`

### Viz 2: Bubble-Chart (Score × Marktwert × Alter)
X = MarketValue (log), Y = MainScore, Bubble = 90s, Farbe = Position.  
Klassisches Value-Identification-Chart. Unten-rechts = Hidden Gems.  
**Daten:** Merge `player_scores_all_seasons_long.csv` + `player_market_values.csv`

### Viz 3: Team-Profil-Heatmap
Zeilen = Teams, Spalten = Off/Mid/Def/Intensity/Overall-Score. Farbe = Wert (normiert pro Saison).  
**Daten:** `squad_scores_all_seasons.csv`

### Viz 4: Percentile-Strip-Chart
Pro Metrik eine horizontale Linie (0–100. Perzentil), Spieler als Punkt. Alle 14 Pizza-Metriken untereinander. Liga + Pos + Season gefiltert.  
FBref-ähnlich — aber mit Liga-Filter und Mehrjahres-Vergleich.  
**Daten:** Alle `_Per90`-Spalten aus `player_pizza_all_seasons.csv`

### Viz 5: Altersstruktur-Dot-Plot (Club Profile)
X = Alter, Y = MainScore, Farbe = Position. Referenzlinien bei 23 und 29.  
Zeigt Kaderbalance auf einen Blick.

---

## 9. Neue Daten-Features & Berechnungen

Alle aus vorhandenen Spalten realisierbar — kein neues Scraping nötig.

| Feature | Formel / Logik | Spalten |
|---------|---------------|---------|
| **Undervalue Signal** | Score-Rang − Marktwert-Rang | `MainScore`, `MarketValue_EUR` |
| **Exit-Risk-Score** | Gewichtete Summe (Alter, Spielzeit, MV-Delta, Trend) | `Age`, `Min`, `MarketValue_EUR`, Score-Delta |
| **Breakthrough-Score** | Score-Anstieg / Age-adjusted Expected | Season-History, `Age` |
| **Age-Adjusted Percentile** | Perzentil innerhalb Alters-Kohorte ±1 Jahr | `Age`, `MainScore` |
| **Wachstumsrate** | Lineare Regression auf Score über Saisons | Season-History |
| **Consistency-Index** | 1 − (StdDev/Mean) über ≥ 3 Saisons | Season-History |
| **Involvement Rate** | Carries/90 + PrgC/90 + KP/90 | Pizza-Metriken |
| **Overperformance** | Gls/90 − xG/90 | `Gls_Per90`, `xG_Per90` |
| **Spielzeit-Trend** | Min_Saison_N / Min_Saison_N-1 | `Min` per Season |
| **Squad Depth Score** | Score-Diff Starter (Top-11) vs. Rest | `player_scores_all_seasons_long.csv` per Club |

**Bekannte Datenlücken:**
- `player_market_values.csv` hat kein Season-Feld → Matching über `Player` + `Squad` (MatchScore-Spalte hilft)
- `IntensityScore` nur auf Squad-Ebene, nicht auf Spieler-Ebene
- Kein Gehaltsdata (größte fehlende Dimension für Value-Scouting — würde Capology-Scraping erfordern)

---

## 10. Was NICHT gebaut wird (Scope Discipline)

- **Vertragsdaten:** Nicht zuverlässig verfügbar; ein Fehler hier zerstört die Glaubwürdigkeit des Tools
- **Video-Integration:** Wyscout gewinnt mit Video — PlayerScore gewinnt mit Transparenz und Preis
- **Live-Daten / Match-by-Match:** Wöchentliche Updates genügen für Scout-Entscheidungen die Wochen dauern
- **ML-Empfehlungen:** Widerspricht dem USP "keine Black Box"
- **Torhüter-Scoring:** Braucht andere Metriken-Struktur; halb-fertige GK-Implementierung schadet mehr als keine
- **Social Features:** Kein Consumer-Produkt, Scouts wollen keine öffentlichen Watchlists
- **Native App (iOS):** Erst nach Web-Parität

---

## 11. Mobile & Responsive

**Primäres Device: Desktop (1440px)** — Scouts arbeiten am Laptop im Büro.  
**Sekundär: iPad** — Matchday auf der Tribüne (Watchlist lesen, Notizen schreiben).

```
Desktop (≥1280px)  — 3-Column: Sidebar + Content + Shortlist Panel
Laptop (≥1024px)   — 2-Column: Sidebar collapsed + Content; Panel als Overlay
Tablet (≥768px)    — Top-Bar + Content; Sidebar als Drawer; Panel als Bottom Sheet
Mobile (<768px)    — Single Column; Sidebar als Full-Screen-Menu
```

**Mobile-Prioritäten:** 1. Spieler-Suche + Score, 2. Watchlist lesen/ergänzen, 3. Notizen schreiben.

---

## 12. User Flow — Haupt-Use-Case in ≤ 30 Sekunden

**"Suche einen LV unter 24, Score > 650, unter 20M€"**

1. Sidebar → "Scouting Mission" (1 Klick)
2. LV auf Pitch-SVG anklicken (1 Klick)
3. Alter-Preset "U24" (1 Klick)
4. Score-Slider auf 650 (2 Sek.)
5. Budget-Input 20M€ (3 Sek.)
6. "Mission starten" → Ergebnisliste erscheint (< 1 Sek., CSV-Daten)
7. Top-Kandidat → "+ Watchlist" (1 Klick)

**Gesamt: 5 Klicks + ~5 Sekunden. Keine Page-Reloads.**

---

## 13. Priorisierter Build-Plan

### Phase 0 — Foundation (1 Woche)
- Sidebar-Navigation implementieren
- `PageShell` auf 3-Column-Layout erweitern
- Inter + JetBrains Mono einbinden
- Neue CSS-Tokens ergänzen

### Phase 1 — Kern-Features (Woche 2–4)
- **Dashboard** `/dashboard` (ersetzt Home)
- **Watchlist** `/watchlist` (localStorage, Kanban)
- **Watchlist-Einstiegspunkte** überall (Rankings, HiddenGems, Profile)
- **Methodology** `/methodology` (Content-Seite)
- **Filter-Pill-Bar** (persistent, Rankings)
- **Command Palette** Cmd+K

### Phase 2 — Scouting-Features (Woche 5–8)
- **Scouting Mission** `/mission` (5-Step-Flow + Pitch-SVG)
- **Club Profile** `/club/:slug` (5 Tabs)
- **Age-Adjusted Percentile** (Berechnung + FIFACard-Badge)
- **Score Sparkline** (Rankings-Rows)
- **Scout Mode Toggle** (NavBar, localStorage)
- **PDF Report v2** (FIFACard + Pizza-Chart via html2canvas)

### Phase 3 — Analytics-Features (Woche 9–12)
- **Talent Tracker** `/talents` (AAP, Breakout-Kandidaten)
- **Transfer Intelligence** `/transfers` (UV/ER/BT-Scores)
- **Player Profile** überarbeiten (Tabs, 3-Column-Layout, Notizen)
- **Percentile-Strip-Chart** (Pizza-Metriken im Kontext)
- **Watchlist PDF-Export** (Shortlist-Vergleichsreport)
- **Mission speichern** (persistente Missions-Boards)

### Phase 4 — B2B-Launch (ab Woche 13)
- FastAPI Backend + User Accounts
- Cloud-Watchlist (geräteübergreifend)
- Team-Sharing für Shortlists
- White-Label-Support (Club-Logo in Reports)
- Notifications (Score-Updates, Transfer-Alerts)

---

## 14. Zusammenfassung — Die 7 wichtigsten Entscheidungen

| Entscheidung | Vorher | Nachher | Begründung |
|---|---|---|---|
| Navigation | Top-Nav, 6 Links | Collapsible Sidebar | B2B-Standard, skaliert auf 20+ Items |
| Startseite | Marketing Hero | Kontextuelles Dashboard | Eingeloggte User brauchen Daten, kein Marketing |
| Datendarstellung | Feature Cards | Dense Tables + Row-Expand | Experten scannen, nicht lesen |
| Persistenz | Stateless | Watchlist + Notes (localStorage) | Scouts arbeiten in Projekten, nicht in Pages |
| Typographie | SF Pro | Inter + JetBrains Mono | Software-Produkt-Signal, Monospace für Zahlen |
| Score-Kontext | Absolut (0–1000) | + Age-Adjusted Percentile | Talente interpretierbar machen |
| Export | Einfacher HTML-Report | PDF mit FIFACard + Pizza + Notes | Report muss in Coaching-Meeting bestehen |

**Die Kernbotschaft:** PlayerScore V2 muss sich wie ein **Werkzeug** anfühlen, nicht wie eine **Website**. Scouts öffnen es wie sie Wyscout öffnen — mit einem klaren Auftrag. Das Design unterstützt diesen Workflow, nicht unterbricht ihn.
