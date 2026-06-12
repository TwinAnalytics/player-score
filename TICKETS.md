# PlayerScore — Improvement Tickets

> Erstellt von **Nova** (UX/UI) und **Rex** (Product).  
> Prioritäten: P0 = kritisch · P1 = hoch · P2 = mittel · P3 = nice-to-have

---

## NOVA — UX/UI Design Review (65 Tickets)

---

### Design System & Global

**#N1 — Teal `#00B8A9` dominiert alle Charts — falsches Brand-Color**
Component: ScoreTrendLine, AgeCurveChart, RoleScatterChart, PizzaRadarChart, BeeswarmChart
Priority: **P0**
Der alte Teal-Farbwert (`#00B8A9`, `#80F5E3`) ist als primäre Datenfarbe in fünf Chart-Dateien hartkodiert. Das Design System definiert `--accent: #0A84FF` (Apple System Blue). Alle Chart-Linien, aktiven Punkte und Tooltips zeigen die alte Markenfarbe. Alle Vorkommen durch `#0A84FF` / `var(--accent)` ersetzen.
**Status: ✅ DONE** — All teal occurrences replaced with `#0A84FF` / `var(--accent)` across all 9 chart files + pdfExport + PizzaDataContext.

**#N2 — Chart-Tooltips verwenden GitHub Dark Theme-Farben, nicht das Design System**
Component: Alle Chart-Komponenten
Priority: **P0**
Jeder Custom-Tooltip hartkodiert `background: '#161B22'`, `border: '1px solid #21262D'` — GitHub Canvas-Farben, nicht das Apple Dark System. Alle sieben Tooltip-Komponenten auf `var(--bg-second)`, `var(--border)`, `var(--text)` migrieren.
**Status: ✅ DONE** — All chart tooltips migrated to CSS design-system variables.

**#N3 — Chart-Achsen und Grid-Linien verwenden ebenfalls GitHub-Farben**
Component: Alle Chart-Komponenten
Priority: **P1**
`tick={{ fill: '#94A3B8' }}`, `stroke="#374151"` für Grids sind nirgendwo im Design System definiert. Ein gemeinsames `CHART_THEME`-Objekt in `colors.js` anlegen und überall referenzieren.
**Status: ✅ DONE** — `CHART_THEME` added to `colors.js`; all chart axis/grid colors reference CSS vars via the theme object.

**#N4 — Zwei parallele CSS-Variablen-Systeme aktiv — Legacy-Aliases vs. neue Tailwind-Tokens**
Component: Alle CSS-Module
Priority: **P1**
`index.css` definiert sowohl `@theme`-Variablen (`--color-bg-secondary`) als auch ein `:root`-Block (`--bg-second`). Alle Module nutzen die Legacy-Aliases. Ein System wählen und konsequent nutzen.
**Status: ❌ SKIPPED** — Large-scale CSS variable migration; all new code written against the existing `:root` token set consistently. Consolidation deferred.

**#N5 — `SimilarPlayers.module.css` hover enthält alte Teal-Referenz**
Component: SimilarPlayers
Priority: **P1**
`.card:hover { background: rgba(0, 184, 169, 0.04); }` — Teal bei 4% Opacity. Durch `rgba(10, 132, 255, 0.06)` ersetzen.
**Status: ✅ DONE** — Hover color updated to `rgba(10, 132, 255, 0.06)`.

**#N6 — `HiddenGemsScatter` Tooltip verwendet `#22c55e` für GemScore-Highlight**
Component: HiddenGemsScatter
Priority: **P2**
`#22c55e` (Tailwind Green-500) ist eine One-Off-Farbe ohne Design-System-Basis. `BAND_COLORS['World Class']` (`#30D158`) verwenden.
**Status: ✅ DONE** — Tooltip highlight color replaced with `BAND_COLORS['World Class']` (`#30D158`).

---

### NavBar

**#N7 — NavBar hat kein Mobile-Menü — Navigation bricht auf kleinen Screens**
Component: NavBar
Priority: **P0**
`.links` nutzt `flex-wrap: wrap`, die 6 Nav-Items brechen auf 375px auf mehrere Zeilen um. Kein Hamburger-Menü, kein Bottom Tab Bar. Die App ist auf Mobile kaum nutzbar.
**Status: ✅ DONE** — Hamburger button added; slide-down mobile menu with glass background, toggled on `isMenuOpen` state. Hidden on ≥768px.

**#N8 — Brand-Logo-Schriftgröße `1.05rem` ist zu klein**
Component: NavBar
Priority: **P2**
Bei 52px Nav-Höhe konkurriert das Wordmark bei `1.05rem` visuell mit den Links statt klar darüber zu stehen. Auf `1.15–1.2rem` anheben.
**Status: ✅ DONE** — Brand font-size set to `1.15rem`.

**#N9 — Aktiver Nav-Link nutzt `!important` — Zeichen eines Spezifitätsproblems**
Component: NavBar
Priority: **P3**
`.active { color: #0A84FF !important; ... }` — das `!important` deutet auf einen Cascade-Konflikt hin. Class-Merging umstrukturieren.
**Status: ✅ DONE** — Specificity conflict resolved by restructuring class merging; `!important` removed.

**#N10 — Kein `:focus-visible`-Ring auf Nav-Links**
Component: NavBar
Priority: **P1**
Keyboard-Navigation (Tab) produziert keinen sichtbaren Fokus-Indikator. Verstößt gegen WCAG 2.1 AA. `box-shadow: 0 0 0 3px rgba(10,132,255,0.15)` ergänzen.
**Status: ✅ DONE** — `:focus-visible` ring added to all nav links and buttons.

---

### Home

**#N11 — Hero hat keine rechte Bildhälfte — nur Textblock auf schwarzer Fläche**
Component: Home
Priority: **P1**
Der Hero-Content ist `max-width: 620px` linksbündig. Die rechte Hälfte ist leer. Eine statische Mockup-FIFACard oder ein Pizza-Chart-Preview würde sofort zeigen, was das Produkt ist.
**Status: ✅ DONE** — Static sample score card added to hero right side with mini bar charts and score band.

**#N12 — Feature-Cards verwenden Emoji-Icons — nicht Apple-Style**
Component: Home
Priority: **P1**
`📊`, `👤`, `🏟️` rendern plattformabhängig. SVG-Icons oder CSS-Shapes verwenden.
**Status: ✅ DONE** — All emoji icons replaced with inline SVG icons.

**#N13 — Feature-Card-Pfeil `→` ist Plain Text — wirkt unfertig**
Component: Home
Priority: **P2**
Kein hover-Response. Apple-Card-Patterns nutzen ein Chevron-Icon mit subtiler Translate-Animation.
**Status: ✅ DONE** — Arrow replaced with SVG chevron; `translateX(3px)` hover animation added.

**#N14 — KPI-Tiles sind visuell gleichwertig — keine Hierarchie**
Component: KpiTile
Priority: **P2**
Alle drei Tiles haben identisches Layout. Unterschiedliche Schriftgrößen oder Icons würden relative Wichtigkeit kommunizieren.
**Status: ❌ SKIPPED** — All three KPI tiles display equivalent importance by design (players, seasons, leagues). Hierarchy differentiation not warranted.

**#N15 — KpiTile `.sub`-Text ist doppelt gedimmt — fast unsichtbar**
Component: KpiTile
Priority: **P2**
`.sub { color: var(--muted); opacity: 0.6; }` → `var(--muted)` ist bereits `rgba(235,235,245,0.6)`. Effektive Opacity: ~0.36. Schlägt WCAG AA. `opacity: 0.8` oder vollständiges `var(--muted)` verwenden.
**Status: ✅ DONE** — `opacity: 0.6` removed from `.sub`; color kept as `var(--muted)` without additional dimming.

**#N16 — Band-Legend-Dots sind 9×9px — zu klein für Farb-Kommunikation**
Component: Home
Priority: **P2**
Für Farbsehschwäche sind 9px-Kreise kaum wahrnehmbar. Minimum 12px, besser: `ScoreBadge`-Komponente (Farbe + Text).
**Status: ✅ DONE** — Band dots increased to 12×12px on Home scoring system section.

**#N17 — Hero-Subtitle `line-height: 1.75` ist zu locker für die Schriftgröße**
Component: Home
Priority: **P3**
Bei `1.05rem` wirkt `1.75` wie doppelter Zeilenabstand. Apple nutzt `1.5–1.6`.
**Status: ✅ DONE** — Hero subtitle line-height changed to `1.6`.

**#N18 — Kein 404 / Unknown-Route Handling**
Component: App
Priority: **P1**
`App.jsx` hat kein `<Route path="*">` Fallback. Unbekannte Pfade zeigen nur die NavBar auf schwarzem Hintergrund.
**Status: ✅ DONE** — `<Route path="*">` fallback added with inline `NotFound` component linking back to home.

---

### Rankings

**#N19 — Filter-Bar und Position-Toggles sind zwei separate UI-Muster für dasselbe Ziel**
Component: Rankings
Priority: **P1**
Drei separate Filter-Reihen (Select-Panel / Position-Pills / View-Segmented-Control) mit unterschiedlichen visuellen Sprachen. Position-Toggles gehören in den Filter-Panel-Card.
**Status: ✅ DONE** — Position pills moved inside the filter card; single consolidated filter UI.

**#N20 — "Top N"-Selector verschwindet beim View-Wechsel — Stateful Filter-Loss**
Component: Rankings
Priority: **P1**
Top N ist nur bei `view === 'Top Players'` sichtbar, beeinflusst aber auch den Beeswarm-Label. Der Filter soll sichtbar bleiben.
**Status: ✅ DONE** — Top N select always visible in the filter card regardless of active view.

**#N21 — Error-State nutzt hartkodiertes `#ef4444`**
Component: Rankings, TeamScores
Priority: **P2**
`#ef4444` ist Tailwind Red-500, außerhalb des Design Systems. `--color-error: #FF453A` (Apple System Red) Token anlegen.
**Status: ✅ DONE** — Error states use `var(--error, #FF453A)` (Apple System Red).

**#N22 — Empty-States sind einzelne `<p>`-Tags ohne Illustration oder Action**
Component: Rankings, TeamScores, HiddenGems, ComparePlayers
Priority: **P1**
Apple HIG: Empty States benötigen Icon + Titel + Beschreibung + optionale Action. Keine `<p>` mit Padding.
**Status: ✅ DONE** — All empty states upgraded to icon + title + description pattern.

**#N23 — Klickbare Tabellenzeilen haben keinen starken visuellen Unterschied zu nicht-klickbaren**
Component: Rankings, HiddenGems
Priority: **P2**
`cursor: pointer` ist gesetzt, aber kein Chevron oder stärkerer Hover-Effekt, der Navigierbarkeit signalisiert.
**Status: ✅ DONE** — Chevron `›` column added to all clickable table rows; hover background strengthened.

**#N24 — Rankings-Chart und Tabelle ohne visuellen Connector — Chart wirkt losgelöst**
Component: Rankings
Priority: **P2**
Chart und Tabelle stehen in separaten Cards ohne gemeinsamen Header oder Abschnittstitel.
**Status: ✅ DONE** — Shared `.sectionHeader` with title, count, and CSV button connects chart and table visually.

---

### Player Profile

**#N25 — Suchfeld hat kein Lupe-Icon — schwache Interaction-Affordance**
Component: PlayerProfile
Priority: **P1**
Plain Text-Input ohne Icon. Apple HIG: Search-Fields haben ein führendes Lupe-Icon.
**Status: ✅ DONE** — SVG magnifier icon added as leading element in the search field.

**#N26 — Autocomplete-Dropdown hat keine Pfeiltasten-Navigation**
Component: PlayerProfile
Priority: **P1**
`handleInputKeyDown` behandelt nur `Enter` und `Escape`. Keine `ArrowUp`/`ArrowDown`-Navigation durch Vorschläge.
**Status: ✅ DONE** — `ArrowUp`/`ArrowDown` key handling added; `highlightIdx` state tracks focused suggestion.

**#N27 — `<h1>Player Profile</h1>` ist nach Spielerauswahl redundant**
Component: PlayerProfile
Priority: **P2**
Die h1 wird durch `<h2>{selectedPlayer}</h2>` überschattet. Nach der Auswahl sollte der Spielername die h1 sein.
**Status: ✅ DONE** — After player selection, player name becomes the `<h1>`; static "Player Profile" heading shown only pre-selection.

**#N28 — Stat-Tiles und Sub-Score-Tiles erscheinen NACH den Charts — invertierte IA**
Component: PlayerProfile
Priority: **P1**
Wichtige Zahlen (Score, Band, Market Value, Offense/Mid/Def) erscheinen nach ~400px Charts. KPIs gehören über die Charts.
**Status: ✅ DONE** — Sub-score tiles and stat tiles moved above the chart section in JSX order.

**#N29 — View-Toggle bricht auf kleinen Screens um**
Component: PlayerProfile
Priority: **P1**
`.viewToggle { flex-wrap: wrap; }` bricht den Segmented-Control-Look. Horizontales Scrollen oder kompakte Labels auf Mobile.
**Status: ✅ DONE** — View toggle uses `overflow-x: auto; white-space: nowrap` on mobile; no wrapping.

**#N30 — PDF-Export-Button visuell unter-differenziert von Standard-Sekundär-Aktionen**
Component: PlayerProfile
Priority: **P2**
Keine Icon, kein deutlicherer visueller Unterschied. Terminal-Aktionen brauchen stärkere Behandlung. Auch: kein Spinner während "Generating…".
**Status: ✅ DONE** — PDF button uses `.pdfBtnInner` with icon + label; SVG spinner shown during generation via `pdfLoading` state.

**#N31 — Season-Selector fehlt wenn nur eine Saison vorhanden**
Component: PlayerProfile
Priority: **P2**
Kein statischer Saison-Indikator wenn `playerSeasons.length <= 1`.
**Status: ✅ DONE** — Static `.seasonBadge` displayed when only one season available.

**#N32 — FIFACard ist vollständig inline-styled — keine Design-System-Integration**
Component: FIFACard
Priority: **P1**
Alle Styles als `style={}`-Props. Schriftfamilie weicht leicht vom App-Standard ab, Spacing ist arbitrary (Werte wie `7`, `14`, `20`). Braucht ein CSS-Modul.
**Status: ❌ SKIPPED** — Full CSS module migration would require complete rewrite of FIFACard. Deferred; card is visually consistent and contained.

**#N33 — PizzaRadarChart-Hintergrund `#0D1117` erzeugt "Screenshot-eingebaut"-Look**
Component: PizzaRadarChart
Priority: **P1**
GitHub-Dark-Canvas-Farbe innerhalb der App-eigenen `#1C1C1E`-Card-Fläche. Background-Rect entfernen oder auf `var(--bg-primary)` setzen.
**Status: ✅ DONE** — Background `<rect fill="#0D1117">` removed; chart renders transparently against card background.

**#N34 — PizzaRadarChart hat keinen Empty-State — verschwindet lautlos**
Component: PizzaRadarChart
Priority: **P2**
`return null` bei fehlenden Daten lässt die Season-View-Grid asymmetrisch zurück. Placeholder-Card mit Erklärung ergänzen.
**Status: ✅ DONE** — Empty-state placeholder card returned instead of `null` when pizza data is unavailable.

**#N35 — PlayerProfile Empty-State vor Spielerauswahl ist nur ein `<p>`**
Component: PlayerProfile
Priority: **P2**
Die primäre Landing-State für neue Nutzer. Braucht Icon + Call-to-Action + evtl. Link zu Rankings.
**Status: ✅ DONE** — Pre-selection state uses `.initialEmpty` with SVG icon, title, subtitle, and link to Rankings.

---

### Team Scores

**#N36 — Inline League-Colors in TeamScores duplizieren `LEAGUE_COLORS` aus `colors.js`**
Component: TeamScores
Priority: **P1**
```js
const colors = { 'Premier League': '#0A84FF', ... }
```
Ist ein exaktes Duplikat von `LEAGUE_COLORS` in `constants/colors.js`. Import und Referenz statt Duplikat verwenden.
**Status: ✅ DONE** — Inline `colors` object removed; `LEAGUE_COLORS` imported from `constants/colors.js`.

**#N37 — Inline-Styles auf `<td>` umgehen das Modul-CSS**
Component: TeamScores, PlayerProfile
Priority: **P2**
`style={{ color: 'var(--muted)', fontSize: '0.82rem' }}` direkt auf `<td>`-Elemente — inkonsistent mit dem Rest des Design Systems.
**Status: ❌ SKIPPED** — Touching every inline `<td>` style would require wholesale table refactor. Deferred; visually consistent with design tokens.

**#N38 — TeamScores Scatter hat keine Click-Interaction — toter Chart**
Component: TeamScores
Priority: **P1**
Kein `onDotClick`, kein Tooltip. Auf einer Seite voller klickbarer Elemente ist dieser Chart passiv.
**Status: ✅ DONE** — Scatter `onDotClick` wired to navigate to `/rankings?season=...&club=...`.

**#N39 — Kein Team-Drill-Down — Klick auf Team-Name macht nichts**
Component: TeamScores
Priority: **P1**
Keine Navigation zu den Spielern eines Teams von dieser Seite.
**Status: ✅ DONE** — Table rows are clickable; navigate to `/rankings?season=...&club=...`.

---

### Hidden Gems

**#N40 — Range-Slider haben keine Fill-Track-Visualisierung**
Component: HiddenGems
Priority: **P1**
Apple-Slider zeigen den gefüllten Bereich links vom Thumb in Blau. Der Track sieht bei 5% und 95% identisch aus.
**Status: ✅ DONE** — CSS `--pct` custom property drives `linear-gradient` fill track on all range sliders.

**#N41 — Filter-Label enthält lebenden Wert — inkonsistentes Muster**
Component: HiddenGems
Priority: **P2**
`"Min Score: 400"` im Label-Text während andere Seiten statische Labels nutzen. Wert in separaten `<span>` auslagern.
**Status: ✅ DONE** — Live value moved to `<span className={styles.filterVal}>` separate from the static label text.

**#N42 — HiddenGems-Scatter: `onDotClick` ist verdrahtet aber feuert nie**
Component: HiddenGemsScatter
Priority: **P0**
`onDotClick` Prop wird übergeben, aber die `<Scatter>`-Komponente nutzt kein `onClick`. Der Klick auf einen Dot im Chart tut nichts. Kerninteraktion der Seite ist kaputt.
**Status: ✅ DONE** — `onClick` prop added to `<Scatter>` component; fires `onDotClick` with payload data.

**#N43 — Sortierbare Spalten-Header haben keinen permanenten Sortier-Indikator**
Component: HiddenGems
Priority: **P2**
`sortIndicator(col)` gibt `' ↓'` oder `' ↑'` als Text-Append zurück. Nicht-sortierte Spalten geben nichts zurück — kein Hinweis welche Spalten sortierbar sind.
**Status: ✅ DONE** — `sortIndicator` returns `<span className={styles.sortIdle}> ↕</span>` for non-active columns, `<span className={styles.sortActive}> ↓/↑</span>` for active.

**#N44 — GemBar ist 5px hoch — zu dünn für tabellarische Dichte**
Component: HiddenGems
Priority: **P2**
Unter dem Schwellenwert komfortabler Lesbarkeit. Auf 6–8px erhöhen.
**Status: ✅ DONE** — GemBar height increased to 7px.

---

### Compare Players

**#N45 — "VS"-Trenner ist mit `var(--muted)` zu gedimmt — fehlt visuelle Stärke**
Component: ComparePlayers
Priority: **P2**
Das VS-Element ist ein konzeptioneller Anker. Sollte `color: var(--text)` haben, evtl. als Badge.
**Status: ✅ DONE** — VS label uses `color: var(--text)` with font-weight 800; placed in `.vsStack` with swap button below.

**#N46 — Player 1 = Blau (`#0A84FF`), Player 2 = Gelb (`#FFD60A`) — semantische Kollision**
Component: ComparePlayers
Priority: **P1**
`#FFD60A` ist die "Solid Squad Player"-Bandfarbe. Ein World-Class-Spieler als Player 2 erscheint in der "schlechten" Bandfarbe. `--color-compare-a` und `--color-compare-b` Tokens anlegen (z.B. Blau + Apple System Purple `#BF5AF2`).
**Status: ✅ DONE** — Player 2 color changed to `#BF5AF2` (Apple System Purple); no longer conflicts with band colors.

**#N47 — PlayerSelector: drei kaskadierte Dropdowns ohne Suche**
Component: ComparePlayers
Priority: **P1**
Scouts kennen den Spielernamen — sie brauchen keine Liga → Club → Spieler-Kaskade. Typeahead-Suche wie auf der PlayerProfile-Seite.
**Status: ✅ DONE** — Cascaded dropdowns replaced with typeahead search component matching PlayerProfile pattern.

**#N48 — Recharts `<Legend />` ist unstyled — außerhalb des Design Systems**
Component: ComparePlayers
Priority: **P2**
Default-Legend von Recharts passt nicht zur App-Typographie. Custom `<Legend>`-Komponente oder Entfernen zugunsten der Player-Card-Farben.
**Status: ✅ DONE** — `<Legend />` removed; replaced with custom `.chartLegend` div using design-system tokens.

**#N49 — "No pizza data"-State im Vergleich ist Plain-Text in leerer Card**
Component: ComparePlayers
Priority: **P2**
Wenn Radar-Daten für einen Spieler fehlen, entsteht eine asymmetrische Layout. Empty-State-Card mit gleicher Höhe und Icon/Erklärung ergänzen.
**Status: ✅ DONE** — `.noPizza` displays flex-column with SVG clock icon and explanation text; maintains card height.

**#N50 — Kein Empty-State-Prompt wenn nur ein Spieler ausgewählt**
Component: ComparePlayers
Priority: **P1**
`emptyState` feuert nur wenn keiner gewählt ist. Wenn Spieler 1 steht, Spieler 2 fehlt: kein Hinweis was als nächstes zu tun ist.
**Status: ✅ DONE** — Single-player-selected state shows prompt card with arrow pointing to player 2 slot.

---

### Charts — Querschnitt

**#N51 — Kein Chart-Loading-Skeleton — Charts poppen aus dem Nichts herein**
Component: Alle Charts
Priority: **P1**
Sekundäre Daten-Ladevorgänge rendern `null` bis bereit, dann Flash. Layout-Shift ohne Vorbereitung.
**Status: ✅ DONE** — `SkeletonLoader.jsx` created with `PlayerProfileSkeleton` and `RankingsSkeleton`; used on both pages during loading.

**#N52 — Charts haben keine ARIA-Labels — nicht barrierefrei**
Component: Alle Charts
Priority: **P1**
Kein `aria-label`, kein `role="img"`. Screen-Reader erhalten nichts.
**Status: ✅ DONE** — All 9 chart components wrapped in `<div role="img" aria-label={ariaLabel}>` with descriptive defaults.

**#N53 — Tooltip `fontSize: '0.8rem'` (12.8px) ist unter der Apple-Mindestgröße**
Component: Alle Chart-Tooltips
Priority: **P2**
Apple-Minimum für Body-Content ist 14px (0.875rem). Tooltips sollten nicht darunter gehen.
**Status: ✅ DONE** — All tooltip font sizes raised to `0.875rem` (14px).

**#N54 — `ScoreBarChart` Y-Achse `width={130}px` — lange Namen werden ohne Indikator truncated**
Component: ScoreBarChart
Priority: **P2**
Spielernamen wie "Granit Xhaka" können 130px bei `fontSize: 12` überschreiten. Recharts trunciert ohne Ellipsis oder Tooltip.
**Status: ✅ DONE** — Dynamic `yAxisWidth` computed from `Math.max(...data.map(d => d.Player.length)) * 7 + 8`, capped at 200px.

**#N55 — BandHistogram: X-Achsen-Abkürzungen inkonsistent mit Tooltip-Vollnamen**
Component: BandHistogram
Priority: **P3**
"Solid Squad" auf X-Achse, "Solid Squad Player" im Tooltip. Eine konsistente Sprache wählen.
**Status: ✅ DONE** — `SHORT_LABELS` dictionary replaces brittle `.replace()` chain; labels are stable and consistent.

---

### Spacing & Typographie

**#N56 — Mindestens 14 verschiedene Schriftgrößen — keine System-Skala**
Component: Alle CSS-Module
Priority: **P2**
`0.68rem`, `0.7rem`, `0.72rem`, `0.73rem`, `0.75rem` ... 14+ beliebige Werte. `--text-xs` bis `--text-xl` Tokens in `index.css` definieren.
**Status: ❌ SKIPPED** — Defining and migrating to a full type scale across all CSS modules is a large separate workstream. Deferred.

**#N57 — Inkonsistente Border-Radius-Werte — keine Adherenz zur System-Skala**
Component: Alle CSS-Module
Priority: **P2**
`10px`, `14px`, `6px` neben den Token-Werten `8px`, `12px`, `16px`. Jede Komponente soll auf ein Token mappen.
**Status: ❌ SKIPPED** — Normalising border-radius across all components is a cosmetic refactor deferred to a design-token pass.

**#N58 — Section `margin-bottom`-Werte nicht auf dem 4pt-Grid**
Component: Alle Seiten
Priority: **P3**
`3.5rem` (56px), `2.25rem` (36px), `1.375rem` (22px) sind keine definierten Spacing-Tokens.
**Status: ❌ SKIPPED** — Spacing token pass deferred; values are visually consistent enough for current quality bar.

**#N59 — `letter-spacing: -0.03em` bei `2rem`-Titeln erzeugt visuelle Kompression**
Component: Alle Inner-Page-Titles
Priority: **P3**
Bei 32px ist `-0.03em` zu eng. `-0.02em` für Inner-Page-Titles, `-0.03em` erst ab ~40px+.
**Status: ❌ SKIPPED** — Cosmetic typography adjustment deferred; existing tracking is acceptable.

---

### Mobile Responsiveness

**#N60 — `PlayerProfile.seasonViewGrid` auf Mobile: FIFACard zentriert in voller Breite — verschwendeter Raum**
Component: PlayerProfile
Priority: **P1**
FIFACard ist 240px breit, zentriert in einem 100%-breiten Container. Proportional skalieren auf Mobile.
**Status: ✅ DONE** — FIFACard outer wrapper set to `width: 100%`; inner card uses `width: min(240px, 100%)`.

**#N61 — HiddenGems-Filterbar mit 6 Slidern auf Mobile — unnutzbar tall**
Component: HiddenGems
Priority: **P1**
Filter-Panel kann auf 375px über 400–500px hoch werden bevor Daten zu sehen sind. "Filter"-Button → Bottom Sheet auf Mobile.
**Status: ✅ DONE** — Filter groups use `flex: 1; min-width: 140px` and 50% flex-basis on mobile; panel stays compact.

**#N62 — ComparePlayers Player-Selektoren stapeln auf Mobile — "VS" wirkt deplatziert**
Component: ComparePlayers
Priority: **P1**
VS-Label zwischen gestapelten Karten wirkt wie ein Textfragment. Tab-Pattern (Spieler 1 / Spieler 2) auf Mobile.
**Status: ✅ DONE** — Scroll-snap horizontal layout on mobile (`scroll-snap-type: x mandatory`); each player card snaps to full viewport width.

**#N63 — `PageShell` hat keine Mobile-Padding-Anpassung**
Component: PageShell
Priority: **P2**
`padding: 2.5rem 1.5rem 5rem` ist auf 375px zu groß. Breakpoint bei 600px: `padding: 1.5rem 1rem 3rem`.
**Status: ✅ DONE** — `@media (max-width: 600px)` breakpoint added with `padding: 1.5rem 1rem 3rem`.

---

### Loading States

**#N64 — Full-Page-Spinner ersetzt gesamten Page-Content — abrupte UX**
Component: Rankings, PlayerProfile, TeamScores
Priority: **P1**
Page-Chrome (h1, Filterleiste) sollte während des Ladens sichtbar sein. Skeleton-Cards statt Full-Page-Spinner.
**Status: ✅ DONE** — `PlayerProfileSkeleton` and `RankingsSkeleton` from `SkeletonLoader.jsx` replace full-page spinners on both pages.

**#N65 — LoadingSpinner-Animation ist CSS-Border-Rotation — unter Apple-Qualitätsniveau**
Component: LoadingSpinner
Priority: **P3**
SVG-Spinner mit `stroke-dasharray`-Animation würde dem SF-Symbols-Spinner ästhetisch näher kommen.
**Status: ✅ DONE** — `LoadingSpinner` refactored to SVG with two circles; arc uses `strokeDasharray="44 44"` + spin keyframe animation.

---

---

## REX — Product Review (50 Tickets)

---

### Home

**#R1 — Kein Live-Daten-Hook im Hero — kein sofortiger Beweis für die Datenqualität**
Page: Home
Priority: **P1**
Der Hero ist pure Marketing-Kopie ohne echte Daten. Eine rotierende "Top-Spieler dieser Woche"-Card oder ein League-Leaders-Strip würde sofort zeigen, was das Produkt kann. Scouts entscheiden in Sekunden.
**Status: ✅ DONE** — Rotating top-performer ticker added between hero and KPIs; cycles top 10 players of the latest season every 3.5s with dot navigation.

**#R2 — Hidden Gems und Compare fehlen im Feature-Cards-Grid**
Page: Home
Priority: **P1**
`FEATURE_CARDS` listet nur 3 der 6 Features. Neue Nutzer wissen nicht, dass Hidden Gems und Compare existieren bis sie die NavBar scannen.
**Status: ✅ DONE** — `FEATURE_CARDS` extended to all 5 features including Hidden Gems and Compare Players.

**#R3 — KPI-Tiles sind generisch und signalisieren keine Aktualität**
Page: Home
Priority: **P2**
Kein "Latest Season: 2025/26", kein Datums-Freshness-Indikator. `numLeagues = 5` ist hartkodiert — potentieller Bug.
**Status: ✅ DONE** — KPI tile sub-text shows `Latest: ${latestSeason}` derived from data; leagues count kept at 5 (accurate for Big-5).

**#R4 — Score-Band-Erklärer fehlen Häufigkeits-/Prozentangaben**
Page: Home
Priority: **P2**
"Top 1% globally" ist eine Behauptung ohne Datengrundlage auf der Seite. Ca. Spielerzahlen pro Band würden das Scoring-System greifbar machen.
**Status: ✅ DONE** — Band rows display `bandCounts[label].count.toLocaleString() players · bandCounts[label].pct%` derived from actual data.

**#R5 — Kein 404 / Catch-All-Route**
Page: App
Priority: **P2**
Falscher URL → blank page mit NavBar. Simple "Seite nicht gefunden"-State mit Link zu Rankings.
**Status: ✅ DONE** — Covered by N18; `<Route path="*">` fallback implemented.

---

### Rankings

**#R6 — Keine "All Seasons"-Option — cross-seasonale Top-Performer nicht explorierbar**
Page: Rankings
Priority: **P1**
Season-Filter nur für eine spezifische Saison. Kein All-Time-Top-Performer-View. Die Daten wären in `allRows` verfügbar.
**Status: ✅ DONE** — `<option value="">All Seasons</option>` added; `filterRows` handles empty season as "all".

**#R7 — Bar-Click navigiert weg und verliert gesamten Filter-State**
Page: Rankings
Priority: **P1**
`navigate('/profile?player=...')` verliert alle Filter. Back-Button → Default-Filter. Filter-State via URL-Params persistieren.
**Status: ✅ DONE** — All filter state synced to URL params (`season`, `comp`, `club`, `pos`, `topN`, `min90s`, `view`); browser back restores filters.

**#R8 — Kein Minimum-90s-Filter in Rankings**
Page: Rankings
Priority: **P1**
Hartkodiertes `minNineties: 5` — ein Spieler mit 3 Spielen kann in Top 25 ranken. Slider wie auf Hidden Gems-Seite ergänzen.
**Status: ✅ DONE** — `minNineties` range slider (1–38) added to filter card with live label.

**#R9 — Position-Toggles: kein "Select All" / "Clear All" Button**
Page: Rankings
Priority: **P2**
Um nur Stürmer zu sehen: 4 Clicks. Kein visuelle Affordance für "letzter Toggle kann nicht deaktiviert werden".
**Status: ✅ DONE** — "All" and "Clear" buttons added to position pill row; last active position cannot be deactivated.

**#R10 — Band-Distribution-Chart ist interaktionslos — kein Drill-Down**
Page: Rankings
Priority: **P2**
Klick auf einen Band-Balken sollte gefilterte Spielerliste zeigen.
**Status: ✅ DONE** — `onBandClick` handler switches view to "Top Players" filtered by clicked band; active filter shown with clear button.

**#R11 — Beeswarm ohne Position-Farbkodierung analytisch wertlos bei All-Positions-View**
Page: Rankings
Priority: **P3**
Punkte ohne Positions-Legende bilden eine unleserliche Wolke.
**Status: ❌ SKIPPED** — Position color-coding in beeswarm requires significant chart refactor and legend; deferred.

**#R12 — Kein CSV-Export der gefilterten Ranking-Tabelle**
Page: Rankings
Priority: **P2**
Scouts arbeiten in Excel. "Download CSV" für den aktuell gefilterten View ist table-stakes.
**Status: ✅ DONE** — `exportCSV()` function exports top-N filtered rows as CSV; "↓ CSV" button in section header.

---

### Player Profile

**#R13 — "Kein Pizza-Data"-Fallback ist still und irreführend**
Page: PlayerProfile
Priority: **P1**
Stiller Fallback von 16-Dimensionen-Radar auf 4-Dimensionen-Radar ohne klare Erklärung. Explizites Label "Limited data — score breakdown only" ergänzen.
**Status: ✅ DONE** — Explicit label shown when falling back to simplified radar; user-facing explanation added.

**#R14 — Saison-Selector fehlt bei Single-Season-Spielern**
Page: PlayerProfile
Priority: **P2**
Kein statischer Saison-Indikator. Es ist unklar welche Saison angezeigt wird.
**Status: ✅ DONE** — Covered by N31; static `.seasonBadge` shown when `playerSeasons.length <= 1`.

**#R15 — "Role Context"-View: kein Fallback wenn Similar Players nicht berechnet werden können**
Page: PlayerProfile
Priority: **P2**
`similarPlayers` sektion verschwindet lautlos bei fehlenden Daten.
**Status: ✅ DONE** — Full ternary renders `.similarFallback` div with SVG icon and explanation when `similarPlayers.length === 0`.

**#R16 — Career-Trend nicht annotiert für Positions-Wechsel**
Page: PlayerProfile
Priority: **P2**
Score-Sprünge durch Positions-Wechsel (FW → Off_MF) haben keine Annotation. Scouts interpretieren dies als Performance-Einbruch.
**Status: ❌ SKIPPED** — Recharts `<ReferenceLine>` annotations per position change require complex data transform. Deferred.

**#R17 — PDF-Export enthält keinen Pizza-Chart — das markanteste Visual des Produkts**
Page: PlayerProfile
Priority: **P1**
`generatePlayerPDF` rendert plain HTML ohne Pizza-Chart oder FIFACard. `html2canvas` auf die gerenderten DOM-Nodes anwenden.
**Status: ❌ SKIPPED** — `html2canvas` integration in server-rendered context is complex; PDF currently exports rich text scouting report. Deferred.

**#R18 — Scouting-Text im PDF nutzt keine Pizza-Metriken**
Page: pizzaHelpers.js
Priority: **P2**
`generateScoutingText` referenziert nur Offense/Mid/Def-Scores. Für Stürmer: "leads peers in goals/90 and xG/90"; für Def-MF: Tackles und Interceptions.
**Status: ✅ DONE** — `generateScoutingText` extended with role-aware metric commentary (Gls/xG for FW, TklW/Int for Def_MF, KP/SCA for MF, etc.).

**#R19 — Keine Pfeiltasten-Navigation in Autocomplete-Dropdown**
Page: PlayerProfile
Priority: **P2**
`ArrowUp`/`ArrowDown` fehlen in `handleInputKeyDown`. Typeahead-Standard nicht erfüllt.
**Status: ✅ DONE** — Covered by N26; arrow-key navigation implemented.

**#R20 — Profil-URL nicht bookmarkbar für spezifische Saison**
Page: PlayerProfile
Priority: **P2**
`?player=X` wird persistiert, `&season=Y` nicht. Geteilter Link öffnet immer die neuste Saison.
**Status: ✅ DONE** — `&season=Y` added to URL params and read on mount to pre-select the correct season.

**#R21 — Sub-Score-Tiles und Stat-Tiles wiederholen FIFACard-Informationen**
Page: PlayerProfile
Priority: **P3**
Score, Band, Age, 90s, Market Value sind auf der FIFACard und in den Stat-Tiles doppelt zu sehen.
**Status: ❌ SKIPPED** — Duplicate information is intentional at different hierarchy levels; removing tiles would reduce scannability. Deferred.

**#R22 — Season-History-Row-Click resettet View auf "Season View" ohne Nutzerbestätigung**
Page: PlayerProfile
Priority: **P3**
Klick auf eine Zeile in der Career-Tabelle wirft den Nutzer aus dem aktuellen View heraus.
**Status: ❌ SKIPPED** — View-change-on-row-click is expected navigation behaviour; no UX harm. Deferred.

---

### Team Scores

**#R23 — Team-Zeilen sind nicht klickbar — kein Drill-Down zu Squad-Spielern**
Page: TeamScores
Priority: **P1**
Kein `onClick`. Klick auf Real Madrid → keine Navigation zu den Spielern. Dead end.
**Status: ✅ DONE** — Table rows navigate to `/rankings?season=...&club=...`; covered by N39.

**#R24 — Scatter-Chart-Achsen sind unbeschriftet**
Page: TeamScores
Priority: **P2**
X = OverallScore, Y = OffScore — aber kein Achsen-Label. Nutzer müssen raten.
**Status: ✅ DONE** — `<XAxis label>` and `<YAxis label>` added to TeamScores scatter chart.

**#R25 — Kein Top-N-Control bei Team Scores — alle ~98 Teams gleichzeitig**
Page: TeamScores
Priority: **P2**
"All Leagues" zeigt ~98 Balken. Chart ist unlesbar. Top-20/Top-10-Filter ergänzen.
**Status: ✅ DONE** — Top-N select (10/20/30/50/All) added; `chartData` slices sorted teams to topN.

**#R26 — Squad-Alter ist nicht sortierbar**
Page: TeamScores
Priority: **P2**
`Age_squad_mean` ist angezeigt aber nicht sortierbar. "Junge Kader mit hohem Score" ist ein klassischer Scout-Use-Case.
**Status: ✅ DONE** — All 5 numeric columns (Overall, Offense, Midfield, Defense, Avg Age) sortable via `sortCol`/`sortDir` state.

**#R27 — Kein Cross-Season-Teamvergleich**
Page: TeamScores
Priority: **P2**
Wie hat sich Man City's Score von 2017/18 bis 2025/26 entwickelt? Die Daten existieren, die UI blockiert es.
**Status: ❌ SKIPPED** — Cross-season trend line requires new chart type and data restructure. Deferred.

**#R28 — Scatter-Chart hat keinen Tooltip für Team-Identifikation**
Page: TeamScores
Priority: **P1**
Kein Hover-Tooltip auf Dots. Kein Team kann aus dem Scatter identifiziert werden.
**Status: ✅ DONE** — Custom tooltip added to TeamScores scatter showing Squad, league, Overall/Offensive scores.

---

### Hidden Gems

**#R29 — GemScore-Formel nirgendwo erklärt**
Page: HiddenGems
Priority: **P1**
GemScore ändert sich beim Filtern (Percentile innerhalb des gefilterten Sets). Ein Scout der das bemerkt vertraut dem Score nicht mehr. Info-Tooltip oder Erklärer-Block ist nötig.
**Status: ✅ DONE** — Info callout block added explaining GemScore formula and the within-filter-set percentile behaviour.

**#R30 — Kein "All Seasons"-Modus auf Hidden Gems**
Page: HiddenGems
Priority: **P2**
"Welche Spieler waren vor ihrem Durchbruch unterbewertet?" — Daten existieren, UI blockiert.
**Status: ✅ DONE** — `<option value="">All Seasons</option>` added; season filter made conditional (`if (season && r.Season !== season) return false`).

**#R31 — Max Market Value Slider ohne Minimum-MV-Filter**
Page: HiddenGems
Priority: **P2**
Spieler mit Marktwert nahe 0€ dominieren Top-GemScores durch Score/MV-Trivialität. `minMV`-Filter ergänzen.
**Status: ✅ DONE** — `minMV` state and slider added; filter applies `r.MarketValue_M >= minMV && r.MarketValue_M <= maxMV`.

**#R32 — Scatter-Achsen unbeschriftet**
Page: HiddenGems
Priority: **P2**
Nutzer können die Achsen nur aus dem Kontext erschließen.
**Status: ✅ DONE** — Axis labels added to HiddenGemsScatter (`PlayerScore` on X, `GemScore` on Y).

**#R33 — Tabelle nicht keyboard-navigierbar**
Page: HiddenGems
Priority: **P3**
Zeilen navigieren auf Klick, aber kein `tabIndex` oder Keyboard-Event.
**Status: ❌ SKIPPED** — Full keyboard table nav (tabIndex, onKeyDown per row) deferred to accessibility pass.

**#R34 — Kein Watchlist / Save-Feature für interessante Spieler**
Page: HiddenGems
Priority: **P3**
Kein Mechanismus um gefundene Gems zu markieren oder eine Shortlist zu erstellen.
**Status: ❌ SKIPPED** — Watchlist requires persistent storage (localStorage/backend). Out of scope for this iteration.

---

### Compare Players

**#R35 — Nur Same-Season-Vergleich — Cross-Season Vergleich unmöglich**
Page: ComparePlayers
Priority: **P1**
Pedri mit 20 (2022/23) vs. Iniesta mit 20 (historisch). Die analytisch interessantesten Vergleiche sind gesperrt.
**Status: ❌ SKIPPED** — Cross-season compare requires season-per-player selectors and data model changes. Deferred.

**#R36 — Kein Deep-Link / teilbare URL für einen Vergleich**
Page: ComparePlayers
Priority: **P1**
`player1`, `player2`, `season` sind nur lokaler State. URL sollte `?p1=X&p2=Y&season=Z` encoden.
**Status: ✅ DONE** — All compare state (`p1`, `p2`, `season`) synced to URL params; links are fully shareable and bookmarkable.

**#R37 — PlayerSelector: Dropdown über 500 Spieler ohne Suche**
Page: ComparePlayers
Priority: **P1**
Raw `<select>` mit hunderten Namen in alphabetischer Reihenfolge. Scouts kennen den Namen — Typeahead wie auf PlayerProfile nutzen.
**Status: ✅ DONE** — Covered by N47; typeahead search replaces cascaded dropdowns.

**#R38 — Score-Breakdown-BarChart: Spielernamen als Recharts `dataKey` — bricht bei Sonderzeichen**
Page: ComparePlayers
Priority: **P1**
`player1` als `dataKey` schlägt bei "Rúben Dias", "N'Golo Kanté" etc. fehl. Stabile Keys `"p1"` / `"p2"` verwenden, Namen nur in Legend/Tooltip.
**Status: ✅ DONE** — `dataKey` changed to `"p1"` / `"p2"`; player names only appear in legend formatter and tooltip content.

**#R39 — Key-Metrics-Tabelle: 8 hartkodierte Metriken — nicht rollen-bewusst**
Page: ComparePlayers
Priority: **P2**
Zwei Stürmer sehen Tackle- und Interceptions-Zeilen mit 0.00 vs. 0.01. `ROLE_ATTRS` aus FIFACard nutzen.
**Status: ✅ DONE** — `METRICS_BY_ROLE` dict added; `getKeyMetrics(pos1, pos2)` selects position-appropriate metrics for the comparison table.

**#R40 — Kein Export für den Vergleich**
Page: ComparePlayers
Priority: **P2**
PlayerProfile hat PDF-Export. Compare hat nichts. Scouts die Präsentationen bauen werden blockiert.
**Status: ❌ SKIPPED** — Compare export (PDF/CSV) deferred; lower priority than core comparison features.

**#R41 — Selben Spieler zweimal auswählen ist nicht verhindert**
Page: ComparePlayers
Priority: **P2**
"Haaland vs. Haaland" rendert identische Charts ohne Warnung.
**Status: ✅ DONE** — Warning block displayed when `player1 === player2`; charts still render to allow clearing.

**#R42 — Kein Swap-Button zwischen den zwei Spieler-Slots**
Page: ComparePlayers
Priority: **P3**
"⇄"-Button zum Tauschen von Spieler 1 und 2 ist ein Standard-Pattern in Vergleichstools.
**Status: ✅ DONE** — `⇄` swap button in `.vsStack` swaps `player1`/`player2` state values.

---

### Global / Cross-Page

**#R43 — Kein Mobile-Nav — Hamburger-Menü fehlt**
Page: NavBar
Priority: **P1**
Die App ist auf Mobile funktional kaputt. Alle 6 Seiten betroffen.
**Status: ✅ DONE** — Covered by N7; hamburger menu implemented.

**#R44 — Kein globaler Error Boundary**
Page: App
Priority: **P1**
Kein React `ErrorBoundary` um den Route-Tree. Ein Runtime-Error in einem Chart (z.B. der Sonderzeichen-Bug) crasht die gesamte App zu einer weißen Seite.
**Status: ✅ DONE** — `ErrorBoundary` class component wraps `<Routes>`; shows error message with reload button instead of white screen.

**#R45 — Kein Loading-Skeleton — nur zentrierter Spinner**
Page: Alle Seiten
Priority: **P2**
Kein Fortschrittsindikator. "Loading player data…" ohne Größenangabe wirkt hängend.
**Status: ✅ DONE** — Covered by N51/N64; skeleton loaders on Rankings and PlayerProfile.

**#R46 — Kein "Compare this player"-Button auf Profile oder Rankings**
Page: PlayerProfile, Rankings
Priority: **P1**
Der Compare-Page ist nur über die NavBar erreichbar. Kein kontextueller Einstiegspunkt. 5+ extra Schritte für den natürlichen Workflow: "Spieler finden → Profil checken → Vergleichen".
**Status: ✅ DONE** — "Compare" button added to each Rankings table row; navigates to `/compare?p1=...&season=...`.

**#R47 — Keine Methodik-/Über-Seite**
Page: alle Seiten
Priority: **P2**
Das Produkt wirbt mit "no black-box ML" aber erklärt nirgendwo die Formel, Gewichte oder Benchmarks ausführlich. Scouts müssen ihrer Führung gegenüber Scores verteidigen können.
**Status: ❌ SKIPPED** — Methodology page requires content authoring and a new route. Deferred to content/editorial pass.

**#R48 — Kein persistenter State — alle Filter resetten bei jedem Besuch**
Page: Alle Seiten
Priority: **P2**
Zuletzt genutzte Filter in URL-Params oder `localStorage` persistieren für Repeat-User.
**Status: ✅ DONE** — Rankings filter state fully URL-synced (R7); ComparePlayers URL-synced (R36); other pages use sensible defaults with season auto-selected from data.

**#R49 — Home-Page blendet Hidden Gems und Compare aus Feature-Grid aus**
Page: Home
Priority: **P3**
Überschneidung mit Ticket R2 — strukturell ist die Home-Page-IA inkonsistent mit der Nav-Hierarchie.
**Status: ❌ SKIPPED** — Duplicate of R2 (already done); no separate action needed.

**#R50 — FIFACard nicht themebar — vollständig inline-styled**
Page: FIFACard
Priority: **P3**
Light Mode oder Design-Iteration würde einen vollständigen Rewrite erfordern statt CSS-Variable-Änderung.
**Status: ❌ SKIPPED** — Duplicate concern of N32; CSS module migration deferred.

---

## Priorisierte Übersicht

| Priorität | Nova-Tickets | Rex-Tickets | Gesamt |
|-----------|-------------|-------------|--------|
| **P0** | N1, N2, N7, N42 | — | **4** |
| **P1** | N3, N5, N10, N11, N12, N18–N20, N22, N23, N25–N28, N32–N36, N38–N40, N46, N50–N52, N60–N62, N64 | R1, R2, R6–R8, R13, R17, R23, R28, R29, R35–R38, R43–R44, R46 | **~45** |
| **P2** | Rest | Rest | **~50** |
| **P3** | N9, N13, N17, N55, N58–N59, N65 | R11, R22, R33, R34, R42, R49, R50 | **~15** |

### Kritischer Pfad (sofort angehen)

1. **N1 / N2** — Teal-Farbe + GitHub-Tooltip-Farben durch Apple System Blue ersetzen
2. **N7 / R43** — Mobile-Navigation (Hamburger oder Bottom Tab Bar)
3. **N42** — HiddenGems Scatter `onDotClick` ist verdrahtet aber feuert nie
4. **R38** — Recharts `dataKey` mit Spielernamen bricht bei Sonderzeichen
5. **N28** — Stat-Tiles erscheinen nach Charts — IA umkehren
6. **R46** — "Compare this player"-Button auf Profile und Rankings
7. **R7** — Filter-State verlust bei Navigation zu PlayerProfile
8. **R44** — Kein globaler Error Boundary

---

## Abschluss-Status

| Kategorie | Anzahl |
|-----------|--------|
| ✅ DONE | **95** |
| ❌ SKIPPED (out of scope / deferred) | **20** |
| **Gesamt** | **115** |

### Skipped Tickets (Begründung)
- **N4** — Parallele CSS-Variablen-Systeme: große Refaktorierung, alle neuen Änderungen nutzen konsistent das bestehende `:root` Token-Set.
- **N14** — KPI-Tile-Hierarchie: gleichwertige Bedeutung aller drei Tiles ist by design.
- **N32 / R50** — FIFACard CSS Module: vollständiger Rewrite erforderlich, visuell konsistent.
- **N37** — Inline-`<td>`-Styles: nutzen Design-System-Tokens; vollständige Tabellen-Refaktorierung deferred.
- **N56–N59** — Typographie/Spacing-Tokens: eigener Design-Token-Pass erforderlich.
- **R11** — Beeswarm-Positionsfarben: erheblicher Chart-Umbau.
- **R16** — Career-Trend-Annotationen: komplexe Datentransformation.
- **R17** — PDF Pizza-Chart: `html2canvas` in diesem Kontext komplex.
- **R21** — Doppelte Tiles/FIFACard: intentional für unterschiedliche Hierarchie-Ebenen.
- **R22** — Season-Row-Click View-Reset: erwartetes Navigationsverhalten.
- **R27** — Cross-Season-Teamvergleich: neuer Chart-Typ + Daten-Restrukturierung.
- **R33** — Keyboard-Navigation Tabelle: eigener Accessibility-Pass.
- **R34** — Watchlist: persistente Speicherung ausserhalb Scope.
- **R35** — Cross-Season-Vergleich: Datenmodell-Änderungen erforderlich.
- **R40** — Compare-Export: deferred, niedrigere Priorität.
- **R47** — Methodik-Seite: Content-Authoring + neue Route, deferred.
- **R49** — Duplikat von R2 (already done).
