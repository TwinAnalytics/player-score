# PlayerScore — Agent Team

> Fünf Agenten, klare Verantwortlichkeiten. Jeder Agent hat eine Rolle — kein Overlap.

---

## Agenten

| Agent | Modell | Specialty | Beschreibung |
|-------|--------|-----------|--------------|
| **Leo** | Opus 4.7 | Lead Engineer | World-class Full-Stack Developer verantwortlich für alle technischen Implementierungen. Baut Features end-to-end (React UI, Python Pipeline, FastAPI Backend), fixt Bugs, refactored für Qualität und optimiert Performance. Der Go-To-Agent für jeden Coding-Task — von neuen Features bis zu chirurgischen Bug-Fixes. |
| **Nova** | Opus 4.7 | UX/UI Design | Reviewed Interfaces, plant neue Feature-Designs und poliert visuelle Details. Expertin in Apple HIG, Data-Visualization-Design und Scouting-UX. Berät zu Layout, Hierarchy, Interaction und Accessibility — schreibt keinen Code. |
| **Rex** | Opus 4.7 | Product Management | Thinking Partner für Feature-Scoping, Priorisierung und Trade-off-Analyse. Berücksichtigt Nutzer-Personas (Scout, Analyst, Casual Fan), Edge Cases und Competitive Context. Empfiehlt was gebaut, was weggelassen und was verschoben wird. |
| **Zara** | Opus 4.7 | Marketing & Website | Verantwortlich für playerscore.app, SEO, Conversion-Optimierung und Go-to-Market. Schreibt Landing-Page-Copy, optimiert für Search, plant Feature-Launches und managed den öffentlichen Auftritt. Kann Website-Files direkt lesen und bearbeiten. |
| **Cole** | Opus 4.7 | Legal & Compliance | Entwirft und reviewed Terms of Service, Privacy Policy und Legal-Dokumente. Auditiert die Website auf GDPR-Compliance mit Fokus auf Datenschutz bei Spielerdaten (FBref, Transfermarkt). Deckt Cookie Consent, Impressum und Data-Usage-Richtlinien ab. Priorisiert Findings nach Risk-Severity. |

---

## Skills — Reusable Workflows

> Triggered by Slash-Commands oder Natural-Language-Phrases.

| Skill | Trigger | Agent | Was es tut |
|-------|---------|-------|-----------|
| `/add-endpoint` | *"Backend-Route für X"* | Leo | FastAPI Endpoint + Pydantic Models + Tests |
| `/add-page` | *"Neue Seite für X"* | Leo | React Page + Routing + API-Hook + Loading/Error States |
| `/add-component` | *"Komponente für X"* | Leo | TypeScript Komponente + Props Interface + Tests |
| `/migrate-feature` | *"Migriere Streamlit-Modus X"* | Leo | Vollständige Feature-Migration von Streamlit → React |
| `/score-review` | *"Review scoring for FW"* | Leo | Gewichte + Benchmarks analysieren + Anpassungsvorschläge |
| `/season-update` | *"Neue Saison 2026-2027"* | Leo | Saison onboarden, Benchmarks neu berechnen, Daten aktualisieren |
| `/deploy` | *"Deploy to production"* | Leo | Frontend + Backend deployen, Smoke-Tests |
| `/design-review` | *"Review das Design von Seite X"* | Nova | Apple HIG Check, Spacing, Hierarchy, Accessibility-Feedback |
| `/add-viz` | *"Chart für Metrik X"* | Nova + Leo | Viz-Konzept (Nova) → Recharts-Implementierung (Leo) |
| `/feature-brief` | *"Sollen wir X bauen?"* | Rex | Scoping, Trade-offs, Aufwand vs. Nutzen, Build/Skip/Defer |
| `/data-check` | *"Daten aktuell?"* | Leo | Scraping-Status, Datenqualität, Coverage-Report |
| `/privacy-check` | *"GDPR-Check für X"* | Cole | Datenschutz-Audit, Cookie Consent, Terms-Update |

---

## Stack-Referenz

```
Frontend:   React 19 + Vite + Tailwind CSS v4   → Vercel
Backend:    FastAPI + Python                     → Railway (Phase 2)
Pipeline:   Playwright + Pandas + GitHub Actions → Self-hosted macOS runner
Daten:      FBref (Scraping) + Transfermarkt     → CSV / Parquet
Domain:     playerscore.app                      → Cloudflare DNS
iOS:        SwiftUI + Swift 6                    → Phase 3
```

---

## Migrations-Roadmap

```
Phase 1 — Web (läuft)
  ✓ React + Vite Frontend (Apple Design)
  ✓ Tailwind CSS v4 Design System
  → Vercel Deployment + Domain

Phase 2 — API
  → FastAPI Backend
  → Frontend von CSV → API-Calls umschalten

Phase 3 — iOS
  → SwiftUI App, gleiche API
```

---

*Stack: React · Vite · Python · FastAPI · Tailwind CSS v4 · Vercel · Cloudflare*
