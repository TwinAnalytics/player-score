"""
generate_presentation.py
------------------------
Generates a self-contained HTML presentation (reveal.js) describing the
PlayerScore project. Output: docs/index.html  →  ready for GitHub Pages.

Usage:
    python generate_presentation.py
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Slide content
# ---------------------------------------------------------------------------

SLIDES = [
    # 0 ─ Title
    {
        "type": "title",
        "title": "PlayerScore",
        "subtitle": "Role-Aware Football Analytics for Europe's Big-5 Leagues",
        "tags": ["Python", "Streamlit", "Playwright", "FBref", "2017–2026"],
    },

    # 1 ─ The Problem
    {
        "type": "problem",
        "title": "The Problem with Player Evaluation",
        "points": [
            "Raw stats (goals, assists) ignore playing position and role",
            "Comparing a striker to a defensive midfielder is meaningless",
            "Most metrics are league-relative — not cross-league transferable",
            "Black-box ML models offer predictions without explanations",
        ],
        "quote": "How do you compare Rodri and Mbappé on the same scale — fairly?",
    },

    # 2 ─ Solution
    {
        "type": "solution",
        "title": "PlayerScore — Transparent & Role-Aware",
        "description": "A benchmark-driven scoring system that evaluates every player against absolute, role-specific standards — not relative to their peers.",
        "pillars": [
            {"icon": "🎯", "label": "Role-Specific", "text": "FW · Off_MF · MF · Def_MF · DF"},
            {"icon": "📐", "label": "Benchmark-Driven", "text": "Fixed absolute thresholds per metric"},
            {"icon": "🔍", "label": "Transparent", "text": "Every weight & benchmark is visible"},
            {"icon": "🌍", "label": "Cross-League", "text": "Consistent across all Big-5 leagues"},
        ],
    },

    # 3 ─ Data coverage
    {
        "type": "stats",
        "title": "Dataset at a Glance",
        "stats": [
            {"value": "9", "label": "Seasons", "sub": "2017/18 – 2025/26"},
            {"value": "5", "label": "Leagues", "sub": "PL · La Liga · Bundesliga · Serie A · Ligue 1"},
            {"value": "6,782", "label": "Unique Players", "sub": "across all seasons"},
            {"value": "23,095", "label": "Player-Season Records", "sub": "fully scored & classified"},
        ],
    },

    # 4 ─ Architecture
    {
        "type": "architecture",
        "title": "System Architecture",
        "steps": [
            {"step": "1", "label": "Scrape", "detail": "Playwright → FBref (11 stat tables)", "icon": "🕷️"},
            {"step": "2", "label": "Process", "detail": "Per-90 normalisation · Position classification", "icon": "⚙️"},
            {"step": "3", "label": "Score", "detail": "Role-specific benchmark scoring (0–1000)", "icon": "📊"},
            {"step": "4", "label": "Aggregate", "detail": "Multi-season career stats · Squad-level scores", "icon": "🗂️"},
            {"step": "5", "label": "Visualise", "detail": "Streamlit dashboard — profiles, lists, scatter plots", "icon": "🖥️"},
        ],
    },

    # 5 ─ Data Collection
    {
        "type": "feature",
        "title": "Automated Data Collection",
        "icon": "🕷️",
        "headline": "Playwright-based FBref Scraper",
        "bullets": [
            "Scrapes <strong>11 statistical categories</strong> per season: standard, shooting, passing, creation, defense, possession, misc …",
            "Merges all tables into a single flat CSV per season",
            "Handles missing columns gracefully (league-specific stat availability)",
            "Weekly CI/CD trigger via <strong>GitHub Actions</strong> — runs every Tuesday 06:00 UTC",
            "Runs on a self-hosted macOS runner; auto-commits updated data to the repo",
        ],
        "code": "DO_SCRAPE=true USE_AUTO_SEASONS=true python run_multi_season_pipeline.py",
    },

    # 6 ─ Position Classification
    {
        "type": "feature",
        "title": "Position Classification",
        "icon": "📍",
        "headline": "From FBref strings to football roles",
        "bullets": [
            "FBref reports multi-role strings like <code>\"MF,DF\"</code> or <code>\"FW,MF\"</code>",
            "<strong>main_pos_from_string()</strong> extracts the primary role",
            "<strong>refine_mf_with_zones()</strong> splits midfielders by touch-zone data:",
            "&nbsp;&nbsp;→ High Att-3rd touches → <strong>Off_MF</strong>",
            "&nbsp;&nbsp;→ High Def-3rd touches → <strong>Def_MF</strong>",
            "Result: 5 outfield roles — FW · Off_MF · MF · Def_MF · DF",
        ],
    },

    # 7 ─ Scoring Formula
    {
        "type": "scoring",
        "title": "The Scoring Formula",
        "formula": "Score = 1000 × Σ(wᵢ × min(xᵢ / bᵢ, 1)) / Σwᵢ",
        "legend": [
            ("wᵢ", "Weight of feature i (role-specific importance)"),
            ("xᵢ", "Player's per-90 value for feature i"),
            ("bᵢ", "Benchmark — the value representing 'full performance'"),
        ],
        "properties": [
            "Absolute: score does not change when new players are added",
            "Capped at 1 per feature — you can't compensate a weak area by excelling elsewhere",
            "Benchmarks are fixed real-world thresholds (e.g. 0.9 goals/90 for FW)",
        ],
    },

    # 8 ─ Role weights
    {
        "type": "weights",
        "title": "Role-Specific Weights & Benchmarks",
        "roles": [
            {
                "name": "FW (Striker / Winger)",
                "color": "#ef4444",
                "top_metrics": [
                    ("Goals/90", "wt 0.40", "bench 0.9"),
                    ("xG/90", "wt 0.20", "bench 0.8"),
                    ("Assists/90", "wt 0.15", "bench 0.6"),
                    ("Key Passes/90", "wt 0.10", "bench 2.5"),
                ],
            },
            {
                "name": "MF (Central Mid)",
                "color": "#3b82f6",
                "top_metrics": [
                    ("Assists/90", "wt 0.22", "bench 0.3"),
                    ("Key Passes/90", "wt 0.20", "bench 2.0"),
                    ("xAG/90", "wt 0.18", "bench 0.6"),
                    ("Prog Passes/90", "wt 0.14", "bench 10.0"),
                ],
            },
            {
                "name": "DF (Defender)",
                "color": "#10b981",
                "top_metrics": [
                    ("Tackles Won/90", "wt 0.30", "bench 6.0"),
                    ("Interceptions/90", "wt 0.25", "bench 3.8"),
                    ("Blocks/90", "wt 0.18", "bench 3.0"),
                    ("Clearances/90", "wt 0.12", "bench 9.0"),
                ],
            },
        ],
    },

    # 9 ─ Score Bands
    {
        "type": "bands",
        "title": "Performance Tiers",
        "bands": [
            {"label": "Exceptional",         "range": "≥ 900", "color": "#f59e0b", "example": "Dembélé · Salah · Vinicius Jr"},
            {"label": "World Class",         "range": "750 – 899", "color": "#a78bfa", "example": "Top-10 players per position"},
            {"label": "Top Starter",         "range": "400 – 749", "color": "#3b82f6", "example": "Regular starters in Big-5"},
            {"label": "Solid Squad Player",  "range": "200 – 399", "color": "#6b7280", "example": "Rotation & backup players"},
            {"label": "Below Big-5 Level",   "range": "< 200",     "color": "#374151", "example": "Limited or below-par output"},
        ],
        "note": "2024/25 sample: Dembélé 908 (Exceptional) · Marmoush 855 (World Class)",
    },

    # 10 ─ App — Player Profiles
    {
        "type": "app_feature",
        "title": "App: Player Profiles",
        "icon": "👤",
        "features": [
            {"label": "Pizza Chart", "desc": "Percentile radar vs Big-5 role peers — attack · possession · defense slices"},
            {"label": "Career Trend", "desc": "Score evolution across all seasons with band shading"},
            {"label": "Scatter Plots", "desc": "xG vs Goals, xAG vs Assists — overperformance vs underperformance"},
            {"label": "Summary Tiles", "desc": "Age · Club · Position · Minutes · Score · Band at a glance"},
        ],
    },

    # 11 ─ App — Top Lists
    {
        "type": "app_feature",
        "title": "App: Top Lists & Rankings",
        "icon": "📊",
        "features": [
            {"label": "Multi-Filter", "desc": "Season · League · Club · Position · Age · Minutes threshold"},
            {"label": "Top-N Bar Charts", "desc": "Best players by primary role score — instantly filterable"},
            {"label": "Score vs Age", "desc": "Beeswarm plot revealing peak age windows per role"},
            {"label": "Band Distribution", "desc": "How the filtered population distributes across tiers"},
        ],
    },

    # 12 ─ App — Team Scores
    {
        "type": "app_feature",
        "title": "App: Team / Squad Intelligence",
        "icon": "🟦",
        "features": [
            {"label": "Squad Strength", "desc": "Minute-weighted offense · midfield · defense scores per club"},
            {"label": "League Rankings", "desc": "Side-by-side squad comparison within a single season"},
            {"label": "Top Contributors", "desc": "Which players drive each squad's score up"},
            {"label": "Multi-Season Trends", "desc": "Track squad development and squad rebuilds over time"},
        ],
    },

    # 13 ─ Tech Stack
    {
        "type": "tech",
        "title": "Technology Stack",
        "groups": [
            {
                "label": "Data & Scraping",
                "items": ["Python 3.11", "Playwright (async)", "pandas", "FBref (open data)"],
                "color": "#f59e0b",
            },
            {
                "label": "Analytics",
                "items": ["Custom benchmark scoring", "Per-90 normalisation", "Touch-zone classification", "Career aggregation"],
                "color": "#3b82f6",
            },
            {
                "label": "Visualisation",
                "items": ["Streamlit", "Altair", "Matplotlib / mplsoccer", "Pizza charts (PyPizza)"],
                "color": "#10b981",
            },
            {
                "label": "DevOps / CI",
                "items": ["GitHub Actions (weekly)", "Self-hosted macOS runner", "Auto-commit pipeline", "GitHub Pages (docs)"],
                "color": "#a78bfa",
            },
        ],
    },

    # 14 ─ CI/CD
    {
        "type": "cicd",
        "title": "Weekly CI/CD Pipeline",
        "steps": [
            "GitHub Actions triggers every Tuesday 06:00 UTC",
            "Self-hosted macOS runner launches run_multi_season_pipeline.py",
            "Playwright scrapes all 5 leagues for current season",
            "Processing & scoring runs automatically",
            "Updated CSVs committed back to repo with timestamp message",
            "Streamlit app immediately reflects fresh data",
        ],
        "badge": "Weekly run (2025-2026) — fully automated",
    },

    # 15 ─ Sample insights
    {
        "type": "insights",
        "title": "Sample Insights — 2024/25 Season",
        "insights": [
            {
                "category": "🔴 Exceptional Forwards",
                "items": [
                    "Ousmane Dembélé (PSG) — 908",
                    "Omar Marmoush (Frankfurt) — 855",
                    "Mohamed Salah (Liverpool) — 821",
                ],
            },
            {
                "category": "🟢 Top Defenders",
                "items": [
                    "Olivier Deman (Werder) — 827",
                    "Thomas Delaine (Strasbourg) — 825",
                    "Loïc N'Gatta (Auxerre) — 739",
                ],
            },
            {
                "category": "📈 Dataset Scale",
                "items": [
                    "2,594 players scored in 2024/25",
                    "96 squads across 5 leagues",
                    "Covering ~2,500 minutes minimum threshold",
                ],
            },
        ],
    },

    # 16 ─ Why PlayerScore
    {
        "type": "why",
        "title": "Why PlayerScore?",
        "tagline": "Transparent · Reproducible · Role-Aware",
        "reasons": [
            {
                "title": "No Black Box",
                "desc": "Every weight, benchmark, and formula is open and inspectable. You know exactly why a player scores 650 and not 800.",
            },
            {
                "title": "Cross-League Fairness",
                "desc": "Absolute benchmarks mean a Bundesliga defender and a Premier League defender are scored on the same scale.",
            },
            {
                "title": "Scouting-Ready",
                "desc": "Position-aware scoring reflects actual roles — a striker's defensive contribution doesn't inflate their score.",
            },
            {
                "title": "Living Dataset",
                "desc": "Automated weekly updates mean scores are always current — no manual refresh needed.",
            },
        ],
    },

    # 17 ─ Future Roadmap
    {
        "type": "roadmap",
        "title": "Roadmap & Future Work",
        "items": [
            {"icon": "🥅", "label": "Goalkeeper Scoring", "desc": "Dedicated GK model with PSxG+/- and save rate benchmarks"},
            {"icon": "📈", "label": "Value Estimation", "desc": "Market value prediction layer on top of benchmark scores"},
            {"icon": "🔁", "label": "Player Comparison", "desc": "Side-by-side profile comparisons with radar overlays"},
            {"icon": "🤖", "label": "NLP Scouting Reports", "desc": "Auto-generated plain-language player summaries via LLM"},
            {"icon": "🌐", "label": "Extended Coverage", "desc": "Eredivisie, Liga Portugal, and more secondary leagues"},
        ],
    },

    # 18 ─ Closing / Links
    {
        "type": "closing",
        "title": "Explore PlayerScore",
        "subtitle": "Open-source · Data-driven · Football-obsessed",
        "links": [
            {"label": "GitHub Repository", "icon": "💻", "url": "https://github.com/TwinAnalytics/player-score"},
            {"label": "Live Presentation", "icon": "📊", "url": "https://twinanalytics.github.io/player-score/"},
        ],
        "stack_line": "Python · Playwright · pandas · Streamlit · GitHub Actions",
    },
]


# ---------------------------------------------------------------------------
# HTML template builder
# ---------------------------------------------------------------------------

def build_slide_html(slide: dict) -> str:
    t = slide["type"]

    if t == "title":
        tags = " ".join(f'<span class="tag">{tag}</span>' for tag in slide["tags"])
        return f"""
<section class="slide-title">
  <div class="title-badge">⚽ Football Analytics Project</div>
  <h1>{slide["title"]}</h1>
  <p class="subtitle">{slide["subtitle"]}</p>
  <div class="tag-row">{tags}</div>
</section>"""

    if t == "problem":
        points = "".join(f'<li>{p}</li>' for p in slide["points"])
        return f"""
<section class="slide-content">
  <h2>{slide["title"]}</h2>
  <ul class="problem-list">{points}</ul>
  <blockquote class="quote">"{slide["quote"]}"</blockquote>
</section>"""

    if t == "solution":
        pillars = "".join(
            f'<div class="pillar"><div class="pillar-icon">{p["icon"]}</div>'
            f'<div class="pillar-label">{p["label"]}</div>'
            f'<div class="pillar-text">{p["text"]}</div></div>'
            for p in slide["pillars"]
        )
        return f"""
<section class="slide-content">
  <h2>{slide["title"]}</h2>
  <p class="lead">{slide["description"]}</p>
  <div class="pillars">{pillars}</div>
</section>"""

    if t == "stats":
        boxes = "".join(
            f'<div class="stat-box"><div class="stat-value">{s["value"]}</div>'
            f'<div class="stat-label">{s["label"]}</div>'
            f'<div class="stat-sub">{s["sub"]}</div></div>'
            for s in slide["stats"]
        )
        return f"""
<section class="slide-content">
  <h2>{slide["title"]}</h2>
  <div class="stat-grid">{boxes}</div>
</section>"""

    if t == "architecture":
        steps = "".join(
            f'<div class="arch-step">'
            f'<div class="arch-icon">{s["icon"]}</div>'
            f'<div class="arch-num">Step {s["step"]}</div>'
            f'<div class="arch-label">{s["label"]}</div>'
            f'<div class="arch-detail">{s["detail"]}</div>'
            f'</div>'
            for s in slide["steps"]
        )
        return f"""
<section class="slide-content">
  <h2>{slide["title"]}</h2>
  <div class="arch-flow">{steps}</div>
</section>"""

    if t == "feature":
        bullets = "".join(f'<li>{b}</li>' for b in slide["bullets"])
        code_block = f'<pre class="code-line"><code>{slide["code"]}</code></pre>' if slide.get("code") else ""
        return f"""
<section class="slide-content">
  <h2>{slide["title"]}</h2>
  <div class="feature-header">{slide["icon"]} {slide["headline"]}</div>
  <ul class="feature-list">{bullets}</ul>
  {code_block}
</section>"""

    if t == "scoring":
        legend = "".join(
            f'<tr><td class="var">{v}</td><td>{d}</td></tr>'
            for v, d in slide["legend"]
        )
        props = "".join(f'<li>{p}</li>' for p in slide["properties"])
        return f"""
<section class="slide-content">
  <h2>{slide["title"]}</h2>
  <div class="formula-box">{slide["formula"]}</div>
  <table class="legend-table">{legend}</table>
  <ul class="props-list">{props}</ul>
</section>"""

    if t == "weights":
        role_cards = ""
        for role in slide["roles"]:
            metrics = "".join(
                f'<tr><td>{m[0]}</td><td class="wt">{m[1]}</td><td class="bench">{m[2]}</td></tr>'
                for m in role["top_metrics"]
            )
            role_cards += (
                f'<div class="role-card" style="border-top:3px solid {role["color"]}">'
                f'<div class="role-name" style="color:{role["color"]}">{role["name"]}</div>'
                f'<table class="metric-table">'
                f'<thead><tr><th>Metric</th><th>Weight</th><th>Benchmark</th></tr></thead>'
                f'<tbody>{metrics}</tbody>'
                f'</table></div>'
            )
        return f"""
<section class="slide-content">
  <h2>{slide["title"]}</h2>
  <div class="role-grid">{role_cards}</div>
</section>"""

    if t == "bands":
        band_rows = "".join(
            f'<div class="band-row">'
            f'<div class="band-dot" style="background:{b["color"]}"></div>'
            f'<div class="band-label" style="color:{b["color"]}">{b["label"]}</div>'
            f'<div class="band-range">{b["range"]}</div>'
            f'<div class="band-example">{b["example"]}</div>'
            f'</div>'
            for b in slide["bands"]
        )
        return f"""
<section class="slide-content">
  <h2>{slide["title"]}</h2>
  <div class="bands-container">{band_rows}</div>
  <p class="band-note">💡 {slide["note"]}</p>
</section>"""

    if t == "app_feature":
        cards = "".join(
            f'<div class="app-card"><div class="app-card-label">{f["label"]}</div>'
            f'<div class="app-card-desc">{f["desc"]}</div></div>'
            for f in slide["features"]
        )
        return f"""
<section class="slide-content">
  <h2><span class="app-icon">{slide["icon"]}</span> {slide["title"]}</h2>
  <div class="app-grid">{cards}</div>
</section>"""

    if t == "tech":
        groups = "".join(
            f'<div class="tech-group" style="border-left:3px solid {g["color"]}">'
            f'<div class="tech-label" style="color:{g["color"]}">{g["label"]}</div>'
            f'<ul>{"".join(f"<li>{item}</li>" for item in g["items"])}</ul>'
            f'</div>'
            for g in slide["groups"]
        )
        return f"""
<section class="slide-content">
  <h2>{slide["title"]}</h2>
  <div class="tech-grid">{groups}</div>
</section>"""

    if t == "cicd":
        steps = "".join(
            f'<div class="cicd-step"><span class="cicd-num">{i+1}</span>{s}</div>'
            for i, s in enumerate(slide["steps"])
        )
        return f"""
<section class="slide-content">
  <h2>{slide["title"]}</h2>
  <div class="cicd-flow">{steps}</div>
  <div class="cicd-badge">✅ {slide["badge"]}</div>
</section>"""

    if t == "insights":
        cols = "".join(
            f'<div class="insight-col">'
            f'<div class="insight-cat">{ins["category"]}</div>'
            f'<ul>{"".join(f"<li>{item}</li>" for item in ins["items"])}</ul>'
            f'</div>'
            for ins in slide["insights"]
        )
        return f"""
<section class="slide-content">
  <h2>{slide["title"]}</h2>
  <div class="insight-grid">{cols}</div>
</section>"""

    if t == "why":
        cards = "".join(
            f'<div class="why-card"><div class="why-title">{r["title"]}</div>'
            f'<div class="why-desc">{r["desc"]}</div></div>'
            for r in slide["reasons"]
        )
        return f"""
<section class="slide-content">
  <h2>{slide["title"]}</h2>
  <p class="tagline">{slide["tagline"]}</p>
  <div class="why-grid">{cards}</div>
</section>"""

    if t == "roadmap":
        items = "".join(
            f'<div class="road-item"><span class="road-icon">{it["icon"]}</span>'
            f'<div><div class="road-label">{it["label"]}</div>'
            f'<div class="road-desc">{it["desc"]}</div></div></div>'
            for it in slide["items"]
        )
        return f"""
<section class="slide-content">
  <h2>{slide["title"]}</h2>
  <div class="road-list">{items}</div>
</section>"""

    if t == "closing":
        links = "".join(
            f'<a class="link-btn" href="{lk["url"]}" target="_blank">{lk["icon"]} {lk["label"]}</a>'
            for lk in slide["links"]
        )
        return f"""
<section class="slide-closing">
  <h2>{slide["title"]}</h2>
  <p class="closing-sub">{slide["subtitle"]}</p>
  <div class="link-row">{links}</div>
  <p class="stack-line">Built with: {slide["stack_line"]}</p>
</section>"""

    return f'<section><h2>{slide.get("title", "")}</h2></section>'


# ---------------------------------------------------------------------------
# Full HTML document
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

:root {
  --bg: #0a0f1e;
  --surface: #111827;
  --surface2: #1f2937;
  --accent: #00B8A9;
  --accent2: #3b82f6;
  --text: #e5e7eb;
  --muted: #9ca3af;
  --border: #374151;
  --fw-color: #ef4444;
  --mf-color: #3b82f6;
  --df-color: #10b981;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body {
  height: 100%;
  background: var(--bg);
  color: var(--text);
  font-family: 'Inter', sans-serif;
  overflow: hidden;
}

/* ── Deck wrapper ── */
#deck {
  position: relative;
  width: 100vw;
  height: 100vh;
  overflow: hidden;
}

/* ── Individual slides ── */
.slide {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 40px 60px;
  opacity: 0;
  transform: translateX(60px);
  transition: opacity .4s ease, transform .4s ease;
  pointer-events: none;
  background: var(--bg);
}
.slide.active {
  opacity: 1;
  transform: translateX(0);
  pointer-events: auto;
}
.slide.exit-left {
  opacity: 0;
  transform: translateX(-60px);
}

/* ── Slide inner containers ── */
section {
  width: 100%;
  max-width: 1000px;
}

/* ── Title slide ── */
.slide-title {
  text-align: center;
}
.title-badge {
  display: inline-block;
  background: rgba(0,184,169,.15);
  border: 1px solid var(--accent);
  color: var(--accent);
  padding: 4px 14px;
  border-radius: 20px;
  font-size: .8rem;
  letter-spacing: .08em;
  margin-bottom: 18px;
}
.slide-title h1 {
  font-size: 4rem;
  font-weight: 800;
  background: linear-gradient(135deg, #00B8A9 0%, #3b82f6 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1.1;
}
.subtitle {
  margin-top: 14px;
  font-size: 1.2rem;
  color: var(--muted);
  max-width: 640px;
  margin-left: auto;
  margin-right: auto;
}
.tag-row {
  margin-top: 24px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
}
.tag {
  background: var(--surface2);
  border: 1px solid var(--border);
  padding: 4px 12px;
  border-radius: 14px;
  font-size: .78rem;
  color: var(--muted);
}

/* ── Content slides ── */
.slide-content h2, .slide-closing h2 {
  font-size: 1.8rem;
  font-weight: 700;
  margin-bottom: 24px;
  color: var(--text);
  border-left: 4px solid var(--accent);
  padding-left: 14px;
}
.app-icon { margin-right: 6px; }

/* problem */
.problem-list { list-style: none; padding: 0; }
.problem-list li {
  padding: 10px 0 10px 24px;
  border-bottom: 1px solid var(--border);
  position: relative;
  font-size: 1.05rem;
  color: var(--text);
}
.problem-list li::before {
  content: '→';
  position: absolute;
  left: 0;
  color: var(--accent);
  font-weight: 700;
}
.quote {
  margin-top: 22px;
  padding: 14px 20px;
  background: var(--surface2);
  border-left: 3px solid var(--accent2);
  border-radius: 6px;
  font-style: italic;
  color: var(--muted);
  font-size: .95rem;
}

/* solution pillars */
.lead { color: var(--muted); margin-bottom: 22px; font-size: 1rem; }
.pillars { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; }
.pillar {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px 12px;
  text-align: center;
}
.pillar-icon { font-size: 1.8rem; margin-bottom: 8px; }
.pillar-label { font-weight: 700; font-size: .9rem; margin-bottom: 4px; color: var(--accent); }
.pillar-text { font-size: .78rem; color: var(--muted); }

/* stats */
.stat-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; }
.stat-box {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 24px 16px;
  text-align: center;
}
.stat-value {
  font-size: 2.6rem;
  font-weight: 800;
  color: var(--accent);
  line-height: 1;
}
.stat-label { font-size: 1rem; font-weight: 600; margin-top: 6px; }
.stat-sub { font-size: .75rem; color: var(--muted); margin-top: 4px; }

/* architecture */
.arch-flow { display: flex; gap: 8px; align-items: stretch; }
.arch-step {
  flex: 1;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px 12px;
  text-align: center;
  position: relative;
}
.arch-step:not(:last-child)::after {
  content: '→';
  position: absolute;
  right: -14px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--accent);
  font-weight: 700;
  font-size: 1.2rem;
  z-index: 1;
}
.arch-icon { font-size: 1.6rem; margin-bottom: 6px; }
.arch-num { font-size: .68rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }
.arch-label { font-weight: 700; font-size: .9rem; margin: 4px 0; color: var(--accent); }
.arch-detail { font-size: .72rem; color: var(--muted); }

/* feature */
.feature-header {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--accent);
  margin-bottom: 14px;
}
.feature-list { list-style: none; padding: 0; }
.feature-list li {
  padding: 8px 0 8px 22px;
  border-bottom: 1px solid var(--border);
  font-size: .95rem;
  position: relative;
}
.feature-list li::before {
  content: '•';
  position: absolute;
  left: 6px;
  color: var(--accent);
}
.code-line {
  margin-top: 16px;
  background: #000;
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 10px 16px;
  font-size: .8rem;
  color: #00ff9f;
  overflow-x: auto;
}

/* scoring */
.formula-box {
  background: var(--surface2);
  border: 1px solid var(--accent);
  border-radius: 10px;
  padding: 18px 24px;
  text-align: center;
  font-size: 1.3rem;
  font-weight: 700;
  color: var(--accent);
  font-family: 'Courier New', monospace;
  margin-bottom: 18px;
}
.legend-table { width: 100%; border-collapse: collapse; margin-bottom: 14px; }
.legend-table td { padding: 6px 10px; border-bottom: 1px solid var(--border); font-size: .88rem; }
.legend-table .var { color: var(--accent); font-weight: 700; font-family: monospace; width: 40px; }
.props-list { list-style: none; padding: 0; }
.props-list li { padding: 6px 0 6px 20px; position: relative; font-size: .88rem; color: var(--muted); }
.props-list li::before { content: '✓'; position: absolute; left: 0; color: #10b981; font-weight: 700; }

/* weights */
.role-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 14px; }
.role-card {
  background: var(--surface);
  border-radius: 10px;
  padding: 16px;
  border: 1px solid var(--border);
}
.role-name { font-weight: 700; font-size: .9rem; margin-bottom: 10px; }
.metric-table { width: 100%; border-collapse: collapse; font-size: .75rem; }
.metric-table th { color: var(--muted); font-weight: 600; padding: 4px 6px; border-bottom: 1px solid var(--border); text-align: left; }
.metric-table td { padding: 4px 6px; border-bottom: 1px solid rgba(55,65,81,.4); }
.wt { color: #f59e0b; }
.bench { color: var(--accent); }

/* bands */
.bands-container { display: flex; flex-direction: column; gap: 10px; margin-bottom: 14px; }
.band-row {
  display: grid;
  grid-template-columns: 16px 200px 120px 1fr;
  align-items: center;
  gap: 12px;
  background: var(--surface);
  border-radius: 8px;
  padding: 10px 16px;
}
.band-dot { width: 12px; height: 12px; border-radius: 50%; }
.band-label { font-weight: 700; font-size: .9rem; }
.band-range { color: var(--muted); font-size: .85rem; font-family: monospace; }
.band-example { font-size: .8rem; color: var(--muted); }
.band-note { font-size: .85rem; color: var(--muted); padding: 8px 0; }

/* app cards */
.app-grid { display: grid; grid-template-columns: repeat(2,1fr); gap: 14px; }
.app-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 18px;
}
.app-card-label { font-weight: 700; color: var(--accent); margin-bottom: 6px; font-size: .95rem; }
.app-card-desc { font-size: .85rem; color: var(--muted); }

/* tech */
.tech-grid { display: grid; grid-template-columns: repeat(2,1fr); gap: 14px; }
.tech-group {
  background: var(--surface);
  border-radius: 10px;
  padding: 16px 16px 16px 20px;
}
.tech-label { font-weight: 700; font-size: .9rem; margin-bottom: 10px; }
.tech-group ul { list-style: none; padding: 0; }
.tech-group li { padding: 4px 0; font-size: .85rem; color: var(--muted); }
.tech-group li::before { content: '→ '; color: var(--accent); }

/* cicd */
.cicd-flow { display: flex; flex-direction: column; gap: 10px; margin-bottom: 18px; }
.cicd-step {
  display: flex;
  align-items: center;
  gap: 14px;
  background: var(--surface);
  border-radius: 8px;
  padding: 10px 16px;
  font-size: .9rem;
}
.cicd-num {
  width: 26px; height: 26px;
  background: var(--accent);
  color: #000;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-weight: 700;
  font-size: .8rem;
  flex-shrink: 0;
}
.cicd-badge {
  background: rgba(16,185,129,.15);
  border: 1px solid #10b981;
  color: #10b981;
  padding: 8px 16px;
  border-radius: 8px;
  font-size: .85rem;
  text-align: center;
}

/* insights */
.insight-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; }
.insight-col {
  background: var(--surface);
  border-radius: 10px;
  padding: 16px;
}
.insight-cat { font-weight: 700; margin-bottom: 10px; font-size: .9rem; }
.insight-col ul { list-style: none; padding: 0; }
.insight-col li {
  padding: 6px 0;
  border-bottom: 1px solid var(--border);
  font-size: .82rem;
  color: var(--muted);
}

/* why */
.tagline { color: var(--accent); font-weight: 600; margin-bottom: 18px; font-size: 1rem; letter-spacing: .05em; }
.why-grid { display: grid; grid-template-columns: repeat(2,1fr); gap: 14px; }
.why-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px 18px;
}
.why-title { font-weight: 700; color: var(--accent); margin-bottom: 6px; font-size: .95rem; }
.why-desc { font-size: .83rem; color: var(--muted); }

/* roadmap */
.road-list { display: flex; flex-direction: column; gap: 12px; }
.road-item {
  display: flex;
  align-items: flex-start;
  gap: 16px;
  background: var(--surface);
  border-radius: 10px;
  padding: 12px 16px;
}
.road-icon { font-size: 1.5rem; flex-shrink: 0; }
.road-label { font-weight: 700; font-size: .9rem; margin-bottom: 2px; color: var(--accent); }
.road-desc { font-size: .82rem; color: var(--muted); }

/* closing */
.slide-closing { text-align: center; }
.closing-sub { color: var(--muted); margin-bottom: 30px; font-size: 1.05rem; }
.link-row { display: flex; gap: 16px; justify-content: center; margin-bottom: 28px; }
.link-btn {
  display: inline-block;
  padding: 12px 28px;
  background: var(--surface2);
  border: 1px solid var(--accent);
  color: var(--accent);
  border-radius: 8px;
  text-decoration: none;
  font-weight: 600;
  font-size: .95rem;
  transition: background .2s;
}
.link-btn:hover { background: rgba(0,184,169,.15); }
.stack-line { color: var(--muted); font-size: .8rem; }

/* ── Navigation ── */
#nav {
  position: fixed;
  bottom: 28px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: 16px;
  background: rgba(17,24,39,.85);
  backdrop-filter: blur(8px);
  border: 1px solid var(--border);
  border-radius: 40px;
  padding: 10px 24px;
  z-index: 100;
}
.nav-btn {
  background: none;
  border: none;
  color: var(--text);
  cursor: pointer;
  font-size: 1.1rem;
  padding: 4px 10px;
  border-radius: 6px;
  transition: background .15s, color .15s;
}
.nav-btn:hover { background: var(--surface2); color: var(--accent); }
#slide-counter { color: var(--muted); font-size: .85rem; min-width: 60px; text-align: center; }
#progress-bar {
  position: fixed;
  top: 0; left: 0;
  height: 3px;
  background: linear-gradient(90deg, #00B8A9, #3b82f6);
  transition: width .35s ease;
  z-index: 200;
}

/* ── Dot indicators ── */
#dots {
  position: fixed;
  right: 20px;
  top: 50%;
  transform: translateY(-50%);
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--border);
  cursor: pointer;
  transition: background .2s, transform .2s;
}
.dot.active { background: var(--accent); transform: scale(1.4); }
"""

JS = """
const slides = document.querySelectorAll('.slide');
const counter = document.getElementById('slide-counter');
const bar = document.getElementById('progress-bar');
const dots = document.querySelectorAll('.dot');
let cur = 0;

function goTo(idx) {
  slides[cur].classList.remove('active');
  slides[cur].classList.add('exit-left');
  setTimeout(() => slides[cur].classList.remove('exit-left'), 400);
  cur = Math.max(0, Math.min(idx, slides.length - 1));
  slides[cur].classList.add('active');
  counter.textContent = (cur + 1) + ' / ' + slides.length;
  bar.style.width = ((cur + 1) / slides.length * 100) + '%';
  dots.forEach((d, i) => d.classList.toggle('active', i === cur));
}

document.getElementById('btn-prev').onclick = () => goTo(cur - 1);
document.getElementById('btn-next').onclick = () => goTo(cur + 1);
dots.forEach((d, i) => d.onclick = () => goTo(i));

document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown' || e.key === ' ') goTo(cur + 1);
  if (e.key === 'ArrowLeft'  || e.key === 'ArrowUp')                    goTo(cur - 1);
});

// Touch/swipe
let startX = 0;
document.addEventListener('touchstart', e => startX = e.touches[0].clientX);
document.addEventListener('touchend', e => {
  const dx = e.changedTouches[0].clientX - startX;
  if (dx < -40) goTo(cur + 1);
  if (dx >  40) goTo(cur - 1);
});

goTo(0);
"""


def generate_html(slides: list[dict]) -> str:
    slide_divs = ""
    for i, slide in enumerate(slides):
        inner = build_slide_html(slide)
        slide_divs += f'<div class="slide" id="slide-{i}">{inner}</div>\n'

    dots_html = "".join(f'<div class="dot" title="Slide {i+1}"></div>' for i in range(len(slides)))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>PlayerScore — Football Analytics Project</title>
  <style>{CSS}</style>
</head>
<body>

<div id="progress-bar"></div>

<div id="deck">
{slide_divs}
</div>

<div id="dots">{dots_html}</div>

<nav id="nav">
  <button class="nav-btn" id="btn-prev">&#8592;</button>
  <span id="slide-counter">1 / {len(slides)}</span>
  <button class="nav-btn" id="btn-next">&#8594;</button>
</nav>

<script>{JS}</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out_dir = Path(__file__).parent / "docs"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "index.html"

    html = generate_html(SLIDES)
    out_file.write_text(html, encoding="utf-8")

    print(f"✅ Presentation generated → {out_file}")
    print(f"   Slides: {len(SLIDES)}")
    print()
    print("Next steps:")
    print("  1. Open locally:   open docs/index.html")
    print("  2. GitHub Pages:   Settings → Pages → Deploy from /docs on main")
    print("  3. LinkedIn:       Add GitHub Pages URL as project link")
