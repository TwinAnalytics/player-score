"""
One-time script to generate PlayerScore_Roadmap_2026.pdf using fpdf2.
Run with:  python generate_roadmap_pdf.py
"""

from fpdf import FPDF

OUTPUT = "PlayerScore_Roadmap_2026.pdf"

# ── Color palette ──────────────────────────────────────────────────────────
TEAL   = (0, 184, 169)
DARK   = (13, 17, 23)
GREY   = (75, 85, 99)
WHITE  = (249, 250, 251)
AMBER  = (253, 230, 138)
PURPLE = (168, 85, 247)
GREEN  = (34, 197, 94)


class RoadmapPDF(FPDF):
    """Custom FPDF subclass with consistent styling."""

    def header(self):
        self.set_fill_color(*DARK)
        self.rect(0, 0, 210, 12, "F")
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*TEAL)
        self.set_xy(10, 3)
        self.cell(0, 6, "PlayerScore · Strategic Roadmap 2026", ln=False)
        self.set_xy(-60, 3)
        self.set_text_color(*GREY)
        self.cell(50, 6, "Confidential - TwinAnalytics", align="R")

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*GREY)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    # ── Helpers ──────────────────────────────────────────────────────────

    def chapter_title(self, num: int, title: str):
        self.ln(6)
        self.set_fill_color(*TEAL)
        self.set_text_color(*DARK)
        self.set_font("Helvetica", "B", 13)
        self.cell(0, 9, f"  {num}.  {title}", ln=True, fill=True)
        self.ln(3)
        self.set_text_color(30, 30, 30)

    def section_title(self, title: str):
        self.ln(3)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*TEAL)
        self.cell(0, 7, title, ln=True)
        self.set_text_color(30, 30, 30)

    def body(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bullet(self, items: list[str], indent: int = 8):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        for item in items:
            self.set_x(self.l_margin + indent)
            self.cell(5, 5.5, chr(149))  # bullet dot
            self.multi_cell(0, 5.5, item)

    def kv_row(self, key: str, value: str, key_w: int = 50):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*TEAL)
        self.cell(key_w, 6, key)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 6, value)

    def table_header(self, cols: list[tuple[str, int]]):
        self.set_fill_color(*DARK)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 9)
        for label, w in cols:
            self.cell(w, 7, label, border=0, fill=True, align="C")
        self.ln()

    def table_row(self, vals: list[tuple[str, int]], shade: bool = False):
        if shade:
            self.set_fill_color(240, 240, 240)
        else:
            self.set_fill_color(255, 255, 255)
        self.set_text_color(30, 30, 30)
        self.set_font("Helvetica", "", 9)
        for text, w in vals:
            self.cell(w, 6.5, text, border="B", fill=True)
        self.ln()

    def colored_badge(self, text: str, rgb: tuple):
        self.set_fill_color(*rgb)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 9)
        self.cell(len(text) * 2.5 + 4, 6, text, fill=True)
        self.set_text_color(30, 30, 30)


# ══════════════════════════════════════════════════════════════════════════════
# Build document
# ══════════════════════════════════════════════════════════════════════════════

def build_pdf() -> RoadmapPDF:
    pdf = RoadmapPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 18, 15)

    # ── Cover page ──────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_fill_color(*DARK)
    pdf.rect(0, 0, 210, 297, "F")

    pdf.set_y(55)
    pdf.set_font("Helvetica", "B", 36)
    pdf.set_text_color(*TEAL)
    pdf.cell(0, 14, "PlayerScore", align="C", ln=True)

    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 8, "Strategic Roadmap 2026", align="C", ln=True)

    pdf.ln(6)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*GREY)
    pdf.cell(0, 6, "Football Analytics Platform · TwinAnalytics", align="C", ln=True)
    pdf.cell(0, 6, "February 2026 · Confidential", align="C", ln=True)

    pdf.ln(20)
    pdf.set_fill_color(*TEAL)
    pdf.rect(30, pdf.get_y(), 150, 1, "F")

    pdf.ln(14)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(180, 180, 180)
    tagline = (
        "Transparent, benchmark-driven player scoring for professional clubs,\n"
        "scouts, and football data enthusiasts - no ML black-box, just clarity."
    )
    pdf.multi_cell(0, 6, tagline, align="C")

    # ── Chapter 1: Executive Summary ────────────────────────────────────
    pdf.add_page()
    pdf.set_fill_color(255, 255, 255)
    pdf.rect(0, 12, 210, 285, "F")

    pdf.chapter_title(1, "Executive Summary")

    pdf.section_title("What is PlayerScore?")
    pdf.body(
        "PlayerScore is an open-source football analytics platform that computes "
        "role-aware performance scores (0-1000) for outfield players across Europe's "
        "Big-5 leagues. Data is scraped weekly from FBref via Playwright and processed "
        "through a transparent, benchmark-driven pipeline - no machine learning, no "
        "hidden weights, full auditability."
    )

    pdf.section_title("Unique Selling Points")
    pdf.bullet([
        "Transparent scoring: every weight and benchmark is human-readable and version-controlled.",
        "Role-specific: separate scoring models for FW, Off_MF, MF, Def_MF, DF.",
        "Weekly fresh data: GitHub Actions CI/CD updates every Tuesday on a self-hosted macOS runner.",
        "Pizza charts with peer-percentile benchmarking across Big-5 competitions.",
        "Market value integration via Transfermarkt (fuzzy-matched).",
        "Hidden Gems detector: high score, low market value - actionable scouting leads.",
        "Fully open source and deployable on Streamlit Cloud for zero hosting cost.",
    ])

    pdf.section_title("Target Audiences")
    pdf.bullet([
        "Football data enthusiasts & analysts - exploratory scouting tool.",
        "Amateur clubs & academies - affordable alternative to Wyscout/Hudl.",
        "Sports journalists - data-backed player profiles.",
        "Recruitment agencies - shortlisting and valuation cross-check.",
        "Aspiring football analysts - learning resource and portfolio showcase.",
    ])

    # ── Chapter 2: Competitive Positioning ──────────────────────────────
    pdf.chapter_title(2, "Competitive Positioning")

    pdf.section_title("Market Landscape")
    pdf.body(
        "The football analytics market is dominated by expensive enterprise tools. "
        "PlayerScore occupies a unique niche: free, explainable, and community-driven."
    )

    cols = [("Platform", 38), ("Data", 32), ("Scoring", 32), ("Price/yr", 28), ("Transparency", 40)]
    pdf.table_header(cols)

    rows = [
        (["PlayerScore",    "FBref (weekly)",     "Rule-based, open",  "Free",       "Full (OSS)"], False),
        (["Smarterscout",   "Opta",               "ML black-box",      "~EUR 3k-EUR 8k",   "None"],        True),
        (["Wyscout / Hudl", "Video + event data", "Proprietary",       "~EUR 5k-EUR 20k",  "None"],        False),
        (["FBref.com",      "StatsBomb",          "Raw stats only",    "Free",       "N/A (raw)"],   True),
        (["Sofascore",      "Own tracking",       "Rating (opaque)",   "Free/EUR 8/mo", "Minimal"],     False),
    ]
    for vals, shade in rows:
        pdf.table_row([(v, w) for v, (_, w) in zip(vals, cols)], shade=shade)

    pdf.ln(4)
    pdf.section_title("Key Differentiator")
    pdf.body(
        "While competitors offer depth (video, event data, tracking), PlayerScore offers "
        "breadth + transparency at zero cost. Analysts can fork the repo, modify benchmarks "
        "for a specific league or age group, and run the pipeline on their own data."
    )

    # ── Chapter 3: Monetisation Strategy ────────────────────────────────
    pdf.add_page()
    pdf.chapter_title(3, "Monetisation Strategy")

    pdf.section_title("Tier Overview")
    pdf.body(
        "PlayerScore will follow a freemium SaaS model with four tiers. "
        "The free tier drives organic growth; paid tiers unlock data exports, "
        "API access, and white-label deployment."
    )

    tiers = [
        ("Free",       "EUR 0 / month",  "All current features - full Big-5 history, pizza charts, rankings."),
        ("Pro",        "EUR 9 / month",  "CSV & PNG exports, advanced filters, season-over-season trend charts."),
        ("Scout",      "EUR 29 / month", "API access (JSON), comparison reports (PDF), Telegram alerts for top finds."),
        ("Club / B2B", "Custom",      "White-label deployment, custom benchmarks, priority data updates, SLA."),
    ]
    tier_colors = [GREEN, TEAL, PURPLE, AMBER]

    for (name, price, desc), color in zip(tiers, tier_colors):
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*color)
        pdf.cell(35, 7, name)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*DARK)
        pdf.cell(35, 7, price)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 7, desc)

    pdf.ln(4)
    pdf.section_title("Revenue Projections (Year 1)")
    rev_cols = [("Tier", 40), ("Target Users", 35), ("Monthly", 35), ("Annual", 40), ("Notes", 40)]
    pdf.table_header(rev_cols)
    rev_rows = [
        (["Pro",        "200",  "EUR 1,800",  "EUR 21,600",  "Early adopters"],      False),
        (["Scout",      "30",   "EUR 870",    "EUR 10,440",  "Freelance scouts"],    True),
        (["Club/B2B",   "2",    "EUR 800",    "EUR 9,600",   "Regional clubs"],      False),
        (["Total",      "232",  "EUR 3,470",  "EUR 41,640",  "Conservative case"],   True),
    ]
    for vals, shade in rev_rows:
        pdf.table_row([(v, w) for v, (_, w) in zip(vals, rev_cols)], shade=shade)

    # ── Chapter 4: Feature Roadmap ───────────────────────────────────────
    pdf.add_page()
    pdf.chapter_title(4, "Feature Roadmap Q1-Q4 2026")

    quarters = {
        "Q1 2026 (Jan-Mar)": [
            "Player comparison mode (two-player side-by-side pizza + score bars)",
            "Download PNG player card (share-ready)",
            "Rankings -> click row -> jump to player profile",
            "Dark mode refinement (GitHub-Dark palette)",
            "PDF roadmap generation (fpdf2)",
        ],
        "Q2 2026 (Apr-Jun)": [
            "Pro tier launch on Stripe (CSV export, PNG download, trend charts)",
            "Season-over-season score trend line per player",
            "Goalkeeper scoring model (separate from outfield)",
            "Telegram bot: weekly 'Hidden Gem' alert for subscribers",
            "Club squad strength heatmap (all positions visualised)",
        ],
        "Q3 2026 (Jul-Sep)": [
            "Scout tier: REST API (FastAPI) with JSON endpoints",
            "PDF comparison reports (two players, automated layout)",
            "xG over-/underperformance tracker per season",
            "Support for second-tier leagues (Championship, 2. Bundesliga, Serie B)",
            "User accounts (Supabase) for saving watchlists",
        ],
        "Q4 2026 (Oct-Dec)": [
            "Club/B2B white-label: custom logo, custom benchmarks, private deployment",
            "Age curve modelling (prime years vs. current score)",
            "Transfer value estimator (regression on score + age + position)",
            "Mobile-first Streamlit layout (responsive sidebar)",
            "Open API launch with public docs (Swagger/OpenAPI)",
        ],
    }

    q_colors = [TEAL, GREEN, PURPLE, AMBER]
    for (quarter, items), color in zip(quarters.items(), q_colors):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(*color)
        pdf.set_text_color(*DARK)
        pdf.cell(0, 7, f"  {quarter}", fill=True, ln=True)
        pdf.bullet(items, indent=5)
        pdf.ln(2)

    # ── Chapter 5: Technical Next Steps ─────────────────────────────────
    pdf.add_page()
    pdf.chapter_title(5, "Technical Next Steps")

    tech_items = {
        "Infrastructure": [
            "Migrate CI/CD from self-hosted macOS runner to GitHub-hosted (ubuntu-latest) for reliability.",
            "Add Playwright caching to reduce cold-start scraping time from ~25 min to ~10 min.",
            "Docker-compose for local development (Streamlit + data volume mount).",
        ],
        "Data Pipeline": [
            "Add data validation step (Great Expectations or Pandera) after scraping.",
            "Version processed CSVs with DVC for reproducible analysis.",
            "Add Goalkeeper scraping category (currently excluded from outfield scoring).",
        ],
        "Scoring Engine": [
            "Introduce season-weighted career scores (recent seasons count more).",
            "Add role confidence score: how many 90s needed for reliable ranking.",
            "Benchmark auto-calibration: refit p95 benchmarks annually from fresh data.",
        ],
        "Application": [
            "Extract render functions from app.py (4,500 lines) into src/views/ modules.",
            "Add pytest suite for scoring.py and processing.py (target: 80% coverage).",
            "Streamlit caching audit: replace st.cache_data with smarter TTL settings.",
        ],
        "Security & Privacy": [
            "Add rate limiting on the future API (FastAPI middleware).",
            "GDPR-compliant usage analytics (self-hosted Umami or Plausible).",
            "Secret scanning in CI (GitHub Advanced Security).",
        ],
    }

    for section, items in tech_items.items():
        pdf.section_title(section)
        pdf.bullet(items)

    # ── Chapter 6: KPIs ─────────────────────────────────────────────────
    pdf.chapter_title(6, "Key Performance Indicators")

    kpi_cols = [("KPI", 65), ("Target (6 months)", 55), ("Target (12 months)", 55), ("Measurement", 25)]
    pdf.table_header(kpi_cols)
    kpis = [
        (["Monthly Active Users (Streamlit)",   "500",      "2,000",     "Streamlit"],  False),
        (["GitHub Stars",                        "150",      "500",       "GitHub"],     True),
        (["Paying subscribers (Pro+)",           "50",       "250",       "Stripe"],     False),
        (["Monthly Recurring Revenue",           "EUR 500",     "EUR 3,500",    "Stripe"],     True),
        (["Pipeline success rate (CI)",          ">= 95%",    ">= 98%",     "GH Actions"], False),
        (["Data freshness (hours since update)", "< 168 h",  "< 48 h",    "App footer"], True),
        (["Test coverage (scoring + processing)","30%",      "80%",       "Codecov"],    False),
    ]
    for vals, shade in kpis:
        pdf.table_row([(v, w) for v, (_, w) in zip(vals, kpi_cols)], shade=shade)

    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(*GREY)
    pdf.cell(0, 6, "Generated by generate_roadmap_pdf.py · PlayerScore · TwinAnalytics · February 2026", align="C")

    return pdf


if __name__ == "__main__":
    pdf = build_pdf()
    pdf.output(OUTPUT)
    print(f"PDF written to: {OUTPUT}")
