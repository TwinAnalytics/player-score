What PlayerScore Does

PlayerScore transforms raw FBref data into interpretable, role-aware insights, making it easier to compare players across leagues, seasons, and clubs.

⸻

📦 Data Acquisition
	•	Automated scraping of Big-5 player stats (FBref) using Playwright
	•	Multi-season dataset from 2017/18 to 2025/26
	•	Robust handling of missing or league-limited stats

⸻

🧠 Feature Engineering
	•	Per-90 normalization for all relevant metrics
	•	Minutes thresholds and data quality filters
	•	Unified positional logic to classify players into:
	•	FW / Off_MF (offensive roles)
	•	MF (midfield roles)
	•	DF / Def_MF (defensive roles)

⸻

📊 Role-Specific Scoring

Each player receives up to three interpretable scores:
	•	Offensive Score (FW, Off_MF)
	•	Midfield Score (MF)
	•	Defensive Score (DF, Def_MF)

These scores are built using:
	•	Distribution-aware normalization
	•	Multi-season benchmarking
	•	Transparent performance tiering

⸻

🖥️ App Features (Streamlit UI)

The included Streamlit app allows fully interactive exploration of all data.

⸻

👤 Player Profiles
	•	Per-season and career views
	•	Pizza charts vs Big-5 role peers
	•	Role-based scatter plots (e.g., xG vs G, xAG vs A)
	•	Career score trend lines
	•	Summary tiles (age, minutes, score, band)

⸻

📊 Top Lists
	•	Season, league, club, position, minutes, and age filters
	•	Top-N bar charts by primary role score
	•	Score vs age beeswarm plot
	•	Band distribution visualizations for filtered sets

⸻

🟦 NEW: Team Scores
	•	Squad-level offense, midfield, and defense rankings
	•	Comparison of squad strength within a league
	•	Identification of top contributors per club
	•	Multi-season squad trends and development analysis

⸻

❓ Why PlayerScore?

Modern football recruitment needs transparent, interpretable, reproducible metrics — not black-box models.

PlayerScore is built around:
	•	Consistency across leagues and competitions
	•	Role-aware evaluation based on real positional behavior
	•	Reproducible scoring logic using open data
	•	Explorable analytics for scouting, recruitment, and squad planning

