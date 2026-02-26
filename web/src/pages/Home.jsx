import { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { usePlayerData } from '../context/PlayerDataContext';
import { useSquadData } from '../context/SquadDataContext';
import KpiTile from '../components/ui/KpiTile';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import PageShell from '../components/layout/PageShell';
import { BAND_THRESHOLDS } from '../constants/scoring';
import { BAND_COLORS } from '../constants/colors';
import styles from './Home.module.css';

const FEATURE_CARDS = [
  {
    to: '/rankings',
    icon: '📊',
    title: 'Player Rankings',
    desc: 'Filter by league, club, position and find the top performers in the Big-5 this season.',
  },
  {
    to: '/profile',
    icon: '👤',
    title: 'Player Profiles',
    desc: "Deep-dive into any player's career trajectory, score breakdown, and peer comparison.",
  },
  {
    to: '/teams',
    icon: '🏟️',
    title: 'Team Scores',
    desc: 'Minute-weighted squad strength rankings across all Big-5 leagues and seasons.',
  },
];

export default function Home() {
  const { playerNames, seasons, loading } = usePlayerData();
  const { squadRows } = useSquadData();

  useEffect(() => { document.title = 'PlayerScore'; }, []);

  const numPlayers = playerNames.length;
  const numSeasons = seasons.length;
  const numLeagues = 5;

  return (
    <PageShell>
      {/* Hero */}
      <section className={styles.hero}>
        <div className={styles.heroContent}>
          <div className={styles.heroBadge}>Football Analytics · Big-5 Leagues</div>
          <h1 className={styles.heroTitle}>
            <span className={styles.accent}>Player</span>Score
          </h1>
          <p className={styles.heroSubtitle}>
            Transparent, role-aware performance scoring for every outfield player
            across the European Big-5. No black-box ML — just benchmark-driven
            metrics that make sense.
          </p>
          <div className={styles.heroCta}>
            <Link to="/rankings" className={styles.ctaPrimary}>Explore Rankings</Link>
            <Link to="/profile" className={styles.ctaSecondary}>Search Player</Link>
          </div>
        </div>
      </section>

      {/* KPIs */}
      <section className={styles.kpiRow}>
        {loading ? (
          <LoadingSpinner message="Loading stats…" />
        ) : (
          <>
            <KpiTile value={numPlayers.toLocaleString()} label="Players Tracked" sub="Unique outfield players" />
            <KpiTile value={numSeasons} label="Seasons" sub="2017/18 – 2025/26" />
            <KpiTile value={numLeagues} label="Leagues" sub="PL · La Liga · Serie A · BL · L1" />
          </>
        )}
      </section>

      {/* Features */}
      <section className={styles.features}>
        <h2 className={styles.sectionTitle}>Explore the Platform</h2>
        <div className={styles.featureGrid}>
          {FEATURE_CARDS.map(({ to, icon, title, desc }) => (
            <Link key={to} to={to} className={styles.featureCard}>
              <div className={styles.featureIcon}>{icon}</div>
              <h3 className={styles.featureTitle}>{title}</h3>
              <p className={styles.featureDesc}>{desc}</p>
              <span className={styles.featureArrow}>→</span>
            </Link>
          ))}
        </div>
      </section>

      {/* Score bands explainer */}
      <section className={styles.bands}>
        <h2 className={styles.sectionTitle}>Scoring System</h2>
        <p className={styles.sectionSub}>
          Scores range 0–1000, computed per role (FW, Off_MF, MF, Def_MF, DF) using
          benchmark-driven weights. Each metric is compared to the p95 value among all
          Big-5 players at that position — no ML, full transparency.
        </p>
        <div className={styles.bandList}>
          {BAND_THRESHOLDS.map(({ label, min, description }) => {
            const next = BAND_THRESHOLDS.find((b) => b.min < min);
            const maxScore = min === 900 ? 1000 : (BAND_THRESHOLDS.find((b) => b.min > min)?.min ?? 900) - 1;
            const range = min === 0 ? '< 200' : min === 900 ? '900 – 1000' : `${min} – ${maxScore}`;
            return (
              <div key={label} className={styles.bandRow}>
                <div
                  className={styles.bandDot}
                  style={{ background: BAND_COLORS[label] }}
                />
                <div className={styles.bandInfo}>
                  <span className={styles.bandName} style={{ color: BAND_COLORS[label] }}>
                    {label}
                  </span>
                  <span className={styles.bandRange}>{range}</span>
                </div>
                <span className={styles.bandDesc}>{description}</span>
              </div>
            );
          })}
        </div>
      </section>
    </PageShell>
  );
}
