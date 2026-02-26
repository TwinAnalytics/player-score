import { useEffect, useState, useMemo } from 'react';
import { useSquadData } from '../context/SquadDataContext';
import PageShell from '../components/layout/PageShell';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import ScoreBadge from '../components/ui/ScoreBadge';
import ClubCrest from '../components/ui/ClubCrest';
import ScoreBarChart from '../components/charts/ScoreBarChart';
import ScatterPlot from '../components/charts/ScatterPlot';
import { scoreToBand, bandColor } from '../constants/scoring';
import { formatScore, formatAge } from '../utils/formatters';
import styles from './TeamScores.module.css';

export default function TeamScores() {
  const { squadRows, seasons, comps, loading, error } = useSquadData();

  const [season, setSeason] = useState('');
  const [comp, setComp] = useState('');

  useEffect(() => { document.title = 'Team Scores · PlayerScore'; }, []);
  useEffect(() => { if (seasons.length && !season) setSeason(seasons[0]); }, [seasons]);

  const filtered = useMemo(() => {
    if (!squadRows) return [];
    let rows = squadRows;
    if (season) rows = rows.filter((r) => r.Season === season);
    if (comp) rows = rows.filter((r) => r.Comp === comp);
    return rows
      .filter((r) => parseFloat(r.OverallScore_squad) > 0)
      .sort((a, b) => parseFloat(b.OverallScore_squad) - parseFloat(a.OverallScore_squad))
      .map((r) => ({
        ...r,
        MainScore: parseFloat(r.OverallScore_squad),
        MainBand: scoreToBand(parseFloat(r.OverallScore_squad)),
        Player: r.Squad,
      }));
  }, [squadRows, season, comp]);

  const scatterData = useMemo(() => {
    return filtered.map((r) => ({
      ...r,
      x: parseFloat(r.OverallScore_squad),
      y: parseFloat(r.OffScore_squad),
    }));
  }, [filtered]);

  const filteredComps = useMemo(() => {
    if (!squadRows || !season) return comps;
    return [...new Set(squadRows.filter((r) => r.Season === season).map((r) => r.Comp))].sort();
  }, [squadRows, season, comps]);

  if (loading) return <PageShell><LoadingSpinner message="Loading squad data…" /></PageShell>;
  if (error) return <PageShell><p style={{ color: '#ef4444' }}>Error: {error}</p></PageShell>;

  return (
    <PageShell wide>
      <div className={styles.header}>
        <h1 className={styles.title}>Team Scores</h1>
        <p className={styles.sub}>Minute-weighted squad strength across the Big-5. Higher score = stronger overall squad.</p>
      </div>

      {/* Filters */}
      <div className={styles.filters}>
        <div className={styles.filterGroup}>
          <label className={styles.filterLabel}>Season</label>
          <select value={season} onChange={(e) => { setSeason(e.target.value); setComp(''); }}>
            {seasons.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
        <div className={styles.filterGroup}>
          <label className={styles.filterLabel}>League</label>
          <select value={comp} onChange={(e) => setComp(e.target.value)}>
            <option value="">All Leagues</option>
            {filteredComps.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
      </div>

      <div className={styles.count}>{filtered.length} teams</div>

      {/* Bar chart */}
      {filtered.length > 0 && (
        <div className={styles.chartWrap}>
          <h3 className={styles.chartTitle}>Squad Score Ranking</h3>
          <ScoreBarChart
            data={filtered}
            height={Math.max(300, filtered.length * 34)}
          />
        </div>
      )}

      {/* Scatter */}
      {filtered.length > 1 && (
        <div className={styles.chartWrap}>
          <h3 className={styles.chartTitle}>Overall vs. Offensive Score</h3>
          <p className={styles.chartSub}>Each dot is a team, colored by league</p>
          <ScatterPlot data={scatterData} xKey="x" yKey="y" height={360} />
          <div className={styles.legend}>
            {[...new Set(filtered.map((r) => r.Comp))].map((c) => {
              const colors = { 'Premier League': '#3b82f6', 'La Liga': '#ef4444', 'Serie A': '#22c55e', 'Bundesliga': '#f97316', 'Ligue 1': '#a855f7' };
              const col = colors[c] || '#00B8A9';
              return (
                <div key={c} className={styles.legendItem}>
                  <div className={styles.legendDot} style={{ background: col }} />
                  <span>{c}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Table */}
      {filtered.length > 0 ? (
        <div className={styles.tableWrap}>
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Team</th>
                <th>League</th>
                <th>Overall</th>
                <th>Offense</th>
                <th>Midfield</th>
                <th>Defense</th>
                <th>Avg Age</th>
                <th>Band</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((row, i) => {
                const band = scoreToBand(row.OverallScore_squad);
                return (
                  <tr key={`${row.Squad}-${i}`}>
                    <td className={styles.rank}>{i + 1}</td>
                    <td>
                      <div className={styles.teamCell}>
                        <ClubCrest clubName={row.Squad} size={22} />
                        <span className={styles.teamName}>{row.Squad}</span>
                      </div>
                    </td>
                    <td style={{ color: 'var(--muted)', fontSize: '0.82rem' }}>{row.Comp}</td>
                    <td>
                      <strong style={{ color: bandColor(band) }}>
                        {formatScore(row.OverallScore_squad)}
                      </strong>
                    </td>
                    <td style={{ color: 'var(--muted)' }}>{formatScore(row.OffScore_squad)}</td>
                    <td style={{ color: 'var(--muted)' }}>{formatScore(row.MidScore_squad)}</td>
                    <td style={{ color: 'var(--muted)' }}>{formatScore(row.DefScore_squad)}</td>
                    <td style={{ color: 'var(--muted)' }}>
                      {parseFloat(row.Age_squad_mean) ? parseFloat(row.Age_squad_mean).toFixed(1) : '—'}
                    </td>
                    <td><ScoreBadge band={band} /></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <p className={styles.empty}>No team data for the selected filters.</p>
      )}
    </PageShell>
  );
}
