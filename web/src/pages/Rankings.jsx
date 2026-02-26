import { useEffect, useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { usePlayerData } from '../context/PlayerDataContext';
import PageShell from '../components/layout/PageShell';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import ScoreBadge from '../components/ui/ScoreBadge';
import ScoreBar from '../components/ui/ScoreBar';
import ClubCrest from '../components/ui/ClubCrest';
import ScoreBarChart from '../components/charts/ScoreBarChart';
import BeeswarmChart from '../components/charts/BeeswarmChart';
import BandHistogram from '../components/charts/BandHistogram';
import { enrichWithPrimaryScore, filterRows, getClubsForComp } from '../utils/dataHelpers';
import { formatScore, formatNineties, formatAge } from '../utils/formatters';
import { POS_LABELS } from '../constants/scoring';
import styles from './Rankings.module.css';

const ALL_POSITIONS = ['FW', 'Off_MF', 'MF', 'Def_MF', 'DF'];
const VIEWS = ['Top Players', 'Score vs Age', 'Band Distribution'];

export default function Rankings() {
  const { allRows, seasons, comps, loading, error } = usePlayerData();
  const navigate = useNavigate();

  const [season, setSeason] = useState('');
  const [comp, setComp] = useState('');
  const [club, setClub] = useState('');
  const [positions, setPositions] = useState(new Set(ALL_POSITIONS));
  const [topN, setTopN] = useState(25);
  const [view, setView] = useState('Top Players');

  useEffect(() => { document.title = 'Rankings · PlayerScore'; }, []);
  useEffect(() => { if (seasons.length && !season) setSeason(seasons[0]); }, [seasons]);

  const clubs = useMemo(() => {
    if (!allRows) return [];
    return getClubsForComp(allRows.filter((r) => !season || r.Season === season), comp);
  }, [allRows, season, comp]);

  const allFiltered = useMemo(() => {
    if (!allRows) return [];
    const rows = filterRows(allRows, {
      season,
      comp: comp || undefined,
      club: club || undefined,
      positions: [...positions],
      minNineties: 5,
    });
    return enrichWithPrimaryScore(rows).filter((r) => r.MainScore !== null && !isNaN(r.MainScore));
  }, [allRows, season, comp, club, positions]);

  const topFiltered = useMemo(
    () => [...allFiltered].sort((a, b) => b.MainScore - a.MainScore).slice(0, topN),
    [allFiltered, topN]
  );

  const togglePos = (pos) => {
    setPositions((prev) => {
      const next = new Set(prev);
      if (next.has(pos)) { if (next.size > 1) next.delete(pos); }
      else next.add(pos);
      return next;
    });
    setClub('');
  };

  const handleBarClick = (data) => {
    if (data?.Player) navigate(`/profile?player=${encodeURIComponent(data.Player)}`);
  };

  if (loading) return <PageShell><LoadingSpinner message="Loading player data…" /></PageShell>;
  if (error) return <PageShell><p style={{ color: '#ef4444' }}>Error: {error}</p></PageShell>;

  return (
    <PageShell wide>
      <div className={styles.header}>
        <h1 className={styles.title}>Player Rankings</h1>
        <p className={styles.sub}>Role-aware scores for every outfield player. Click a row or bar to view their profile.</p>
      </div>

      {/* Filters */}
      <div className={styles.filters}>
        <div className={styles.filterGroup}>
          <label className={styles.filterLabel}>Season</label>
          <select value={season} onChange={(e) => { setSeason(e.target.value); setClub(''); }}>
            {seasons.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
        <div className={styles.filterGroup}>
          <label className={styles.filterLabel}>League</label>
          <select value={comp} onChange={(e) => { setComp(e.target.value); setClub(''); }}>
            <option value="">All Leagues</option>
            {comps.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <div className={styles.filterGroup}>
          <label className={styles.filterLabel}>Club</label>
          <select value={club} onChange={(e) => setClub(e.target.value)}>
            <option value="">All Clubs</option>
            {clubs.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        {view === 'Top Players' && (
          <div className={styles.filterGroup}>
            <label className={styles.filterLabel}>Top N</label>
            <select value={topN} onChange={(e) => setTopN(+e.target.value)}>
              {[10, 15, 25, 50].map((n) => <option key={n} value={n}>Top {n}</option>)}
            </select>
          </div>
        )}
      </div>

      {/* Position toggles */}
      <div className={styles.posToggles}>
        {ALL_POSITIONS.map((pos) => (
          <button
            key={pos}
            className={`${styles.posBtn} ${positions.has(pos) ? styles.posBtnActive : ''}`}
            onClick={() => togglePos(pos)}
          >
            {POS_LABELS[pos] || pos}
          </button>
        ))}
      </div>

      {/* View toggle */}
      <div className={styles.viewToggle}>
        {VIEWS.map((v) => (
          <button
            key={v}
            className={`${styles.viewBtn} ${view === v ? styles.viewBtnActive : ''}`}
            onClick={() => setView(v)}
          >
            {v}
          </button>
        ))}
      </div>

      <div className={styles.count}>{allFiltered.length} players</div>

      {/* Top Players view */}
      {view === 'Top Players' && topFiltered.length > 0 && (
        <>
          <div className={styles.chartWrap}>
            <ScoreBarChart
              data={topFiltered}
              height={Math.max(320, topFiltered.length * 36)}
              onBarClick={handleBarClick}
            />
          </div>
          <div className={styles.tableWrap}>
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Player</th>
                  <th>Club</th>
                  <th>League</th>
                  <th>Pos</th>
                  <th>Age</th>
                  <th>90s</th>
                  <th>Score</th>
                  <th>Band</th>
                </tr>
              </thead>
              <tbody>
                {topFiltered.map((row, i) => (
                  <tr
                    key={`${row.Player}-${i}`}
                    className={styles.tableRow}
                    onClick={() => navigate(`/profile?player=${encodeURIComponent(row.Player)}`)}
                  >
                    <td className={styles.rank}>{i + 1}</td>
                    <td className={styles.playerCell}>
                      <span className={styles.playerName}>{row.Player}</span>
                    </td>
                    <td>
                      <div className={styles.clubCell}>
                        <ClubCrest clubName={row.Squad} size={20} />
                        <span>{row.Squad}</span>
                      </div>
                    </td>
                    <td className={styles.muted}>{row.Comp}</td>
                    <td><span className={styles.posTag}>{row.Pos}</span></td>
                    <td className={styles.muted}>{formatAge(row.Age)}</td>
                    <td className={styles.muted}>{formatNineties(row['90s'])}</td>
                    <td>
                      <div className={styles.scoreCell}>
                        <ScoreBar score={row.MainScore} band={row.MainBand} />
                        <span className={styles.scoreVal}>{formatScore(row.MainScore)}</span>
                      </div>
                    </td>
                    <td><ScoreBadge band={row.MainBand} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* Beeswarm view */}
      {view === 'Score vs Age' && allFiltered.length > 0 && (
        <div className={styles.chartWrap}>
          <div style={{ padding: '0 0.5rem 0.5rem', fontSize: '0.82rem', color: 'var(--muted)' }}>
            Click a dot to open the player profile. Top {topN} players labeled.
          </div>
          <BeeswarmChart
            data={[...allFiltered].sort((a, b) => b.MainScore - a.MainScore)}
            topN={topN}
            onDotClick={(d) => d?.Player && navigate(`/profile?player=${encodeURIComponent(d.Player)}`)}
            height={450}
          />
        </div>
      )}

      {/* Band distribution */}
      {view === 'Band Distribution' && allFiltered.length > 0 && (
        <div className={styles.chartWrap}>
          <BandHistogram
            data={allFiltered}
            height={320}
            title={`Band distribution — ${comp || 'All leagues'} · ${season}`}
          />
        </div>
      )}

      {allFiltered.length === 0 && (
        <p className={styles.empty}>No players match the current filters.</p>
      )}
    </PageShell>
  );
}
