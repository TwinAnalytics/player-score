import { useEffect, useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { usePlayerData } from '../context/PlayerDataContext';
import { useMarketValue } from '../context/MarketValueContext';
import PageShell from '../components/layout/PageShell';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import ScoreBadge from '../components/ui/ScoreBadge';
import ClubCrest from '../components/ui/ClubCrest';
import HiddenGemsScatter from '../components/charts/HiddenGemsScatter';
import { enrichWithPrimaryScore } from '../utils/dataHelpers';
import { formatScore, formatAge, formatNineties } from '../utils/formatters';
import { POS_LABELS } from '../constants/scoring';
import styles from './HiddenGems.module.css';

const ALL_POSITIONS = ['FW', 'Off_MF', 'MF', 'Def_MF', 'DF'];

function percentileRank(value, allValues) {
  const below = allValues.filter((v) => v < value).length;
  return (below / allValues.length) * 100;
}

export default function HiddenGems() {
  const { allRows, seasons, comps, loading } = usePlayerData();
  const { getMV } = useMarketValue();
  const navigate = useNavigate();

  const [season, setSeason] = useState('');
  const [minScore, setMinScore] = useState(400);
  const [maxMV, setMaxMV] = useState(30);
  const [minNineties, setMinNineties] = useState(5);
  const [selectedComps, setSelectedComps] = useState(new Set());
  const [positions, setPositions] = useState(new Set(ALL_POSITIONS));
  const [sortCol, setSortCol] = useState('GemScore');
  const [sortDir, setSortDir] = useState('desc');

  useEffect(() => { document.title = 'Hidden Gems · PlayerScore'; }, []);
  useEffect(() => { if (seasons.length && !season) setSeason(seasons[0]); }, [seasons]);
  useEffect(() => {
    if (comps.length && selectedComps.size === 0) setSelectedComps(new Set(comps));
  }, [comps]);

  const toggleComp = (c) => {
    setSelectedComps((prev) => {
      const next = new Set(prev);
      if (next.has(c)) { if (next.size > 1) next.delete(c); }
      else next.add(c);
      return next;
    });
  };

  const togglePos = (pos) => {
    setPositions((prev) => {
      const next = new Set(prev);
      if (next.has(pos)) { if (next.size > 1) next.delete(pos); }
      else next.add(pos);
      return next;
    });
  };

  const gemsData = useMemo(() => {
    if (!allRows || !season) return [];

    const filtered = allRows.filter((r) => {
      if (r.Season !== season) return false;
      if (!positions.has(r.Pos)) return false;
      if (selectedComps.size && !selectedComps.has(r.Comp)) return false;
      const nineties = parseFloat(r['90s']) || 0;
      if (nineties < minNineties) return false;
      return true;
    });

    const enriched = enrichWithPrimaryScore(filtered).filter(
      (r) => r.MainScore !== null && !isNaN(r.MainScore) && r.MainScore >= minScore
    );

    // Join market values
    const withMV = enriched
      .map((r) => {
        const mvEur = getMV(r.Player, r.Squad);
        const mvM = mvEur ? mvEur / 1_000_000 : null;
        return { ...r, MarketValue_EUR: mvEur, MarketValue_M: mvM };
      })
      .filter((r) => r.MarketValue_M !== null && r.MarketValue_M > 0 && r.MarketValue_M <= maxMV);

    // GemScore = percentile(MainScore / MarketValue_M) / 10
    const vfm = withMV.map((r) => r.MainScore / r.MarketValue_M);
    const withGem = withMV.map((r, i) => ({
      ...r,
      VFM: vfm[i],
      GemScore: parseFloat((percentileRank(vfm[i], vfm) / 10).toFixed(1)),
    }));

    return withGem;
  }, [allRows, season, positions, selectedComps, minScore, maxMV, minNineties, getMV]);

  const sorted = useMemo(() => {
    return [...gemsData].sort((a, b) => {
      const av = a[sortCol] ?? 0;
      const bv = b[sortCol] ?? 0;
      return sortDir === 'desc' ? bv - av : av - bv;
    });
  }, [gemsData, sortCol, sortDir]);

  const handleSort = (col) => {
    if (col === sortCol) setSortDir((d) => (d === 'desc' ? 'asc' : 'desc'));
    else { setSortCol(col); setSortDir('desc'); }
  };

  const sortIndicator = (col) => {
    if (col !== sortCol) return '';
    return sortDir === 'desc' ? ' ↓' : ' ↑';
  };

  if (loading) return <PageShell><LoadingSpinner message="Loading player data…" /></PageShell>;

  return (
    <PageShell wide>
      <div className={styles.header}>
        <h1 className={styles.title}>Hidden Gems</h1>
        <p className={styles.sub}>
          Find undervalued players — high score, low market value. GemScore = value-for-money percentile (0–10).
        </p>
      </div>

      {/* Filters */}
      <div className={styles.filters}>
        <div className={styles.filterGroup}>
          <label className={styles.filterLabel}>Season</label>
          <select value={season} onChange={(e) => setSeason(e.target.value)}>
            {seasons.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
        <div className={styles.filterGroup}>
          <label className={styles.filterLabel}>Min Score: {minScore}</label>
          <input
            type="range" min={0} max={950} step={25}
            value={minScore}
            onChange={(e) => setMinScore(+e.target.value)}
            className={styles.rangeInput}
          />
        </div>
        <div className={styles.filterGroup}>
          <label className={styles.filterLabel}>Max Market Value: €{maxMV}M</label>
          <input
            type="range" min={1} max={200} step={1}
            value={maxMV}
            onChange={(e) => setMaxMV(+e.target.value)}
            className={styles.rangeInput}
          />
        </div>
        <div className={styles.filterGroup}>
          <label className={styles.filterLabel}>Min 90s: {minNineties}</label>
          <input
            type="range" min={1} max={38} step={1}
            value={minNineties}
            onChange={(e) => setMinNineties(+e.target.value)}
            className={styles.rangeInput}
          />
        </div>
      </div>

      {/* Position toggles */}
      <div className={styles.toggleRow}>
        <span className={styles.toggleLabel}>Position:</span>
        {ALL_POSITIONS.map((pos) => (
          <button
            key={pos}
            className={`${styles.toggleBtn} ${positions.has(pos) ? styles.toggleActive : ''}`}
            onClick={() => togglePos(pos)}
          >
            {POS_LABELS[pos] || pos}
          </button>
        ))}
      </div>

      {/* League toggles */}
      {comps.length > 0 && (
        <div className={styles.toggleRow}>
          <span className={styles.toggleLabel}>League:</span>
          {comps.map((c) => (
            <button
              key={c}
              className={`${styles.toggleBtn} ${selectedComps.has(c) ? styles.toggleActive : ''}`}
              onClick={() => toggleComp(c)}
            >
              {c}
            </button>
          ))}
        </div>
      )}

      <div className={styles.count}>{gemsData.length} players</div>

      {/* Scatter chart */}
      {gemsData.length > 0 && (
        <div className={styles.chartWrap}>
          <HiddenGemsScatter
            data={gemsData}
            height={420}
            onDotClick={(d) => d?.Player && navigate(`/profile?player=${encodeURIComponent(d.Player)}`)}
          />
        </div>
      )}

      {/* Table */}
      {sorted.length > 0 && (
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
                <th className={styles.sortable} onClick={() => handleSort('MainScore')}>
                  Score{sortIndicator('MainScore')}
                </th>
                <th className={styles.sortable} onClick={() => handleSort('MarketValue_M')}>
                  Market Value{sortIndicator('MarketValue_M')}
                </th>
                <th className={styles.sortable} onClick={() => handleSort('GemScore')}>
                  Gem Score{sortIndicator('GemScore')}
                </th>
                <th>Band</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((row, i) => (
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
                  <td style={{ color: 'var(--muted)' }}>{row.Comp}</td>
                  <td>
                    <span className={styles.posTag}>{POS_LABELS[row.Pos] || row.Pos}</span>
                  </td>
                  <td style={{ color: 'var(--muted)' }}>{formatAge(row.Age)}</td>
                  <td style={{ color: 'var(--muted)' }}>{formatNineties(row['90s'])}</td>
                  <td style={{ fontWeight: 700 }}>{formatScore(row.MainScore)}</td>
                  <td style={{ color: 'var(--muted)' }}>€{row.MarketValue_M?.toFixed(1)}M</td>
                  <td>
                    <div className={styles.gemScoreCell}>
                      <div className={styles.gemBar}>
                        <div
                          className={styles.gemBarFill}
                          style={{ width: `${(row.GemScore / 10) * 100}%` }}
                        />
                      </div>
                      <span className={styles.gemVal}>{row.GemScore.toFixed(1)}</span>
                    </div>
                  </td>
                  <td><ScoreBadge band={row.MainBand} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {gemsData.length === 0 && !loading && (
        <p className={styles.empty}>No players match the current filters.</p>
      )}
    </PageShell>
  );
}
