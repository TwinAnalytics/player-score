import { useEffect, useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { usePlayerData } from '../context/PlayerDataContext';
import { usePizzaData } from '../context/PizzaDataContext';
import { useMarketValue } from '../context/MarketValueContext';
import PageShell from '../components/layout/PageShell';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import ClubCrest from '../components/ui/ClubCrest';
import ScoreBadge from '../components/ui/ScoreBadge';
import PizzaRadarChart from '../components/charts/PizzaRadarChart';
import { enrichWithPrimaryScore, getClubsForComp } from '../utils/dataHelpers';
import { computePizzaPercentiles } from '../utils/pizzaHelpers';
import { formatScore, formatAge, formatNineties, formatMarketValue } from '../utils/formatters';
import { getPrimaryScore, POS_LABELS, bandColor } from '../constants/scoring';
import { BAND_COLORS } from '../constants/colors';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import styles from './ComparePlayers.module.css';

const PIZZA_METRIC_LABELS = {
  Succ_Per90: 'Dribbles/90',
  'Cmp%': 'Pass Cmp%',
  PrgC_Per90: 'Prog Carries/90',
  PrgR_Per90: 'Prog Passes Rec/90',
  TB_Per90: 'Through Balls/90',
  Gls_Per90: 'Goals/90',
  Ast_Per90: 'Assists/90',
  xG_Per90: 'xG/90',
  xAG_Per90: 'xAG/90',
  SoT_Per90: 'Shots on Target/90',
  SCA_Per90: 'Shot-Creating/90',
  KP_Per90: 'Key Passes/90',
  TklW_Per90: 'Tackles Won/90',
  Int_Per90: 'Interceptions/90',
  Blocks_Per90: 'Blocks/90',
  Clr_Per90: 'Clearances/90',
};

const KEY_METRICS = [
  'Gls_Per90', 'Ast_Per90', 'xG_Per90', 'xAG_Per90',
  'KP_Per90', 'SCA_Per90', 'TklW_Per90', 'Int_Per90',
];

function PlayerSelector({ label, season, allRows, comps, onSelect, selectedPlayer }) {
  const [comp, setComp] = useState('');
  const [club, setClub] = useState('');
  const [player, setPlayer] = useState(selectedPlayer || '');

  const clubs = useMemo(() => {
    if (!allRows) return [];
    return getClubsForComp(allRows.filter((r) => r.Season === season && (!comp || r.Comp === comp)), comp);
  }, [allRows, season, comp]);

  const players = useMemo(() => {
    if (!allRows) return [];
    const filtered = allRows.filter(
      (r) =>
        r.Season === season &&
        (!comp || r.Comp === comp) &&
        (!club || r.Squad === club)
    );
    return [...new Set(filtered.map((r) => r.Player))].sort();
  }, [allRows, season, comp, club]);

  const handlePlayer = (name) => {
    setPlayer(name);
    onSelect(name);
  };

  return (
    <div className={styles.selectorPanel}>
      <div className={styles.selectorTitle}>{label}</div>
      <div className={styles.selectorControls}>
        <div className={styles.selectGroup}>
          <label className={styles.selectLabel}>League</label>
          <select value={comp} onChange={(e) => { setComp(e.target.value); setClub(''); setPlayer(''); onSelect(''); }}>
            <option value="">All Leagues</option>
            {comps.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <div className={styles.selectGroup}>
          <label className={styles.selectLabel}>Club</label>
          <select value={club} onChange={(e) => { setClub(e.target.value); setPlayer(''); onSelect(''); }}>
            <option value="">All Clubs</option>
            {clubs.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <div className={styles.selectGroup}>
          <label className={styles.selectLabel}>Player</label>
          <select value={player} onChange={(e) => handlePlayer(e.target.value)}>
            <option value="">Select player…</option>
            {players.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>
      </div>
    </div>
  );
}

export default function ComparePlayers() {
  const { allRows, seasons, comps, loading } = usePlayerData();
  const { getPizzaForPlayer, getPeers } = usePizzaData();
  const { getMV } = useMarketValue();
  const navigate = useNavigate();

  const [season, setSeason] = useState('');
  const [player1, setPlayer1] = useState('');
  const [player2, setPlayer2] = useState('');

  useEffect(() => { document.title = 'Compare Players · PlayerScore'; }, []);
  useEffect(() => { if (seasons.length && !season) setSeason(seasons[0]); }, [seasons]);

  const getPlayerData = (playerName) => {
    if (!playerName || !allRows) return null;
    const rows = allRows.filter((r) => r.Player === playerName && r.Season === season);
    if (!rows.length) return null;
    const row = rows[0];
    const enriched = enrichWithPrimaryScore([row])[0];
    const { score, band } = getPrimaryScore(enriched);
    const pizzaRow = getPizzaForPlayer(playerName, season);
    const peers = getPeers(enriched.Pos, season);
    const pizzaData = pizzaRow && peers.length ? computePizzaPercentiles(pizzaRow, peers) : [];
    const mv = getMV(playerName, enriched.Squad);
    return { row: enriched, score, band, pizzaRow, pizzaData, mv };
  };

  const p1Data = useMemo(() => getPlayerData(player1), [player1, season, allRows]);
  const p2Data = useMemo(() => getPlayerData(player2), [player2, season, allRows]);

  // Score comparison bar data
  const scoreBarData = useMemo(() => {
    if (!p1Data && !p2Data) return [];
    return [
      {
        category: 'Offense',
        [player1 || 'P1']: p1Data ? parseFloat(p1Data.row.OffScore_abs) || 0 : 0,
        [player2 || 'P2']: p2Data ? parseFloat(p2Data.row.OffScore_abs) || 0 : 0,
      },
      {
        category: 'Midfield',
        [player1 || 'P1']: p1Data ? parseFloat(p1Data.row.MidScore_abs) || 0 : 0,
        [player2 || 'P2']: p2Data ? parseFloat(p2Data.row.MidScore_abs) || 0 : 0,
      },
      {
        category: 'Defense',
        [player1 || 'P1']: p1Data ? parseFloat(p1Data.row.DefScore_abs) || 0 : 0,
        [player2 || 'P2']: p2Data ? parseFloat(p2Data.row.DefScore_abs) || 0 : 0,
      },
    ];
  }, [p1Data, p2Data, player1, player2]);

  if (loading) return <PageShell><LoadingSpinner message="Loading player data…" /></PageShell>;

  const bothSelected = p1Data && p2Data;

  return (
    <PageShell wide>
      <div className={styles.header}>
        <h1 className={styles.title}>Compare Players</h1>
        <p className={styles.sub}>Select two players from the same season to compare their profiles side by side.</p>
      </div>

      {/* Season selector */}
      <div className={styles.seasonRow}>
        <label className={styles.seasonLabel}>Season</label>
        <select value={season} onChange={(e) => { setSeason(e.target.value); setPlayer1(''); setPlayer2(''); }}>
          {seasons.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
      </div>

      {/* Player selectors */}
      <div className={styles.selectors}>
        <PlayerSelector
          label="Player 1"
          season={season}
          allRows={allRows}
          comps={comps}
          onSelect={setPlayer1}
          selectedPlayer={player1}
        />
        <div className={styles.vsLabel}>VS</div>
        <PlayerSelector
          label="Player 2"
          season={season}
          allRows={allRows}
          comps={comps}
          onSelect={setPlayer2}
          selectedPlayer={player2}
        />
      </div>

      {/* Player header cards */}
      {(p1Data || p2Data) && (
        <div className={styles.playerCards}>
          {[{ data: p1Data, name: player1, color: '#00B8A9' }, { data: p2Data, name: player2, color: '#fde68a' }].map(
            ({ data, name, color }, idx) => (
              <div key={idx} className={styles.playerCard} style={{ borderColor: `${color}44` }}>
                {data ? (
                  <>
                    <div className={styles.cardTop}>
                      <ClubCrest clubName={data.row.Squad} size={40} />
                      <div>
                        <div
                          className={styles.cardName}
                          onClick={() => navigate(`/profile?player=${encodeURIComponent(name)}`)}
                          style={{ cursor: 'pointer' }}
                        >
                          {name}
                        </div>
                        <div className={styles.cardMeta}>
                          {data.row.Squad} · {POS_LABELS[data.row.Pos] || data.row.Pos}
                        </div>
                      </div>
                    </div>
                    <div className={styles.cardStats}>
                      <div className={styles.cardStat}>
                        <span className={styles.cardStatVal} style={{ color }}>{formatScore(data.score)}</span>
                        <span className={styles.cardStatLabel}>Score</span>
                      </div>
                      <div className={styles.cardStat}>
                        <ScoreBadge band={data.band} />
                        <span className={styles.cardStatLabel}>Band</span>
                      </div>
                      <div className={styles.cardStat}>
                        <span className={styles.cardStatVal}>{formatAge(data.row.Age)}</span>
                        <span className={styles.cardStatLabel}>Age</span>
                      </div>
                      <div className={styles.cardStat}>
                        <span className={styles.cardStatVal}>{formatNineties(data.row['90s'])}</span>
                        <span className={styles.cardStatLabel}>90s</span>
                      </div>
                      {data.mv && (
                        <div className={styles.cardStat}>
                          <span className={styles.cardStatVal} style={{ fontSize: '0.9rem' }}>{formatMarketValue(data.mv)}</span>
                          <span className={styles.cardStatLabel}>Market Value</span>
                        </div>
                      )}
                    </div>
                  </>
                ) : (
                  <div className={styles.cardEmpty}>
                    {name ? `No data for "${name}"` : `Select Player ${idx + 1}`}
                  </div>
                )}
              </div>
            )
          )}
        </div>
      )}

      {bothSelected && (
        <>
          {/* Score bar chart */}
          <div className={styles.section}>
            <h3 className={styles.sectionTitle}>Score Breakdown</h3>
            <div className={styles.chartCard}>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={scoreBarData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                  <XAxis dataKey="category" stroke="var(--muted)" tick={{ fontSize: 12 }} />
                  <YAxis domain={[0, 1000]} stroke="var(--muted)" tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ background: 'var(--bg-second)', border: '1px solid var(--border)', borderRadius: 8 }}
                    labelStyle={{ color: 'var(--text)', fontWeight: 700 }}
                    itemStyle={{ color: 'var(--muted)' }}
                    formatter={(val) => Math.round(val)}
                  />
                  <Legend />
                  <Bar dataKey={player1} fill="#00B8A9" radius={[4, 4, 0, 0]} maxBarSize={60} />
                  <Bar dataKey={player2} fill="#fde68a" radius={[4, 4, 0, 0]} maxBarSize={60} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Pizza charts */}
          {(p1Data.pizzaData.length > 0 || p2Data.pizzaData.length > 0) && (
            <div className={styles.section}>
              <h3 className={styles.sectionTitle}>Score Profile (Percentile vs. Peers)</h3>
              <div className={styles.pizzaRow}>
                <div className={styles.pizzaCol}>
                  <div className={styles.pizzaTitle} style={{ color: '#00B8A9' }}>{player1}</div>
                  {p1Data.pizzaData.length > 0
                    ? <PizzaRadarChart data={p1Data.pizzaData} height={340} />
                    : <div className={styles.noPizza}>No pizza data available</div>}
                </div>
                <div className={styles.pizzaCol}>
                  <div className={styles.pizzaTitle} style={{ color: '#fde68a' }}>{player2}</div>
                  {p2Data.pizzaData.length > 0
                    ? <PizzaRadarChart data={p2Data.pizzaData} height={340} />
                    : <div className={styles.noPizza}>No pizza data available</div>}
                </div>
              </div>
            </div>
          )}

          {/* Key metrics table */}
          {(p1Data.pizzaRow || p2Data.pizzaRow) && (
            <div className={styles.section}>
              <h3 className={styles.sectionTitle}>Key Metrics (per 90)</h3>
              <div className={styles.metricsTableWrap}>
                <table className={styles.metricsTable}>
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th style={{ color: '#00B8A9' }}>{player1}</th>
                      <th style={{ color: '#fde68a' }}>{player2}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {KEY_METRICS.map((key) => {
                      const v1 = p1Data.pizzaRow ? parseFloat(p1Data.pizzaRow[key]) : null;
                      const v2 = p2Data.pizzaRow ? parseFloat(p2Data.pizzaRow[key]) : null;
                      const p1Better = v1 !== null && v2 !== null && v1 > v2;
                      const p2Better = v1 !== null && v2 !== null && v2 > v1;
                      return (
                        <tr key={key}>
                          <td className={styles.metricName}>{PIZZA_METRIC_LABELS[key] || key}</td>
                          <td className={p1Better ? styles.metricBetter : ''}>
                            {v1 !== null && !isNaN(v1) ? v1.toFixed(2) : '—'}
                          </td>
                          <td className={p2Better ? styles.metricBetter : ''}>
                            {v2 !== null && !isNaN(v2) ? v2.toFixed(2) : '—'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {!player1 && !player2 && (
        <div className={styles.emptyState}>
          <p>Select two players above to compare them.</p>
        </div>
      )}
    </PageShell>
  );
}
