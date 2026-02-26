import { useEffect, useState, useMemo, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { usePlayerData } from '../context/PlayerDataContext';
import { usePizzaData } from '../context/PizzaDataContext';
import { useMarketValue } from '../context/MarketValueContext';
import PageShell from '../components/layout/PageShell';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import ScoreBadge from '../components/ui/ScoreBadge';
import ClubCrest from '../components/ui/ClubCrest';
import FIFACard from '../components/ui/FIFACard';
import PizzaRadarChart from '../components/charts/PizzaRadarChart';
import ScoreTrendLine from '../components/charts/ScoreTrendLine';
import AgeCurveChart from '../components/charts/AgeCurveChart';
import RoleScatterChart from '../components/charts/RoleScatterChart';
import SimilarPlayers from '../components/ui/SimilarPlayers';
import { getPrimaryScore, scoreToBand, bandColor, POS_LABELS } from '../constants/scoring';
import { computeRadarData, enrichWithPrimaryScore } from '../utils/dataHelpers';
import { computePizzaPercentiles, findSimilarByPizza, generateScoutingText } from '../utils/pizzaHelpers';
import { formatScore, formatNineties, formatAge, formatMarketValue } from '../utils/formatters';
import { generatePlayerPDF } from '../utils/pdfExport';
import styles from './PlayerProfile.module.css';

const VIEWS = ['Season View', 'Role Context', 'Age Curve', 'Career Trend'];

export default function PlayerProfile() {
  const { playerMap, playerNames, allRows, loading } = usePlayerData();
  const { getPizzaForPlayer, getPeers, loading: pizzaLoading } = usePizzaData();
  const { getMV } = useMarketValue();
  const [searchParams, setSearchParams] = useSearchParams();

  const [query, setQuery] = useState(searchParams.get('player') || '');
  const [selectedPlayer, setSelectedPlayer] = useState(searchParams.get('player') || '');
  const [selectedSeason, setSelectedSeason] = useState('');
  const [view, setView] = useState('Season View');
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [exportingPDF, setExportingPDF] = useState(false);

  useEffect(() => { document.title = `${selectedPlayer || 'Player Profile'} · PlayerScore`; }, [selectedPlayer]);

  const playerRows = useMemo(() => {
    if (!selectedPlayer || !playerMap.size) return [];
    return playerMap.get(selectedPlayer) || [];
  }, [selectedPlayer, playerMap]);

  const playerSeasons = useMemo(() => {
    return [...new Set(playerRows.map((r) => r.Season))].sort().reverse();
  }, [playerRows]);

  useEffect(() => {
    if (playerSeasons.length && !selectedSeason) setSelectedSeason(playerSeasons[0]);
  }, [playerSeasons]);

  const currentRow = useMemo(() => {
    if (!playerRows.length) return null;
    return playerRows.find((r) => r.Season === selectedSeason) || playerRows[0];
  }, [playerRows, selectedSeason]);

  const { score: mainScore, band: mainBand } = currentRow ? getPrimaryScore(currentRow) : {};
  const marketValue = currentRow ? getMV(currentRow.Player, currentRow.Squad) : null;

  // Pizza data
  const pizzaRow = currentRow ? getPizzaForPlayer(currentRow.Player, currentRow.Season) : null;
  const pizzaPeers = currentRow ? getPeers(currentRow.Pos, currentRow.Season) : [];

  const pizzaData = useMemo(() => {
    if (pizzaRow && pizzaPeers.length) return computePizzaPercentiles(pizzaRow, pizzaPeers);
    if (currentRow && allRows) {
      const peers = allRows.filter((r) => r.Pos === currentRow.Pos && r.Season === currentRow.Season);
      return computeRadarData(currentRow, peers);
    }
    return [];
  }, [pizzaRow, pizzaPeers, currentRow, allRows]);

  // Similar players
  const similarPlayers = useMemo(() => {
    if (!currentRow || !allRows) return [];
    if (pizzaRow && pizzaPeers.length > 5) {
      const similar = findSimilarByPizza(pizzaRow, pizzaPeers, 5);
      return enrichWithPrimaryScore(similar.map((p) => {
        const scoreRow = allRows.find((r) => r.Player === p.Player && r.Season === currentRow.Season && r.Pos === currentRow.Pos);
        return scoreRow || p;
      })).filter((r) => r.Player !== currentRow.Player && r.MainScore !== null);
    }
    return [];
  }, [pizzaRow, pizzaPeers, currentRow, allRows]);

  // Career data
  const careerData = useMemo(() => {
    return enrichWithPrimaryScore(playerRows)
      .filter((r) => r.MainScore !== null && !isNaN(r.MainScore))
      .sort((a, b) => a.Season.localeCompare(b.Season));
  }, [playerRows]);

  // Autocomplete
  const suggestions = useMemo(() => {
    if (!query || query.length < 2) return [];
    const q = query.toLowerCase();
    return playerNames.filter((n) => n.toLowerCase().includes(q)).slice(0, 8);
  }, [query, playerNames]);

  const handleSelect = (name) => {
    setSelectedPlayer(name);
    setQuery(name);
    setShowSuggestions(false);
    setSelectedSeason('');
    setSearchParams({ player: name });
  };

  const handleInputKeyDown = (e) => {
    if (e.key === 'Enter' && suggestions.length > 0) handleSelect(suggestions[0]);
    if (e.key === 'Escape') setShowSuggestions(false);
  };

  const handleExportPDF = useCallback(async () => {
    if (!currentRow || !mainScore) return;
    setExportingPDF(true);
    try {
      const scoutingText = generateScoutingText(currentRow, mainScore, mainBand, marketValue);
      await generatePlayerPDF({
        row: currentRow,
        score: mainScore,
        band: mainBand,
        bandColor: bandColor(mainBand),
        offScore: parseFloat(currentRow.OffScore_abs) || 0,
        midScore: parseFloat(currentRow.MidScore_abs) || 0,
        defScore: parseFloat(currentRow.DefScore_abs) || 0,
        marketValue,
        scoutingText,
        season: selectedSeason,
      });
    } finally {
      setExportingPDF(false);
    }
  }, [currentRow, mainScore, mainBand, marketValue, selectedSeason]);

  if (loading) return <PageShell><LoadingSpinner message="Loading player data…" /></PageShell>;

  return (
    <PageShell>
      <div className={styles.header}>
        <h1 className={styles.title}>Player Profile</h1>
      </div>

      {/* Search */}
      <div className={styles.searchWrap}>
        <div className={styles.searchBox}>
          <input
            className={styles.searchInput}
            type="text"
            placeholder="Search for a player…"
            value={query}
            onChange={(e) => { setQuery(e.target.value); setShowSuggestions(true); }}
            onFocus={() => setShowSuggestions(true)}
            onBlur={() => setTimeout(() => setShowSuggestions(false), 150)}
            onKeyDown={handleInputKeyDown}
          />
          {showSuggestions && suggestions.length > 0 && (
            <ul className={styles.suggestions}>
              {suggestions.map((name) => (
                <li key={name} className={styles.suggestion} onMouseDown={() => handleSelect(name)}>
                  {name}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {!selectedPlayer && (
        <div className={styles.empty}><p>Search for a player to view their profile.</p></div>
      )}

      {selectedPlayer && !currentRow && (
        <p style={{ color: 'var(--muted)' }}>No data found for "{selectedPlayer}".</p>
      )}

      {currentRow && (
        <>
          {/* Player header */}
          <div className={styles.playerHeader}>
            <div className={styles.playerHeaderLeft}>
              <ClubCrest clubName={currentRow.Squad} size={52} />
              <div>
                <h2 className={styles.playerName}>{selectedPlayer}</h2>
                <div className={styles.playerMeta}>
                  <span>{currentRow.Squad}</span>
                  <span className={styles.dot}>·</span>
                  <span>{currentRow.Comp}</span>
                  <span className={styles.dot}>·</span>
                  <span>{POS_LABELS[currentRow.Pos] || currentRow.Pos}</span>
                </div>
              </div>
            </div>
            <div className={styles.headerActions}>
              {playerSeasons.length > 1 && (
                <select
                  value={selectedSeason}
                  onChange={(e) => { setSelectedSeason(e.target.value); setView('Season View'); }}
                  className={styles.seasonSel}
                >
                  {playerSeasons.map((s) => <option key={s} value={s}>{s}</option>)}
                </select>
              )}
              <button
                className={styles.pdfBtn}
                onClick={handleExportPDF}
                disabled={exportingPDF}
              >
                {exportingPDF ? 'Generating…' : 'Export PDF'}
              </button>
            </div>
          </div>

          {/* View toggle */}
          <div className={styles.viewToggle}>
            {VIEWS.map((v) => (
              <button
                key={v}
                className={`${styles.toggleBtn} ${view === v ? styles.toggleActive : ''}`}
                onClick={() => setView(v)}
              >
                {v}
              </button>
            ))}
          </div>

          {/* Charts */}
          <div className={styles.chartSection}>

            {/* Season View: FIFA Card + Pizza side by side */}
            {view === 'Season View' && (
              <div className={styles.seasonViewGrid}>
                <div className={styles.chartCard}>
                  <h3 className={styles.chartTitle}>Player Card · {selectedSeason}</h3>
                  <p className={styles.chartSub}>{POS_LABELS[currentRow.Pos] || currentRow.Pos} · {currentRow.Squad}</p>
                  <div style={{ marginTop: '1rem' }}>
                    <FIFACard
                      row={currentRow}
                      score={mainScore}
                      band={mainBand}
                      pizzaRow={pizzaRow}
                      marketValue={marketValue}
                    />
                  </div>
                </div>
                <div className={styles.chartCard}>
                  <h3 className={styles.chartTitle}>
                    Score Profile vs. Peers
                    {pizzaLoading && <span style={{ color: 'var(--muted)', fontSize: '0.75rem', fontWeight: 400, marginLeft: 8 }}>Loading…</span>}
                  </h3>
                  <p className={styles.chartSub}>
                    Percentile rank vs. all {POS_LABELS[currentRow.Pos] || currentRow.Pos}s · {selectedSeason}
                    {pizzaRow ? ' · 16 dimensions' : ' · 4 dimensions'}
                  </p>
                  <PizzaRadarChart data={pizzaData} height={400} />
                </div>
              </div>
            )}

            {/* Role Context: scatter + similar players */}
            {view === 'Role Context' && (
              <div className={styles.chartCard}>
                <h3 className={styles.chartTitle}>
                  Role Scatter · {POS_LABELS[currentRow.Pos] || currentRow.Pos}s in {selectedSeason}
                </h3>
                <p className={styles.chartSub}>
                  {pizzaRow
                    ? 'Role-specific key metrics. Dashed lines = median. Dots colored by band.'
                    : 'No pizza data — using score row. Select a season with full data for the scatter.'}
                </p>
                <RoleScatterChart
                  peers={pizzaPeers}
                  playerRow={pizzaRow || currentRow}
                  allRows={allRows}
                  height={400}
                />
                {similarPlayers.length > 0 && (
                  <div style={{ marginTop: '1.75rem' }}>
                    <SimilarPlayers players={similarPlayers} />
                  </div>
                )}
              </div>
            )}

            {/* Age Curve */}
            {view === 'Age Curve' && (
              <div className={styles.chartCard}>
                <h3 className={styles.chartTitle}>Score Development vs. Role Peers</h3>
                <p className={styles.chartSub}>Player score by age vs. median of all {POS_LABELS[currentRow.Pos] || currentRow.Pos}s</p>
                <AgeCurveChart allRows={allRows} player={selectedPlayer} role={currentRow.Pos} height={320} />
              </div>
            )}

            {/* Career Trend */}
            {view === 'Career Trend' && (
              <div className={styles.chartCard}>
                <h3 className={styles.chartTitle}>Career Score Trend</h3>
                <p className={styles.chartSub}>Primary role score across all tracked seasons</p>
                <ScoreTrendLine data={careerData} height={320} />
              </div>
            )}
          </div>

          {/* Sub-score tiles */}
          <div className={styles.subScoreTiles}>
            {[
              { label: 'Offense', val: currentRow.OffScore_abs },
              { label: 'Midfield', val: currentRow.MidScore_abs },
              { label: 'Defense', val: currentRow.DefScore_abs },
            ].map(({ label, val }) => {
              const s = parseFloat(val);
              const band = isNaN(s) ? 'Below Big-5 Level' : scoreToBand(s);
              return (
                <div key={label} className={styles.subTile}>
                  <div className={styles.subTileLabel}>{label}</div>
                  <div className={styles.subTileValue} style={{ color: bandColor(band) }}>
                    {isNaN(s) ? '—' : Math.round(s)}
                  </div>
                  <div className={styles.subTileBar}>
                    <div
                      className={styles.subTileBarFill}
                      style={{ width: `${isNaN(s) ? 0 : Math.min(s / 10, 100)}%`, background: bandColor(band) }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          {/* Stat tiles */}
          <div className={styles.statTiles}>
            <div className={styles.statTile} style={{ borderColor: `${bandColor(mainBand)}44` }}>
              <div className={styles.statValue} style={{ color: bandColor(mainBand) }}>
                {formatScore(mainScore)}
              </div>
              <div className={styles.statLabel}>Player Score</div>
            </div>
            <div className={styles.statTile}>
              <div className={styles.statValue}>
                <ScoreBadge band={mainBand || scoreToBand(mainScore)} />
              </div>
              <div className={styles.statLabel}>Band</div>
            </div>
            <div className={styles.statTile}>
              <div className={styles.statValue}>{formatNineties(currentRow['90s'])}</div>
              <div className={styles.statLabel}>90s Played</div>
            </div>
            <div className={styles.statTile}>
              <div className={styles.statValue}>{formatAge(currentRow.Age)}</div>
              <div className={styles.statLabel}>Age</div>
            </div>
            {marketValue && (
              <div className={styles.statTile}>
                <div className={styles.statValue} style={{ fontSize: '1.1rem' }}>
                  {formatMarketValue(marketValue)}
                </div>
                <div className={styles.statLabel}>Market Value</div>
              </div>
            )}
          </div>

          {/* Season history */}
          {careerData.length > 1 && (
            <div className={styles.historySection}>
              <h3 className={styles.sectionTitle}>Season History</h3>
              <div className={styles.tableWrap}>
                <table>
                  <thead>
                    <tr>
                      <th>Season</th><th>Club</th><th>League</th><th>Pos</th><th>90s</th><th>Score</th><th>Band</th>
                    </tr>
                  </thead>
                  <tbody>
                    {careerData.map((row, i) => (
                      <tr
                        key={i}
                        className={row.Season === selectedSeason ? styles.activeSeason : ''}
                        onClick={() => { setSelectedSeason(row.Season); setView('Season View'); }}
                        style={{ cursor: 'pointer' }}
                      >
                        <td>{row.Season}</td>
                        <td>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                            <ClubCrest clubName={row.Squad} size={18} />
                            {row.Squad}
                          </div>
                        </td>
                        <td style={{ color: 'var(--muted)' }}>{row.Comp}</td>
                        <td><span style={{ color: 'var(--accent)', fontSize: '0.78rem', fontWeight: 600 }}>{row.Pos}</span></td>
                        <td style={{ color: 'var(--muted)' }}>{formatNineties(row['90s'])}</td>
                        <td style={{ fontWeight: 700 }}>{formatScore(row.MainScore)}</td>
                        <td><ScoreBadge band={row.MainBand} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </PageShell>
  );
}
