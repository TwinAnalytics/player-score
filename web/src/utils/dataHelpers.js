import { getPrimaryScore, scoreToBand } from '../constants/scoring';

export function getLatestSeason(rows) {
  if (!rows || rows.length === 0) return null;
  const seasons = [...new Set(rows.map((r) => r.Season).filter(Boolean))];
  return seasons.sort().at(-1);
}

export function getSeasons(rows) {
  if (!rows) return [];
  return [...new Set(rows.map((r) => r.Season).filter(Boolean))].sort().reverse();
}

export function getComps(rows) {
  if (!rows) return [];
  return [...new Set(rows.map((r) => r.Comp).filter(Boolean))].sort();
}

export function getClubsForComp(rows, comp) {
  if (!rows) return [];
  const filtered = comp ? rows.filter((r) => r.Comp === comp) : rows;
  return [...new Set(filtered.map((r) => r.Squad).filter(Boolean))].sort();
}

export function enrichWithPrimaryScore(rows) {
  return rows.map((row) => {
    const { score, band } = getPrimaryScore(row);
    return {
      ...row,
      MainScore: score,
      MainBand: band || (score !== null ? scoreToBand(score) : 'Below Big-5 Level'),
    };
  });
}

export function filterRows(rows, { season, comp, club, positions, minNineties = 5 }) {
  let result = rows;
  if (season) result = result.filter((r) => r.Season === season);
  if (comp) result = result.filter((r) => r.Comp === comp);
  if (club) result = result.filter((r) => r.Squad === club);
  if (positions && positions.length > 0) {
    result = result.filter((r) => positions.includes(r.Pos));
  }
  result = result.filter((r) => parseFloat(r['90s'] || r['90s_x'] || 0) >= minNineties);
  return result;
}

export function buildPlayerMap(rows) {
  const map = new Map();
  for (const row of rows) {
    const name = row.Player;
    if (!name) continue;
    if (!map.has(name)) map.set(name, []);
    map.get(name).push(row);
  }
  return map;
}

export function computePercentileRank(value, allValues) {
  if (!allValues || allValues.length === 0) return 0;
  const sorted = [...allValues].filter((v) => !isNaN(v)).sort((a, b) => a - b);
  const rank = sorted.filter((v) => v <= value).length;
  return Math.round((rank / sorted.length) * 100);
}

export function computeRadarData(playerRow, peers) {
  const dims = [
    { key: 'OffScore_abs', label: 'Offense' },
    { key: 'MidScore_abs', label: 'Midfield' },
    { key: 'DefScore_abs', label: 'Defense' },
    { key: '90s', label: 'Minutes' },
  ];

  return dims.map(({ key, label }) => {
    const playerVal = parseFloat(playerRow[key] || 0);
    const peerVals = peers.map((r) => parseFloat(r[key] || 0));
    const pct = computePercentileRank(playerVal, peerVals);
    return { dimension: label, percentile: pct };
  });
}
