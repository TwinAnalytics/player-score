import { PIZZA_DIMS } from '../context/PizzaDataContext';
import { computePercentileRank } from './dataHelpers';

/**
 * Compute 16-dimension pizza percentiles for a player row vs. a list of peers.
 * Returns array of { dimension, label, group, percentile } objects.
 */
export function computePizzaPercentiles(playerRow, peers) {
  if (!playerRow || !peers || peers.length === 0) return [];

  return PIZZA_DIMS.map(({ key, label, group }) => {
    const playerVal = parseFloat(playerRow[key]);
    if (isNaN(playerVal)) {
      return { dimension: label, label, group, percentile: 0 };
    }
    const peerVals = peers
      .map((r) => parseFloat(r[key]))
      .filter((v) => !isNaN(v));
    const percentile = computePercentileRank(playerVal, peerVals);
    return { dimension: label, label, group, percentile };
  });
}

/**
 * Compute Euclidean distance similarity between a target player and all peers.
 * Returns top N most similar players by pizza dimensions.
 */
export function findSimilarByPizza(targetRow, peers, n = 5) {
  if (!targetRow || !peers || peers.length === 0) return [];

  const dims = PIZZA_DIMS.map((d) => d.key);

  // Get all peer values per dimension for percentile normalization
  const peerPercentiles = peers.map((peer) => {
    return dims.map((key) => {
      const val = parseFloat(peer[key]);
      const allVals = peers.map((r) => parseFloat(r[key])).filter((v) => !isNaN(v));
      return isNaN(val) ? 50 : computePercentileRank(val, allVals);
    });
  });

  // Compute target percentiles
  const targetPcts = dims.map((key) => {
    const val = parseFloat(targetRow[key]);
    const allVals = peers.map((r) => parseFloat(r[key])).filter((v) => !isNaN(v));
    return isNaN(val) ? 50 : computePercentileRank(val, allVals);
  });

  // Euclidean distance
  const distances = peers.map((peer, i) => {
    if (peer.Player === targetRow.Player) return { peer, dist: Infinity };
    const pcts = peerPercentiles[i];
    const dist = Math.sqrt(
      targetPcts.reduce((sum, tp, j) => sum + (tp - pcts[j]) ** 2, 0)
    );
    return { peer, dist };
  });

  return distances
    .filter((d) => isFinite(d.dist))
    .sort((a, b) => a.dist - b.dist)
    .slice(0, n)
    .map((d) => d.peer);
}

/**
 * Generate auto scouting text for a player.
 */
export function generateScoutingText(row, score, band, marketValueEUR) {
  const name = row.Player?.split(' ').pop() || row.Player || 'The player';
  const pos = row.Pos || '';
  const age = Math.round(parseFloat(row.Age)) || '?';
  const squad = row.Squad || '';
  const league = row.Comp || '';
  const season = row.Season || '';
  const nineties = parseFloat(row['90s'] || 0).toFixed(1);

  const offScore = parseFloat(row.OffScore_abs) || 0;
  const midScore = parseFloat(row.MidScore_abs) || 0;
  const defScore = parseFloat(row.DefScore_abs) || 0;
  const maxDim = Math.max(offScore, midScore, defScore);
  const strongestArea =
    maxDim === offScore ? 'offensive output' :
    maxDim === midScore ? 'midfield contribution' :
    'defensive work';

  const posLabel = { FW: 'forward', Off_MF: 'attacking midfielder', MF: 'central midfielder', Def_MF: 'defensive midfielder', DF: 'defender' }[pos] || pos;

  const bandDesc = {
    Exceptional: 'an elite performer, ranking among the best in Europe',
    'World Class': 'a consistently world-class performer',
    'Top Starter': 'a reliable starter at Big-5 level',
    'Solid Squad Player': 'a solid squad contributor',
    'Below Big-5 Level': 'currently below the Big-5 average',
  }[band] || 'a notable performer';

  const mvStr = marketValueEUR
    ? ` Valued at ${marketValueEUR >= 1_000_000 ? `€${(marketValueEUR / 1_000_000).toFixed(0)}M` : `€${(marketValueEUR / 1_000).toFixed(0)}K`}.`
    : '';

  return `${name} is a ${age}-year-old ${posLabel} playing for ${squad} in the ${league}. ` +
    `In the ${season} season, they achieved a PlayerScore of ${Math.round(score)} (${band}) — ${bandDesc}. ` +
    `Their standout area is their ${strongestArea}, covering ${nineties} 90s played.${mvStr}`;
}
