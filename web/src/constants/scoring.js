import { BAND_COLORS } from './colors';

export const BAND_THRESHOLDS = [
  { min: 900, label: 'Exceptional', description: 'Elite — top 1% globally' },
  { min: 750, label: 'World Class', description: 'Consistently excellent performer' },
  { min: 400, label: 'Top Starter', description: 'Regular starter at Big-5 level' },
  { min: 200, label: 'Solid Squad Player', description: 'Rotation-quality contributor' },
  { min: 0, label: 'Below Big-5 Level', description: 'Below Big-5 average' },
];

export function getPrimaryScore(row) {
  const pos = row.Pos;
  if (pos === 'FW' || pos === 'Off_MF') {
    return { score: parseFloat(row.OffScore_abs), band: row.OffBand };
  }
  if (pos === 'MF') {
    return { score: parseFloat(row.MidScore_abs), band: row.MidBand };
  }
  if (pos === 'DF' || pos === 'Def_MF') {
    return { score: parseFloat(row.DefScore_abs), band: row.DefBand };
  }
  return { score: null, band: null };
}

export function scoreToBand(score) {
  const s = parseFloat(score);
  if (isNaN(s)) return 'Below Big-5 Level';
  if (s >= 900) return 'Exceptional';
  if (s >= 750) return 'World Class';
  if (s >= 400) return 'Top Starter';
  if (s >= 200) return 'Solid Squad Player';
  return 'Below Big-5 Level';
}

export function bandColor(band) {
  return BAND_COLORS[band] || BAND_COLORS['Below Big-5 Level'];
}

export const POS_LABELS = {
  FW: 'Forward',
  Off_MF: 'Attacking Mid',
  MF: 'Midfielder',
  Def_MF: 'Defensive Mid',
  DF: 'Defender',
};
