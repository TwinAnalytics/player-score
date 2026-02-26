export function formatScore(score) {
  const s = parseFloat(score);
  if (isNaN(s)) return '—';
  return Math.round(s).toString();
}

export function formatNineties(val) {
  const n = parseFloat(val);
  if (isNaN(n)) return '—';
  return n.toFixed(1);
}

export function formatAge(val) {
  const n = parseFloat(val);
  if (isNaN(n)) return '—';
  return Math.round(n).toString();
}

export function formatMarketValue(val) {
  const n = parseFloat(val);
  if (isNaN(n) || n === 0) return '—';
  if (n >= 1_000_000) return `€${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `€${(n / 1_000).toFixed(0)}K`;
  return `€${n}`;
}

export function normalizeSeason(s) {
  if (!s) return s;
  return s.replace('_', '-');
}
