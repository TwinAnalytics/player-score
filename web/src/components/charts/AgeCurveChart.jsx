import { useMemo } from 'react';
import {
  ComposedChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Legend,
} from 'recharts';
import { BAND_COLORS } from '../../constants/colors';
import { enrichWithPrimaryScore } from '../../utils/dataHelpers';

const THRESHOLDS = [
  { value: 750, color: BAND_COLORS['World Class'] },
  { value: 400, color: BAND_COLORS['Top Starter'] },
];

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload) return null;
  return (
    <div style={{
      background: '#161B22', border: '1px solid #21262D', borderRadius: '0.5rem',
      padding: '0.6rem 0.9rem', fontSize: '0.8rem', color: '#F9FAFB',
    }}>
      <div style={{ color: '#94A3B8', marginBottom: 4 }}>Age {label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color }}>
          {p.name}: <strong>{p.value != null ? Math.round(p.value) : '—'}</strong>
        </div>
      ))}
    </div>
  );
};

export default function AgeCurveChart({ allRows, player, role, height = 280 }) {
  const { playerLine, peerLine, xDomain } = useMemo(() => {
    if (!allRows || !player || !role) return { playerLine: [], peerLine: [], xDomain: [18, 38] };

    const enriched = enrichWithPrimaryScore(allRows);

    // Peer median by age
    const rolePeers = enriched.filter((r) => r.Pos === role && r.MainScore !== null && !isNaN(r.MainScore));
    const ageMap = new Map();
    for (const r of rolePeers) {
      const age = Math.round(parseFloat(r.Age));
      if (isNaN(age)) continue;
      if (!ageMap.has(age)) ageMap.set(age, []);
      ageMap.get(age).push(parseFloat(r.MainScore));
    }
    const peerLine = [...ageMap.entries()]
      .filter(([a]) => a >= 16 && a <= 40)
      .sort(([a], [b]) => a - b)
      .map(([age, scores]) => ({
        age,
        median: scores.sort((a, b) => a - b)[Math.floor(scores.length / 2)],
      }));

    // Player trajectory by age
    const playerRows = enriched
      .filter((r) => r.Player === player && r.MainScore !== null && !isNaN(r.MainScore))
      .map((r) => ({ age: Math.round(parseFloat(r.Age)), score: parseFloat(r.MainScore), season: r.Season }))
      .filter((r) => !isNaN(r.age))
      .sort((a, b) => a.age - b.age);

    const allAges = [...new Set([...peerLine.map((d) => d.age), ...playerRows.map((d) => d.age)])].sort((a, b) => a - b);
    const minAge = Math.max(16, Math.min(...allAges) - 1);
    const maxAge = Math.min(40, Math.max(...allAges) + 1);

    // Merge into single array for ComposedChart
    const ageSet = new Set([...peerLine.map((d) => d.age), ...playerRows.map((d) => d.age)]);
    const combined = [...ageSet].sort((a, b) => a - b).map((age) => {
      const peer = peerLine.find((d) => d.age === age);
      const pl = playerRows.find((d) => d.age === age);
      return { age, median: peer?.median ?? null, score: pl?.score ?? null };
    });

    return { playerLine: combined, peerLine, xDomain: [minAge, maxAge] };
  }, [allRows, player, role]);

  if (!playerLine.length) return null;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={playerLine} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis
          dataKey="age"
          type="number"
          domain={xDomain}
          tick={{ fill: '#94A3B8', fontSize: 11 }}
          tickLine={false}
          axisLine={{ stroke: '#374151' }}
          label={{ value: 'Age', position: 'insideBottom', offset: -10, fill: '#94A3B8', fontSize: 11 }}
        />
        <YAxis
          domain={[0, 1000]}
          tick={{ fill: '#94A3B8', fontSize: 11 }}
          tickLine={false}
          axisLine={false}
        />
        <Tooltip content={<CustomTooltip />} />
        {THRESHOLDS.map(({ value, color }) => (
          <ReferenceLine key={value} y={value} stroke={color} strokeDasharray="4 3" strokeOpacity={0.35} />
        ))}
        <Line
          type="monotone" dataKey="median" name="Role peers (median)"
          stroke="#6B7280" strokeDasharray="6 3" strokeWidth={1.5}
          dot={false} connectNulls
        />
        <Line
          type="monotone" dataKey="score" name={player?.split(' ').pop() || 'Player'}
          stroke="#00B8A9" strokeWidth={2.5}
          dot={{ fill: '#00B8A9', r: 4, strokeWidth: 0 }}
          activeDot={{ r: 6, fill: '#80F5E3' }}
          connectNulls
        />
        <Legend
          wrapperStyle={{ fontSize: '0.75rem', color: '#94A3B8', paddingTop: 8 }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
