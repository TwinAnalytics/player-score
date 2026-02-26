import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, ReferenceLine, ReferenceArea,
} from 'recharts';
import { BAND_COLORS } from '../../constants/colors';
import { formatScore } from '../../utils/formatters';

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.[0]) return null;
  const d = payload[0].payload;
  const color = BAND_COLORS[d.MainBand] || '#6B7280';
  return (
    <div style={{
      background: '#161B22', border: '1px solid #21262D', borderRadius: '0.5rem',
      padding: '0.6rem 0.9rem', fontSize: '0.8rem', color: '#F9FAFB', minWidth: 160,
    }}>
      <div style={{ fontWeight: 700, marginBottom: 4 }}>{d.Player}</div>
      <div style={{ color: '#94A3B8', marginBottom: 6 }}>{d.Squad} · {d.Comp}</div>
      <div>Score: <strong style={{ color }}>{formatScore(d.MainScore)}</strong></div>
      <div>Market Value: <strong>€{d.MarketValue_M?.toFixed(1)}M</strong></div>
      <div>Gem Score: <strong style={{ color: '#22c55e' }}>{d.GemScore}/10</strong></div>
    </div>
  );
};

export default function HiddenGemsScatter({ data, height = 440 }) {
  if (!data || data.length === 0) return null;

  const scores = data.map((d) => d.MainScore);
  const mvs = data.map((d) => d.MarketValue_M);

  const medScore = [...scores].sort((a, b) => a - b)[Math.floor(scores.length / 2)] || 500;
  const medMV = [...mvs].sort((a, b) => a - b)[Math.floor(mvs.length / 2)] || 10;

  const xMin = Math.max(0, Math.min(...scores) - 20);
  const xMax = Math.min(1000, Math.max(...scores) + 20);
  const yMin = Math.max(0, Math.min(...mvs) * 0.8);
  const yMax = Math.max(...mvs) * 1.2;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 10, right: 20, bottom: 24, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />

        {/* Quadrant backgrounds */}
        <ReferenceArea x1={xMin} x2={medScore} y1={medMV} y2={yMax} fill="rgba(234,179,8,0.07)" />
        <ReferenceArea x1={medScore} x2={xMax} y1={medMV} y2={yMax} fill="rgba(59,130,246,0.07)" />
        <ReferenceArea x1={xMin} x2={medScore} y1={yMin} y2={medMV} fill="rgba(107,114,128,0.07)" />
        <ReferenceArea x1={medScore} x2={xMax} y1={yMin} y2={medMV} fill="rgba(34,197,94,0.10)" />

        {/* Quadrant labels */}
        <ReferenceLine x={medScore} stroke="#374151" strokeDasharray="4 3" label={{ value: 'Median Score', position: 'top', fill: '#6B7280', fontSize: 9 }} />
        <ReferenceLine y={medMV} stroke="#374151" strokeDasharray="4 3" label={{ value: 'Median MV', position: 'right', fill: '#6B7280', fontSize: 9 }} />

        <XAxis
          type="number" dataKey="MainScore" name="Score"
          domain={[xMin, xMax]}
          tick={{ fill: '#94A3B8', fontSize: 11 }}
          tickLine={false} axisLine={{ stroke: '#374151' }}
          label={{ value: 'Player Score', position: 'insideBottom', offset: -14, fill: '#94A3B8', fontSize: 11 }}
        />
        <YAxis
          type="number" dataKey="MarketValue_M" name="Market Value (€M)"
          domain={[yMin, yMax]}
          tick={{ fill: '#94A3B8', fontSize: 11 }}
          tickLine={false} axisLine={false}
          tickFormatter={(v) => `€${v.toFixed(0)}M`}
          label={{ value: 'Market Value (€M)', angle: -90, position: 'insideLeft', fill: '#94A3B8', fontSize: 11 }}
        />

        <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '4 4', stroke: '#374151' }} />

        <Scatter data={data}>
          {data.map((entry, i) => (
            <Cell key={i} fill={BAND_COLORS[entry.MainBand] || '#6B7280'} fillOpacity={0.88} />
          ))}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}
