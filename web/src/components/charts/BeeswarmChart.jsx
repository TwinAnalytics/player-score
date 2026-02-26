import { useMemo } from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, LabelList,
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
      padding: '0.6rem 0.9rem', fontSize: '0.8rem', color: '#F9FAFB',
    }}>
      <div style={{ fontWeight: 700, marginBottom: 2 }}>{d.Player}</div>
      <div style={{ color: '#94A3B8', marginBottom: 4 }}>{d.Squad} · {d.Comp}</div>
      <div>Score: <strong style={{ color }}>{formatScore(d.MainScore)}</strong></div>
      <div style={{ color: '#94A3B8' }}>Age: {Math.round(parseFloat(d.Age))} · {d.Pos}</div>
    </div>
  );
};

export default function BeeswarmChart({ data, topN = 10, onDotClick, height = 420 }) {
  if (!data || data.length === 0) return null;

  const plotData = useMemo(() => {
    const topKeys = new Set(data.slice(0, topN).map((d) => d.Player + '|' + d.Squad));
    return data.map((d, i) => ({
      ...d,
      x: parseFloat(d.MainScore) || 0,
      y: (parseFloat(d.Age) || 25) + (((i * 2654435761) % 100) / 100 - 0.5) * 0.8,
      isTop: topKeys.has(d.Player + '|' + d.Squad),
    }));
  }, [data, topN]);

  const regular = plotData.filter((d) => !d.isTop);
  const highlighted = plotData.filter((d) => d.isTop);

  const ages = plotData.map((d) => d.y).filter((v) => !isNaN(v));
  const yMin = Math.floor(Math.min(...ages)) - 1;
  const yMax = Math.ceil(Math.max(...ages)) + 1;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis
          type="number" dataKey="x" name="Score"
          domain={[0, 1000]}
          tick={{ fill: '#94A3B8', fontSize: 11 }}
          tickLine={false} axisLine={{ stroke: '#374151' }}
          label={{ value: 'Player Score', position: 'insideBottom', offset: -12, fill: '#94A3B8', fontSize: 11 }}
        />
        <YAxis
          type="number" dataKey="y" name="Age"
          domain={[yMin, yMax]}
          tick={{ fill: '#94A3B8', fontSize: 11 }}
          tickLine={false} axisLine={false}
          tickFormatter={(v) => Math.round(v)}
          label={{ value: 'Age', angle: -90, position: 'insideLeft', fill: '#94A3B8', fontSize: 11 }}
        />
        <Tooltip content={<CustomTooltip />} cursor={false} />

        {/* Regular dots */}
        <Scatter data={regular} onClick={(d) => onDotClick?.(d)}>
          {regular.map((d, i) => (
            <Cell key={i} fill={BAND_COLORS[d.MainBand] || '#6B7280'} fillOpacity={0.55} />
          ))}
        </Scatter>

        {/* Top-N highlighted */}
        <Scatter data={highlighted} onClick={(d) => onDotClick?.(d)}>
          {highlighted.map((d, i) => (
            <Cell key={i} fill={BAND_COLORS[d.MainBand] || '#00B8A9'} fillOpacity={0.95} />
          ))}
          <LabelList
            dataKey="Player"
            position="right"
            style={{ fill: '#E5E7EB', fontSize: 9 }}
            formatter={(v) => v?.split(' ').pop()}
          />
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}
