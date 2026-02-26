import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts';
import { LEAGUE_COLORS } from '../../constants/colors';
import { formatScore } from '../../utils/formatters';

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.[0]) return null;
  const d = payload[0].payload;
  return (
    <div style={{
      background: '#161B22',
      border: '1px solid #21262D',
      borderRadius: '0.5rem',
      padding: '0.6rem 0.9rem',
      fontSize: '0.8rem',
      color: '#F9FAFB',
      minWidth: 160,
    }}>
      <div style={{ fontWeight: 700, marginBottom: 4 }}>{d.Squad}</div>
      <div style={{ color: '#94A3B8', marginBottom: 6 }}>{d.Comp}</div>
      <div>Overall: <strong style={{ color: '#00B8A9' }}>{formatScore(d.x)}</strong></div>
      <div>Offense: <strong>{formatScore(d.y)}</strong></div>
      <div>Midfield: <strong>{formatScore(d.OffScore_squad || d.MidScore_squad)}</strong></div>
    </div>
  );
};

export default function ScatterPlot({ data, xKey = 'x', yKey = 'y', height = 360 }) {
  if (!data || data.length === 0) return null;

  const xVals = data.map((d) => parseFloat(d[xKey])).filter((v) => !isNaN(v));
  const yVals = data.map((d) => parseFloat(d[yKey])).filter((v) => !isNaN(v));
  const xMin = Math.max(0, Math.min(...xVals) - 30);
  const xMax = Math.min(1000, Math.max(...xVals) + 30);
  const yMin = Math.max(0, Math.min(...yVals) - 30);
  const yMax = Math.min(1000, Math.max(...yVals) + 30);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis
          type="number"
          dataKey={xKey}
          name="Overall Score"
          domain={[xMin, xMax]}
          tick={{ fill: '#94A3B8', fontSize: 11 }}
          tickLine={false}
          axisLine={{ stroke: '#374151' }}
          label={{ value: 'Overall Score', position: 'insideBottom', offset: -12, fill: '#94A3B8', fontSize: 11 }}
        />
        <YAxis
          type="number"
          dataKey={yKey}
          name="Offense Score"
          domain={[yMin, yMax]}
          tick={{ fill: '#94A3B8', fontSize: 11 }}
          tickLine={false}
          axisLine={false}
          label={{ value: 'Offense Score', angle: -90, position: 'insideLeft', fill: '#94A3B8', fontSize: 11 }}
        />
        <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '4 4', stroke: '#374151' }} />
        <Scatter data={data.map((d) => ({ ...d, x: parseFloat(d[xKey]), y: parseFloat(d[yKey]) }))}>
          {data.map((entry, i) => (
            <Cell
              key={i}
              fill={LEAGUE_COLORS[entry.Comp] || '#00B8A9'}
              fillOpacity={0.85}
            />
          ))}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}
