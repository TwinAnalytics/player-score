import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts';
import { BAND_COLORS } from '../../constants/colors';

const THRESHOLDS = [
  { value: 900, label: 'Exceptional', color: BAND_COLORS['Exceptional'] },
  { value: 750, label: 'World Class', color: BAND_COLORS['World Class'] },
  { value: 400, label: 'Top Starter', color: BAND_COLORS['Top Starter'] },
  { value: 200, label: 'Solid Squad', color: BAND_COLORS['Solid Squad Player'] },
];

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.[0]) return null;
  return (
    <div style={{
      background: '#161B22',
      border: '1px solid #21262D',
      borderRadius: '0.5rem',
      padding: '0.6rem 0.9rem',
      fontSize: '0.8rem',
      color: '#F9FAFB',
    }}>
      <div style={{ color: '#94A3B8', marginBottom: 4 }}>{label}</div>
      <div>Score: <strong style={{ color: '#00B8A9' }}>{Math.round(payload[0].value)}</strong></div>
    </div>
  );
};

export default function ScoreTrendLine({ data, height = 280 }) {
  if (!data || data.length === 0) return null;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis
          dataKey="Season"
          tick={{ fill: '#94A3B8', fontSize: 11 }}
          tickLine={false}
          axisLine={{ stroke: '#374151' }}
        />
        <YAxis
          domain={[0, 1000]}
          tick={{ fill: '#94A3B8', fontSize: 11 }}
          tickLine={false}
          axisLine={false}
        />
        <Tooltip content={<CustomTooltip />} />
        {THRESHOLDS.map(({ value, label, color }) => (
          <ReferenceLine
            key={value}
            y={value}
            stroke={color}
            strokeDasharray="4 4"
            strokeOpacity={0.4}
            label={{ value: label, position: 'right', fill: color, fontSize: 9, opacity: 0.7 }}
          />
        ))}
        <Line
          type="monotone"
          dataKey="MainScore"
          stroke="#00B8A9"
          strokeWidth={2.5}
          dot={{ fill: '#00B8A9', r: 4, strokeWidth: 0 }}
          activeDot={{ r: 6, fill: '#80F5E3' }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
