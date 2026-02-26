import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, LabelList,
} from 'recharts';
import { BAND_COLORS } from '../../constants/colors';
import { formatScore } from '../../utils/formatters';

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.[0]) return null;
  const d = payload[0].payload;
  const color = BAND_COLORS[d.MainBand] || '#6B7280';
  return (
    <div style={{
      background: '#161B22',
      border: '1px solid #21262D',
      borderRadius: '0.5rem',
      padding: '0.6rem 0.9rem',
      fontSize: '0.8rem',
      color: '#F9FAFB',
    }}>
      <div style={{ fontWeight: 700, marginBottom: 4 }}>{d.Player}</div>
      <div style={{ color: '#94A3B8' }}>{d.Squad} · {d.Comp}</div>
      <div style={{ marginTop: 6 }}>
        Score: <strong style={{ color }}>{formatScore(d.MainScore)}</strong>
      </div>
      <div style={{ color }}>Band: {d.MainBand}</div>
    </div>
  );
};

export default function ScoreBarChart({ data, height = 400, onBarClick }) {
  if (!data || data.length === 0) return null;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={data}
        layout="vertical"
        margin={{ top: 4, right: 60, left: 4, bottom: 4 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" horizontal={false} />
        <XAxis
          type="number"
          domain={[0, 1000]}
          tick={{ fill: '#94A3B8', fontSize: 11 }}
          tickLine={false}
          axisLine={{ stroke: '#374151' }}
        />
        <YAxis
          type="category"
          dataKey="Player"
          width={130}
          tick={{ fill: '#F9FAFB', fontSize: 12 }}
          tickLine={false}
          axisLine={false}
        />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
        <Bar dataKey="MainScore" radius={[0, 4, 4, 0]} onClick={onBarClick}>
          {data.map((entry, i) => (
            <Cell
              key={i}
              fill={BAND_COLORS[entry.MainBand] || '#6B7280'}
              cursor={onBarClick ? 'pointer' : 'default'}
            />
          ))}
          <LabelList
            dataKey="MainScore"
            position="right"
            formatter={formatScore}
            style={{ fill: '#94A3B8', fontSize: 11 }}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
