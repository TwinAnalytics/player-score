import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, LabelList,
} from 'recharts';
import { BAND_COLORS, BAND_ORDER } from '../../constants/colors';

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.[0]) return null;
  const d = payload[0].payload;
  const color = BAND_COLORS[d.band] || '#6B7280';
  return (
    <div style={{
      background: '#161B22', border: '1px solid #21262D', borderRadius: '0.5rem',
      padding: '0.5rem 0.8rem', fontSize: '0.8rem', color: '#F9FAFB',
    }}>
      <div style={{ color, fontWeight: 700 }}>{d.band}</div>
      <div>{d.count} players</div>
    </div>
  );
};

export default function BandHistogram({ data, height = 280, title }) {
  // data: array of player rows with MainBand
  const counts = BAND_ORDER.map((band) => ({
    band,
    shortLabel: band.replace(' Level', '').replace('Solid Squad Player', 'Solid Squad'),
    count: data.filter((r) => r.MainBand === band).length,
  }));

  return (
    <div>
      {title && (
        <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#F9FAFB', marginBottom: 8 }}>
          {title}
        </div>
      )}
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={counts} margin={{ top: 24, right: 12, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
          <XAxis
            dataKey="shortLabel"
            tick={{ fill: '#94A3B8', fontSize: 10 }}
            tickLine={false}
            axisLine={{ stroke: '#374151' }}
          />
          <YAxis
            tick={{ fill: '#94A3B8', fontSize: 11 }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
          <Bar dataKey="count" radius={[4, 4, 0, 0]}>
            {counts.map((entry, i) => (
              <Cell key={i} fill={BAND_COLORS[entry.band] || '#6B7280'} />
            ))}
            <LabelList
              dataKey="count"
              position="top"
              style={{ fill: '#94A3B8', fontSize: 11, fontWeight: 700 }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
