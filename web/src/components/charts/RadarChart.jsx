import {
  RadarChart as RechartsRadar, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip,
} from 'recharts';

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.[0]) return null;
  const d = payload[0].payload;
  return (
    <div style={{
      background: '#161B22',
      border: '1px solid #21262D',
      borderRadius: '0.5rem',
      padding: '0.5rem 0.8rem',
      fontSize: '0.8rem',
      color: '#F9FAFB',
    }}>
      <div style={{ fontWeight: 600 }}>{d.dimension}</div>
      <div style={{ color: '#00B8A9' }}>Percentile: {d.percentile}</div>
    </div>
  );
};

export default function PlayerRadarChart({ data, height = 320 }) {
  if (!data || data.length === 0) return null;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <RechartsRadar data={data} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
        <PolarGrid stroke="#374151" />
        <PolarAngleAxis
          dataKey="dimension"
          tick={{ fill: '#94A3B8', fontSize: 12 }}
        />
        <PolarRadiusAxis
          angle={90}
          domain={[0, 100]}
          tick={{ fill: '#374151', fontSize: 9 }}
          tickCount={4}
          axisLine={false}
        />
        <Radar
          dataKey="percentile"
          stroke="#00B8A9"
          fill="#00B8A9"
          fillOpacity={0.25}
          strokeWidth={2}
          dot={{ fill: '#00B8A9', r: 3 }}
        />
        <Tooltip content={<CustomTooltip />} />
      </RechartsRadar>
    </ResponsiveContainer>
  );
}
