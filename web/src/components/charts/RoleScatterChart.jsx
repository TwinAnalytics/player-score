import { useMemo } from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Label, ReferenceLine,
} from 'recharts';
import { BAND_COLORS } from '../../constants/colors';
import { enrichWithPrimaryScore } from '../../utils/dataHelpers';

const ROLE_AXES = {
  FW:     { x: 'xG_Per90',   xL: 'xG per 90',             y: 'Gls_Per90',  yL: 'Goals per 90' },
  Off_MF: { x: 'xAG_Per90',  xL: 'xAG per 90',            y: 'Ast_Per90',  yL: 'Assists per 90' },
  MF:     { x: 'xAG_Per90',  xL: 'xAG per 90',            y: 'Ast_Per90',  yL: 'Assists per 90' },
  Def_MF: { x: 'Int_Per90',  xL: 'Interceptions per 90',  y: 'TklW_Per90', yL: 'Tackles Won per 90' },
  DF:     { x: 'Int_Per90',  xL: 'Interceptions per 90',  y: 'TklW_Per90', yL: 'Tackles Won per 90' },
};

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;
  return (
    <div style={{
      background: '#161B22', border: '1px solid #374151', borderRadius: 8,
      padding: '8px 12px', fontSize: 12, color: '#F9FAFB', lineHeight: 1.5,
    }}>
      <div style={{ fontWeight: 700 }}>{d.Player}</div>
      <div style={{ color: '#94A3B8' }}>{d.Squad}</div>
      <div style={{ marginTop: 4 }}>
        {payload[0].name}: <strong style={{ color: '#00B8A9' }}>{Number(payload[0].value).toFixed(3)}</strong>
      </div>
      <div>
        {payload[1]?.name}: <strong style={{ color: '#00B8A9' }}>{Number(payload[1]?.value).toFixed(3)}</strong>
      </div>
    </div>
  );
};

const CustomDot = (props) => {
  const { cx, cy, payload, isPlayer } = props;
  if (isPlayer) {
    return (
      <g>
        <circle cx={cx} cy={cy} r={9} fill="#00B8A9" stroke="#F9FAFB" strokeWidth={1.5} />
        <text
          x={cx + 13} y={cy - 8}
          fill="#F9FAFB" fontSize={10} fontWeight={700}
          style={{ fontFamily: 'system-ui, sans-serif', textShadow: '0 1px 3px rgba(0,0,0,0.8)' }}
        >
          {payload.Player?.split(' ').pop()}
        </text>
      </g>
    );
  }
  const color = BAND_COLORS[payload.MainBand] || '#374151';
  return <circle cx={cx} cy={cy} r={4} fill={color} fillOpacity={0.5} stroke="none" />;
};

export default function RoleScatterChart({ peers, playerRow, allRows, height = 400 }) {
  const axes = ROLE_AXES[playerRow?.Pos] || ROLE_AXES['MF'];

  const { peerData, playerPoint, medX, medY } = useMemo(() => {
    if (!peers || !playerRow) return { peerData: [], playerPoint: null, medX: 0, medY: 0 };

    // Enrich peers with band info by joining with allRows
    const enrichedPeers = peers
      .map((p) => {
        const scoreRow = allRows?.find((r) => r.Player === p.Player && r.Season === p.Season && r.Pos === p.Pos);
        const band = scoreRow ? enrichWithPrimaryScore([scoreRow])[0]?.MainBand : undefined;
        const xVal = parseFloat(p[axes.x]);
        const yVal = parseFloat(p[axes.y]);
        if (isNaN(xVal) || isNaN(yVal)) return null;
        return { ...p, xVal, yVal, MainBand: band };
      })
      .filter(Boolean);

    const pxVal = parseFloat(playerRow[axes.x]);
    const pyVal = parseFloat(playerRow[axes.y]);
    const playerPoint = (!isNaN(pxVal) && !isNaN(pyVal))
      ? { ...playerRow, xVal: pxVal, yVal: pyVal }
      : null;

    const allX = enrichedPeers.map((p) => p.xVal).filter(Boolean).sort((a, b) => a - b);
    const allY = enrichedPeers.map((p) => p.yVal).filter(Boolean).sort((a, b) => a - b);
    const medX = allX[Math.floor(allX.length / 2)] || 0;
    const medY = allY[Math.floor(allY.length / 2)] || 0;

    return { peerData: enrichedPeers, playerPoint, medX, medY };
  }, [peers, playerRow, axes, allRows]);

  if (!playerPoint) return (
    <div style={{ color: 'var(--muted)', textAlign: 'center', padding: '2rem' }}>
      No data available for this player/season.
    </div>
  );

  return (
    <div>
      <ResponsiveContainer width="100%" height={height}>
        <ScatterChart margin={{ top: 10, right: 30, bottom: 40, left: 30 }}>
          <CartesianGrid stroke="#374151" strokeOpacity={0.4} strokeDasharray="4 4" />
          <XAxis
            dataKey="xVal"
            type="number"
            name={axes.xL}
            stroke="#4b5563"
            tick={{ fill: '#94A3B8', fontSize: 11 }}
            domain={['auto', 'auto']}
          >
            <Label value={axes.xL} offset={-8} position="insideBottom" fill="#94A3B8" fontSize={11} />
          </XAxis>
          <YAxis
            dataKey="yVal"
            type="number"
            name={axes.yL}
            stroke="#4b5563"
            tick={{ fill: '#94A3B8', fontSize: 11 }}
            domain={['auto', 'auto']}
          >
            <Label value={axes.yL} angle={-90} position="insideLeft" fill="#94A3B8" fontSize={11} offset={10} />
          </YAxis>

          {/* Median reference lines */}
          <ReferenceLine x={medX} stroke="#4b5563" strokeDasharray="5 4" strokeOpacity={0.7} />
          <ReferenceLine y={medY} stroke="#4b5563" strokeDasharray="5 4" strokeOpacity={0.7} />

          <Tooltip content={<CustomTooltip />} />

          {/* Peer dots */}
          <Scatter
            data={peerData}
            shape={(props) => <CustomDot {...props} isPlayer={false} />}
          />

          {/* Player dot */}
          <Scatter
            data={[playerPoint]}
            shape={(props) => <CustomDot {...props} isPlayer={true} />}
          />
        </ScatterChart>
      </ResponsiveContainer>

      {/* Band color legend */}
      <div style={{ display: 'flex', justifyContent: 'center', flexWrap: 'wrap', gap: '0.75rem', marginTop: 4 }}>
        {Object.entries(BAND_COLORS).map(([band, color]) => (
          <div key={band} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '0.7rem', color: '#94A3B8' }}>
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: color, opacity: 0.75, flexShrink: 0 }} />
            {band}
          </div>
        ))}
        <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '0.7rem', color: '#00B8A9', fontWeight: 700 }}>
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#00B8A9', flexShrink: 0 }} />
          Selected Player
        </div>
      </div>
    </div>
  );
}
