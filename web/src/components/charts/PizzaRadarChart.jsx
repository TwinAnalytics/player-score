import { PIZZA_GROUP_COLORS } from '../../context/PizzaDataContext';

const COLORS = PIZZA_GROUP_COLORS;
const GROUP_ORDER = ['Possession', 'Attacking', 'Defending'];
const BG = '#0D1117';
const GRID_LEVELS = [20, 40, 60, 80, 100];

function splitLabel(text, maxLen = 9) {
  if (text.length <= maxLen) return [text];
  const sp = text.lastIndexOf(' ', maxLen);
  if (sp > 0) return [text.slice(0, sp), text.slice(sp + 1)];
  return [text.slice(0, maxLen), text.slice(maxLen)];
}

export default function PizzaRadarChart({ data, height = 420, title }) {
  if (!data || data.length === 0) return null;

  const size = Math.min(height, 480);
  const cx = size / 2;
  const cy = size / 2;
  const maxR = size * 0.315;
  const labelR = maxR + 28;
  const N = data.length;
  const sliceAngle = (2 * Math.PI) / N;

  const polarXY = (angle, r) => ({
    x: cx + r * Math.cos(angle),
    y: cy + r * Math.sin(angle),
  });

  const slicePath = (i, r) => {
    if (r < 0.5) return `M ${cx} ${cy} Z`;
    const a0 = -Math.PI / 2 + i * sliceAngle;
    const a1 = -Math.PI / 2 + (i + 1) * sliceAngle;
    const large = a1 - a0 > Math.PI ? 1 : 0;
    const s = polarXY(a0, r);
    const e = polarXY(a1, r);
    return `M ${cx} ${cy} L ${s.x.toFixed(2)} ${s.y.toFixed(2)} A ${r.toFixed(2)} ${r.toFixed(2)} 0 ${large} 1 ${e.x.toFixed(2)} ${e.y.toFixed(2)} Z`;
  };

  const tooltipStyle = {
    background: '#161B22',
    border: '1px solid #374151',
    borderRadius: 6,
    padding: '6px 10px',
    fontSize: 12,
    color: '#F9FAFB',
    pointerEvents: 'none',
    position: 'absolute',
  };

  return (
    <div>
      {title && (
        <div style={{ textAlign: 'center', fontSize: '0.85rem', fontWeight: 700, color: '#F9FAFB', marginBottom: 4 }}>
          {title}
        </div>
      )}
      <div style={{ display: 'flex', justifyContent: 'center' }}>
        <svg
          viewBox={`0 0 ${size} ${size}`}
          width="100%"
          style={{ maxWidth: size, display: 'block' }}
        >
          {/* Dark background */}
          <rect width={size} height={size} fill={BG} />

          {/* Grid circles */}
          {GRID_LEVELS.map((lvl) => (
            <circle
              key={lvl}
              cx={cx} cy={cy}
              r={maxR * (lvl / 100)}
              fill="none"
              stroke={lvl === 100 ? '#6b7280' : '#374151'}
              strokeWidth={lvl === 100 ? 0.8 : 0.6}
              strokeDasharray={lvl < 100 ? '3 3' : undefined}
            />
          ))}

          {/* Radial dividers */}
          {data.map((_, i) => {
            const angle = -Math.PI / 2 + i * sliceAngle;
            const end = polarXY(angle, maxR);
            return (
              <line
                key={i}
                x1={cx} y1={cy}
                x2={end.x.toFixed(2)} y2={end.y.toFixed(2)}
                stroke="#374151" strokeWidth={0.6}
              />
            );
          })}

          {/* Background slices (dim, full radius) */}
          {data.map((d, i) => (
            <path
              key={`bg-${i}`}
              d={slicePath(i, maxR)}
              fill={COLORS[d.group] || '#00B8A9'}
              fillOpacity={0.12}
              stroke={BG}
              strokeWidth={0.8}
            />
          ))}

          {/* Filled slices */}
          {data.map((d, i) => {
            const fillR = (Math.max(d.percentile, 0) / 100) * maxR;
            return (
              <path
                key={`fill-${i}`}
                d={slicePath(i, Math.max(fillR, 0.5))}
                fill={COLORS[d.group] || '#00B8A9'}
                fillOpacity={0.92}
                stroke={BG}
                strokeWidth={0.8}
              />
            );
          })}

          {/* Percentile value badges */}
          {data.map((d, i) => {
            const fillR = (Math.max(d.percentile, 0) / 100) * maxR;
            if (fillR < 12) return null;
            const midAngle = -Math.PI / 2 + (i + 0.5) * sliceAngle;
            const badgeR = fillR * 0.58 + 6;
            const pos = polarXY(midAngle, badgeR);
            return (
              <g key={`val-${i}`}>
                <rect
                  x={pos.x - 10} y={pos.y - 7}
                  width={20} height={13}
                  rx={3} ry={3}
                  fill="#00B8A9"
                  stroke="#000" strokeWidth={0.4}
                />
                <text
                  x={pos.x} y={pos.y + 4.5}
                  textAnchor="middle"
                  fill="#000"
                  fontSize={7}
                  fontWeight={800}
                  style={{ fontFamily: 'system-ui, sans-serif' }}
                >
                  {Math.round(d.percentile)}
                </text>
              </g>
            );
          })}

          {/* Metric labels */}
          {data.map((d, i) => {
            const midAngle = -Math.PI / 2 + (i + 0.5) * sliceAngle;
            const pos = polarXY(midAngle, labelR);
            const cosA = Math.cos(midAngle);
            const textAnchor = Math.abs(cosA) < 0.35 ? 'middle' : cosA > 0 ? 'start' : 'end';
            const color = COLORS[d.group] || '#94A3B8';
            const lines = splitLabel(d.label, 9);
            const lineH = 9;
            const startY = pos.y - ((lines.length - 1) * lineH) / 2;

            return (
              <text
                key={`lbl-${i}`}
                textAnchor={textAnchor}
                fill={color}
                fontSize={7}
                fontWeight={600}
                style={{ fontFamily: 'system-ui, sans-serif' }}
              >
                {lines.map((line, li) => (
                  <tspan key={li} x={pos.x.toFixed(2)} y={(startY + li * lineH).toFixed(2)}>
                    {line}
                  </tspan>
                ))}
              </text>
            );
          })}
        </svg>
      </div>

      {/* Legend */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: '1.1rem', marginTop: 10, flexWrap: 'wrap' }}>
        {GROUP_ORDER.map((g) => (
          <div key={g} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: '0.72rem', color: '#94A3B8', fontWeight: 600 }}>
            <div style={{ width: 10, height: 10, borderRadius: 2, background: COLORS[g], flexShrink: 0 }} />
            {g}
          </div>
        ))}
      </div>
    </div>
  );
}
