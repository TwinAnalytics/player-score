import ClubCrest from './ClubCrest';
import ScoreBadge from './ScoreBadge';
import { formatMarketValue } from '../../utils/formatters';
import { POS_LABELS, bandColor } from '../../constants/scoring';

const ROLE_ATTRS = {
  FW:     [['Gls_Per90','Goals/90'], ['xG_Per90','xG/90'], ['Ast_Per90','Assists/90'], ['SoT_Per90','Shots on Tgt/90']],
  Off_MF: [['Gls_Per90','Goals/90'], ['Ast_Per90','Assists/90'], ['xAG_Per90','xAG/90'], ['SCA_Per90','Shot-Creating/90']],
  MF:     [['Ast_Per90','Assists/90'], ['xAG_Per90','xAG/90'], ['KP_Per90','Key Passes/90'], ['SCA_Per90','Shot-Creating/90']],
  Def_MF: [['TklW_Per90','Tackles Won/90'], ['Int_Per90','Interceptions/90'], ['KP_Per90','Key Passes/90'], ['Cmp%','Pass Cmp%']],
  DF:     [['Blocks_Per90','Blocks/90'], ['Int_Per90','Interceptions/90'], ['Clr_Per90','Clearances/90'], ['TklW_Per90','Tackles Won/90']],
};

function fmt(key, val) {
  if (val === null || val === undefined || isNaN(val)) return '—';
  if (key === 'Cmp%') return `${parseFloat(val).toFixed(1)}%`;
  return parseFloat(val).toFixed(2);
}

export default function FIFACard({ row, score, band, pizzaRow, marketValue }) {
  if (!row || !score) return null;

  const pos = row.Pos || '';
  const posLabel = POS_LABELS[pos] || pos;
  const attrs = ROLE_ATTRS[pos] || ROLE_ATTRS['MF'];
  const overall = Math.round(score);
  const color = bandColor(band);
  const age = Math.round(parseFloat(row.Age)) || '?';
  const nineties = parseFloat(row['90s'] || 0).toFixed(1);

  return (
    <div style={{
      display: 'flex',
      justifyContent: 'center',
    }}>
      <div style={{
        position: 'relative',
        width: 240,
        borderRadius: 20,
        padding: '16px 18px 14px',
        boxSizing: 'border-box',
        background: 'radial-gradient(circle at 5% 5%, #2EF2E0 0%, #00897B 42%, #0B1F1E 100%)',
        boxShadow: '0 16px 40px rgba(0,0,0,0.6)',
        color: '#fff',
        fontFamily: 'system-ui, -apple-system, sans-serif',
        flexShrink: 0,
      }}>
        {/* Top row: overall + meta */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          {/* Left: score + position + band */}
          <div>
            <div style={{ fontSize: 48, fontWeight: 900, lineHeight: 1, textShadow: '0 2px 8px rgba(0,0,0,0.4)' }}>
              {overall}
            </div>
            <div style={{ fontSize: 13, fontWeight: 700, letterSpacing: 2, marginTop: 2 }}>
              {posLabel}
            </div>
            <div style={{
              marginTop: 7,
              padding: '3px 9px',
              borderRadius: 999,
              fontSize: 9,
              textTransform: 'uppercase',
              letterSpacing: '0.07em',
              background: 'rgba(0,0,0,0.28)',
              border: '1px solid rgba(255,255,255,0.3)',
              display: 'inline-block',
              fontWeight: 700,
            }}>
              {band}
            </div>
            {marketValue && (
              <div style={{ marginTop: 6, fontSize: 12, fontWeight: 800, color: '#fde68a', letterSpacing: 0.5 }}>
                {formatMarketValue(marketValue)}
              </div>
            )}
          </div>
          {/* Right: age + league */}
          <div style={{ textAlign: 'right', fontSize: 11, opacity: 0.9 }}>
            <div style={{ fontWeight: 700 }}>Age {age}</div>
            <div style={{ marginTop: 4 }}>{nineties} 90s</div>
            <div style={{ marginTop: 4, fontSize: 10, opacity: 0.8, maxWidth: 90, textAlign: 'right' }}>
              {row.Comp}
            </div>
          </div>
        </div>

        {/* Player name */}
        <div style={{
          marginTop: 20,
          fontSize: 17,
          fontWeight: 900,
          letterSpacing: 1.5,
          textTransform: 'uppercase',
          textShadow: '0 2px 6px rgba(0,0,0,0.5)',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}>
          {row.Player}
        </div>

        {/* Club */}
        <div style={{
          marginTop: 4,
          fontSize: 11,
          opacity: 0.95,
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          overflow: 'hidden',
        }}>
          <ClubCrest clubName={row.Squad} size={16} />
          <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {row.Squad}
          </span>
        </div>

        {/* Divider */}
        <div style={{
          width: '60%',
          height: 1,
          marginTop: 14,
          marginBottom: 12,
          background: 'linear-gradient(to right, rgba(255,255,255,0.45), rgba(255,255,255,0))',
        }} />

        {/* Key attributes */}
        <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.08em', opacity: 0.7, textTransform: 'uppercase', marginBottom: 8 }}>
          Key Attributes
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', rowGap: 6, columnGap: 8 }}>
          {attrs.map(([key, label]) => {
            const val = pizzaRow ? fmt(key, pizzaRow[key]) : '—';
            return (
              <div key={key} style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <span style={{ fontSize: 9, opacity: 0.8, fontWeight: 500 }}>{label}</span>
                <span style={{ fontSize: 13, fontWeight: 800 }}>{val}</span>
              </div>
            );
          })}
        </div>

        {/* Footer */}
        <div style={{
          marginTop: 14,
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: 8,
          opacity: 0.55,
          letterSpacing: '0.05em',
          textTransform: 'uppercase',
        }}>
          <span>PlayerScore</span>
          <span>FBref Big-5</span>
        </div>
      </div>
    </div>
  );
}
