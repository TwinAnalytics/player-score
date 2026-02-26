import { BAND_COLORS } from '../../constants/colors';
import styles from './ScoreBadge.module.css';

export default function ScoreBadge({ band }) {
  const color = BAND_COLORS[band] || BAND_COLORS['Below Big-5 Level'];
  return (
    <span className={styles.badge} style={{ color, borderColor: `${color}44`, background: `${color}18` }}>
      {band || '—'}
    </span>
  );
}
