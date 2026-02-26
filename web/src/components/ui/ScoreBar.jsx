import { bandColor } from '../../constants/scoring';
import styles from './ScoreBar.module.css';

export default function ScoreBar({ score, band, showValue = false }) {
  const pct = Math.min(Math.max(parseFloat(score) / 1000, 0), 1) * 100;
  const color = band ? bandColor(band) : 'var(--accent)';

  return (
    <div className={styles.wrap}>
      <div className={styles.track}>
        <div className={styles.fill} style={{ width: `${pct}%`, background: color }} />
      </div>
      {showValue && <span className={styles.val}>{Math.round(parseFloat(score))}</span>}
    </div>
  );
}
