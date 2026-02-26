import { useNavigate } from 'react-router-dom';
import ClubCrest from './ClubCrest';
import ScoreBadge from './ScoreBadge';
import { BAND_COLORS } from '../../constants/colors';
import { formatScore } from '../../utils/formatters';
import styles from './SimilarPlayers.module.css';

export default function SimilarPlayers({ players }) {
  const navigate = useNavigate();

  if (!players || players.length === 0) return null;

  return (
    <div className={styles.wrap}>
      <h3 className={styles.title}>Similar Players</h3>
      <div className={styles.list}>
        {players.map((p, i) => {
          const score = parseFloat(p.MainScore);
          const color = BAND_COLORS[p.MainBand] || '#6B7280';
          return (
            <div
              key={i}
              className={styles.card}
              onClick={() => navigate(`/profile?player=${encodeURIComponent(p.Player)}`)}
            >
              <ClubCrest clubName={p.Squad} size={28} />
              <div className={styles.info}>
                <div className={styles.name}>{p.Player}</div>
                <div className={styles.meta}>{p.Squad} · {p.Comp}</div>
              </div>
              <div className={styles.score} style={{ color }}>
                {formatScore(score)}
              </div>
              <ScoreBadge band={p.MainBand} />
            </div>
          );
        })}
      </div>
    </div>
  );
}
