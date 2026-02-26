import { useState } from 'react';
import { crestUrl, slugify } from '../../utils/crestUrl';
import styles from './ClubCrest.module.css';

export default function ClubCrest({ clubName, size = 24 }) {
  const [failed, setFailed] = useState(false);
  const url = crestUrl(clubName);
  const initials = (clubName || '?').slice(0, 2).toUpperCase();

  if (!url || failed) {
    return (
      <div
        className={styles.fallback}
        style={{ width: size, height: size, fontSize: size * 0.38 }}
        title={clubName}
      >
        {initials}
      </div>
    );
  }

  return (
    <img
      src={url}
      alt={clubName}
      width={size}
      height={size}
      className={styles.img}
      onError={() => setFailed(true)}
      title={clubName}
    />
  );
}
