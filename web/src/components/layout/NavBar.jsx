import { NavLink } from 'react-router-dom';
import styles from './NavBar.module.css';

const NAV_ITEMS = [
  { to: '/', label: 'Home', end: true },
  { to: '/rankings', label: 'Rankings' },
  { to: '/profile', label: 'Player Profile' },
  { to: '/teams', label: 'Team Scores' },
  { to: '/hidden-gems', label: 'Hidden Gems' },
  { to: '/compare', label: 'Compare' },
];

export default function NavBar() {
  return (
    <nav className={styles.nav}>
      <div className={styles.inner}>
        <NavLink to="/" className={styles.brand} end>
          <span className={styles.brandAccent}>Player</span>Score
        </NavLink>
        <ul className={styles.links}>
          {NAV_ITEMS.map(({ to, label, end }) => (
            <li key={to}>
              <NavLink
                to={to}
                end={end}
                className={({ isActive }) =>
                  `${styles.link} ${isActive ? styles.active : ''}`
                }
              >
                {label}
              </NavLink>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  );
}
