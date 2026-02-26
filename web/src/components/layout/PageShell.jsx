import styles from './PageShell.module.css';

export default function PageShell({ children, wide = false }) {
  return (
    <main className={`${styles.shell} ${wide ? styles.wide : ''}`}>
      {children}
    </main>
  );
}
