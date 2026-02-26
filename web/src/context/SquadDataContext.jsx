import { createContext, useContext, useMemo } from 'react';
import { useCSV } from '../hooks/useCSV';

const BASE = import.meta.env.BASE_URL;
const SQUAD_CSV = `${BASE}data/squad_scores_all_seasons.csv`;

const SquadDataContext = createContext(null);

export function SquadDataProvider({ children }) {
  const { data: squadRows, loading, error } = useCSV(SQUAD_CSV);

  const seasons = useMemo(() => {
    if (!squadRows) return [];
    return [...new Set(squadRows.map((r) => r.Season).filter(Boolean))].sort().reverse();
  }, [squadRows]);

  const comps = useMemo(() => {
    if (!squadRows) return [];
    return [...new Set(squadRows.map((r) => r.Comp).filter(Boolean))].sort();
  }, [squadRows]);

  const value = { squadRows, seasons, comps, loading, error };

  return (
    <SquadDataContext.Provider value={value}>
      {children}
    </SquadDataContext.Provider>
  );
}

export function useSquadData() {
  const ctx = useContext(SquadDataContext);
  if (!ctx) throw new Error('useSquadData must be used inside SquadDataProvider');
  return ctx;
}
