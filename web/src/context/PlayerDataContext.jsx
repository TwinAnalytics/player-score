import { createContext, useContext, useMemo } from 'react';
import { useCSV } from '../hooks/useCSV';
import { buildPlayerMap, getSeasons, getComps } from '../utils/dataHelpers';

const BASE = import.meta.env.BASE_URL;
const LONG_CSV = `${BASE}data/player_scores_all_seasons_long.csv`;

const PlayerDataContext = createContext(null);

export function PlayerDataProvider({ children }) {
  const { data: allRows, loading, error } = useCSV(LONG_CSV);

  const playerMap = useMemo(() => {
    if (!allRows) return new Map();
    return buildPlayerMap(allRows);
  }, [allRows]);

  const seasons = useMemo(() => getSeasons(allRows || []), [allRows]);
  const comps = useMemo(() => getComps(allRows || []), [allRows]);
  const playerNames = useMemo(() => [...playerMap.keys()].sort(), [playerMap]);

  const value = { allRows, playerMap, playerNames, seasons, comps, loading, error };

  return (
    <PlayerDataContext.Provider value={value}>
      {children}
    </PlayerDataContext.Provider>
  );
}

export function usePlayerData() {
  const ctx = useContext(PlayerDataContext);
  if (!ctx) throw new Error('usePlayerData must be used inside PlayerDataProvider');
  return ctx;
}
