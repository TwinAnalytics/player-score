import { createContext, useContext, useMemo } from 'react';
import { useCSV } from '../hooks/useCSV';

const BASE = import.meta.env.BASE_URL;
const MV_CSV = `${BASE}data/player_market_values.csv`;

const MarketValueContext = createContext(null);

export function MarketValueProvider({ children }) {
  const { data: mvRows, loading, error } = useCSV(MV_CSV);

  // Map key: "Player||Squad" → MarketValue_EUR
  const mvMap = useMemo(() => {
    if (!mvRows) return new Map();
    const map = new Map();
    for (const row of mvRows) {
      if (row.Player && row.Squad) {
        map.set(`${row.Player}||${row.Squad}`, parseFloat(row.MarketValue_EUR) || null);
      }
    }
    return map;
  }, [mvRows]);

  function getMV(player, squad) {
    return mvMap.get(`${player}||${squad}`) ?? null;
  }

  const value = { mvMap, getMV, loading, error };

  return (
    <MarketValueContext.Provider value={value}>
      {children}
    </MarketValueContext.Provider>
  );
}

export function useMarketValue() {
  const ctx = useContext(MarketValueContext);
  if (!ctx) throw new Error('useMarketValue must be used inside MarketValueProvider');
  return ctx;
}
