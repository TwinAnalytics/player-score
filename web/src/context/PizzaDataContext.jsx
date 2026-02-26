import { createContext, useContext, useMemo } from 'react';
import { useCSV } from '../hooks/useCSV';

const BASE = import.meta.env.BASE_URL;
const PIZZA_CSV = `${BASE}data/player_pizza_all_seasons.csv`;

const PizzaDataContext = createContext(null);

export const PIZZA_DIMS = [
  { key: 'Succ_Per90',  label: 'Dribbles',      group: 'Possession' },
  { key: 'Cmp%',        label: 'Pass Cmp%',      group: 'Possession' },
  { key: 'PrgC_Per90',  label: 'Prog Carries',   group: 'Possession' },
  { key: 'PrgP_Per90',  label: 'Prog Passes',    group: 'Possession' },
  { key: 'TB_Per90',    label: 'Through Balls',   group: 'Possession' },
  { key: 'Gls_Per90',   label: 'Goals',          group: 'Attacking' },
  { key: 'Ast_Per90',   label: 'Assists',        group: 'Attacking' },
  { key: 'xG_Per90',    label: 'xG',             group: 'Attacking' },
  { key: 'xAG_Per90',   label: 'xAG',            group: 'Attacking' },
  { key: 'SoT_Per90',   label: 'Shots on Target', group: 'Attacking' },
  { key: 'SCA_Per90',   label: 'Shot-Creating',  group: 'Attacking' },
  { key: 'KP_Per90',    label: 'Key Passes',     group: 'Attacking' },
  { key: 'TklW_Per90',  label: 'Tackles Won',    group: 'Defending' },
  { key: 'Int_Per90',   label: 'Interceptions',  group: 'Defending' },
  { key: 'Blocks_Per90',label: 'Blocks',         group: 'Defending' },
  { key: 'Clr_Per90',   label: 'Clearances',     group: 'Defending' },
];

export const PIZZA_GROUP_COLORS = {
  Possession: '#80F5E3',
  Attacking:  '#00B8A9',
  Defending:  '#006058',
};

export function PizzaDataProvider({ children }) {
  const { data: pizzaRows, loading, error } = useCSV(PIZZA_CSV);

  function getPizzaForPlayer(playerName, season) {
    if (!pizzaRows) return null;
    return pizzaRows.find((r) => r.Player === playerName && r.Season === season) || null;
  }

  function getPeers(role, season) {
    if (!pizzaRows) return [];
    return pizzaRows.filter((r) => r.Pos === role && r.Season === season);
  }

  const value = { pizzaRows, getPizzaForPlayer, getPeers, loading, error };

  return (
    <PizzaDataContext.Provider value={value}>
      {children}
    </PizzaDataContext.Provider>
  );
}

export function usePizzaData() {
  const ctx = useContext(PizzaDataContext);
  if (!ctx) throw new Error('usePizzaData must be used inside PizzaDataProvider');
  return ctx;
}
