import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import './index.css';
import App from './App.jsx';
import { PlayerDataProvider } from './context/PlayerDataContext';
import { SquadDataProvider } from './context/SquadDataContext';
import { MarketValueProvider } from './context/MarketValueContext';
import { PizzaDataProvider } from './context/PizzaDataContext';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter basename="/player-score">
      <PlayerDataProvider>
        <SquadDataProvider>
          <MarketValueProvider>
            <PizzaDataProvider>
              <App />
            </PizzaDataProvider>
          </MarketValueProvider>
        </SquadDataProvider>
      </PlayerDataProvider>
    </BrowserRouter>
  </StrictMode>,
);
