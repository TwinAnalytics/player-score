import { Routes, Route } from 'react-router-dom';
import NavBar from './components/layout/NavBar';
import Home from './pages/Home';
import Rankings from './pages/Rankings';
import PlayerProfile from './pages/PlayerProfile';
import TeamScores from './pages/TeamScores';
import HiddenGems from './pages/HiddenGems';
import ComparePlayers from './pages/ComparePlayers';

export default function App() {
  return (
    <>
      <NavBar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/rankings" element={<Rankings />} />
        <Route path="/profile" element={<PlayerProfile />} />
        <Route path="/teams" element={<TeamScores />} />
        <Route path="/hidden-gems" element={<HiddenGems />} />
        <Route path="/compare" element={<ComparePlayers />} />
      </Routes>
    </>
  );
}
