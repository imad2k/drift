// src/pages/HomePage.jsx
import React, { useState, useEffect } from 'react';
import { getTopPicks, getUserTickers } from '../api/companyApi';
import AddCompanyModal from '../components/AddCompanyModal';
import { Link } from 'react-router-dom';

export default function HomePage() {
  const username = 'Leila'; // Replace with dynamic auth info if available
  const [topPicks, setTopPicks] = useState([]);
  const [userTickers, setUserTickers] = useState([]);
  const [showModal, setShowModal] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const picksResponse = await getTopPicks();
        // Ensure we have an array of picks
        setTopPicks(Array.isArray(picksResponse.data) ? picksResponse.data : []);

        const tickersResponse = await getUserTickers(username);
        setUserTickers(tickersResponse.tickers || []);
      } catch (error) {
        console.error('Error fetching data:', error);
        setError('Failed to load data');
      }
    }
    fetchData();
  }, [username]);

  // Callback when a new ticker is added in the modal
  const handleAddTicker = (newTicker) => {
    setUserTickers((prev) => [...prev, newTicker]);
    setShowModal(false);
  };

  if (error) {
    return <div className="error-message">{error}</div>;
  }

  return (
    <div className="home-container">
      <h1>Welcome, {username}!</h1>
      <p>Manage your stock predictions and watchlist.</p>
      <button onClick={() => setShowModal(true)} className="add-ticker-btn">
        + Add Ticker
      </button>
      <section className="user-tickers-section">
        <h2>Your Watchlist</h2>
        {userTickers.length === 0 ? (
          <p>No tickers added yet.</p>
        ) : (
          <ul className="ticker-list">
            {userTickers.map((tickerObj) => (
              <li key={tickerObj.symbol} className="ticker-item">
                <Link to={`/company/${tickerObj.symbol}`}>{tickerObj.symbol}</Link>
              </li>
            ))}
          </ul>
        )}
      </section>
      <section className="top-picks-section">
        <h2>Top Picks</h2>
        {!Array.isArray(topPicks) || topPicks.length === 0 ? (
          <p>Loading top picks...</p>
        ) : (
          <ul className="ticker-list">
            {topPicks.map((pick) => (
              <li key={pick.symbol} className="ticker-item">
                {pick.symbol} - {pick.name}
              </li>
            ))}
          </ul>
        )}
      </section>
      {showModal && (
        <AddCompanyModal
          username={username}
          onClose={() => setShowModal(false)}
          onAddTicker={handleAddTicker}
        />
      )}
    </div>
  );
}
