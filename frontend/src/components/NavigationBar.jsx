// src/components/NavigationBar.jsx
import React from 'react';
import { useNavigate } from 'react-router-dom';

export default function NavigationBar({ tickers, currentTicker, onCompanySelect, onOpenModal, onLogout }) {
  const navigate = useNavigate();

  return (
    <header className="navBar">
      <div className="logoNameWrapper">
        <div className="logo">Drift</div>
      </div>
      <div className="pageLinksWrapper">
        {tickers.map(tickerObj => (
          <button 
            key={tickerObj.symbol} 
            className={tickerObj.symbol === currentTicker ? 'selectedNavLink' : 'navLink'}
            onClick={() => onCompanySelect(tickerObj.symbol)}
          >
            {tickerObj.symbol}
          </button>
        ))}
        <button className="navLink" onClick={onOpenModal}>+ New</button>
      </div>
      <div className="logout">
        {/* Replace with your avatar/dropdown logic */}
        <img 
          src="/path/to/avatar.png" 
          alt="User" 
          className="navLogo"
          onClick={onLogout}
          style={{cursor: 'pointer'}}
        />
      </div>
    </header>
  );
}
