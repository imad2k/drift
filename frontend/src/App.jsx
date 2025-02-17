// src/App.jsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
// import HomePage from './pages/HomePage';
import CompanyDetailsPage from './pages/CompanyDetailsPage';
import './styles.css';

export default function App() {
  return (
    <Router>
      <div>
        
        <Routes>
              {/* <Route path="/" element={<HomePage />} /> */}
              <Route path="/" element={<CompanyDetailsPage />} />
              {/* <Route path="/company/:ticker" element={<CompanyDetailsPage />} /> */}
        </Routes>
        {/* <header class="top-nav">
          <Link to="/" className="app-title">Drift</Link>
        </header> */}
        
      
      </div>
    </Router>
  );
}
