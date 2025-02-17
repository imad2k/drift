// src/components/AddCompanyModal.jsx
import React, { useState, useEffect } from 'react';
import { getTopPicks, addUserTicker } from '../api/companyApi';

export default function AddCompanyModal({ username, onClose, onAddTicker }) {
  const [companies, setCompanies] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    async function fetchCompanies() {
      const picks = await getTopPicks();
      setCompanies(picks);
    }
    fetchCompanies();
  }, []);

  // Filter companies based on search term
  const filteredCompanies = companies.filter(company =>
    company.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    company.symbol.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleAdd = async (company) => {
    try {
      await addUserTicker(username, company.symbol, company.name);
      onAddTicker({ symbol: company.symbol, name: company.name });
    } catch (error) {
      console.error('Error adding ticker:', error);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <h2>Add Ticker</h2>
        <input
          type="text"
          placeholder="Search companies..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="search-input"
        />
        <ul className="company-list">
          {filteredCompanies.map((company) => (
            <li key={company.symbol} className="company-item">
              <button onClick={() => handleAdd(company)} className="company-btn">
                {company.symbol} - {company.name}
              </button>
            </li>
          ))}
        </ul>
        <button onClick={onClose} className="close-modal-btn">Close</button>
      </div>
    </div>
  );
}
