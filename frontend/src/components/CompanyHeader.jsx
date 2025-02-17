// src/components/CompanyHeader.jsx
import React from 'react';

export default function CompanyHeader({ companyName, logoUrl }) {
  return (
    <div className="header-row">
      {logoUrl && <img src={logoUrl} alt={`${companyName} logo`} className="company-logo" />}
      <h1>{companyName}</h1>
    </div>
  );
}
