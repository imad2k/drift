// src/components/ModelAccuracy.jsx
import React from 'react';

export default function ModelAccuracy({ metrics }) {
  // Assume metrics is an object: { accuracy, r2, mse, mae }
  return (
    <div className="performanceMetricsContainer">
      <div className="metric-item">
        <h4>Accuracy</h4>
        <p>{metrics.accuracy || 'N/A'}</p>
      </div>
      <div className="metric-item">
        <h4>R2</h4>
        <p>{metrics.r2 || 'N/A'}</p>
      </div>
      <div className="metric-item">
        <h4>MSE</h4>
        <p>{metrics.mse || 'N/A'}</p>
      </div>
      <div className="metric-item">
        <h4>MAE</h4>
        <p>{metrics.mae || 'N/A'}</p>
      </div>
    </div>
  );
}
