// src/components/Predictions.jsx
import React, { useState } from 'react';

const horizons = ["1D", "2D", "3D", "7D", "30D"];

function PredictionItem({ modelName, prediction, date }) {
  return (
    <div className="prediction-item">
      <h4>{modelName}</h4>
      <div className="prediction-value">{prediction}</div>
      <div className="prediction-date">{date}</div>
    </div>
  );
}

export default function Predictions({ predictions }) {
  const [selectedHorizon, setSelectedHorizon] = useState("1D");

  // Filter predictions by horizon – assuming each prediction has a field forecast_horizon
  const filteredPredictions = predictions.filter(pred => pred.forecast_horizon === selectedHorizon);

  const models = ["Ensemble", "XGBoost", "GradientBoost", "CATBoost", "RandomForest", "LightGMB"];
  
  return (
    <div className="predictionsContainer">
      <div className="horizon-toggle">
        {horizons.map(h => (
          <button 
            key={h} 
            className={selectedHorizon === h ? 'selected-toggle' : 'toggle-btn'}
            onClick={() => setSelectedHorizon(h)}
          >
            {h}
          </button>
        ))}
      </div>
      <div className="predictions-grid">
        {models.map(model => {
          const modelPred = filteredPredictions.find(pred => pred.model_name === model);
          return (
            <PredictionItem 
              key={model}
              modelName={model}
              prediction={modelPred ? modelPred.predicted_value : 'N/A'}
              date={modelPred ? modelPred.forecast_date : ''}
            />
          );
        })}
      </div>
    </div>
  );
}
