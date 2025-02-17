// src/components/PerformanceChart.jsx
import React, { useState } from 'react';
import { ResponsiveLine } from '@nivo/line';

const availableModels = ["Ensemble", "XGBoost", "GradientBoost", "CATBoost", "RandomForest", "LightGMB"];

export default function PerformanceChart({ performanceData }) {
  const [selectedModel, setSelectedModel] = useState("Ensemble");

  // Assume performanceData is an object keyed by model name containing an array of {x: time, y: percent} data
  const modelData = performanceData[selectedModel] || [];

  const chartData = [
    {
      id: selectedModel,
      data: modelData
    }
  ];

  return (
    <div className="performanceChartContainer">
      <div className="model-selection-nav">
        {availableModels.map(model => (
          <button 
            key={model} 
            className={selectedModel === model ? 'selected-model' : 'model-btn'}
            onClick={() => setSelectedModel(model)}
          >
            {model}
          </button>
        ))}
      </div>
      <div style={{ height: '100%' }}>
        <ResponsiveLine
          data={chartData}
          margin={{ top: 50, right: 50, bottom: 50, left: 60 }}
          xScale={{ type: 'point' }}
          yScale={{ type: 'linear', min: 'auto', max: 'auto', stacked: false }}
          axisBottom={{
            orient: 'bottom',
            tickSize: 5,
            tickPadding: 5,
            tickRotation: 45,
            legend: 'Time',
            legendOffset: 36,
            legendPosition: 'middle',
          }}
          axisLeft={{
            orient: 'left',
            tickSize: 5,
            tickPadding: 5,
            tickRotation: 0,
            legend: 'Percent Accuracy',
            legendOffset: -40,
            legendPosition: 'middle',
          }}
          colors={{ scheme: 'nivo' }}
          pointSize={8}
          pointColor={{ theme: 'background' }}
          pointBorderWidth={2}
          pointBorderColor={{ from: 'serieColor' }}
          useMesh={true}
        />
      </div>
    </div>
  );
}
