// src/components/HeatMapComponent.jsx
import React, { useState } from 'react';
import { ResponsiveHeatMap } from '@nivo/heatmap';

const trainingWindows = [
  { label: '8M', value: 250 },
  { label: '7M', value: 210 },
  { label: '6M', value: 180 },
  { label: '5M', value: 150 }
];

export default function HeatMapComponent({ heatmapData }) {
  const [selectedWindow, setSelectedWindow] = useState(trainingWindows[0].value);

  // You may filter or modify heatmapData based on selectedWindow
  return (
    <div className="heatmapContainer">
      <div className="heatmap-dropdown">
        <select value={selectedWindow} onChange={e => setSelectedWindow(Number(e.target.value))}>
          {trainingWindows.map(win => (
            <option key={win.value} value={win.value}>{win.label}</option>
          ))}
        </select>
      </div>
      <div style={{ height: '100%', width: '100%' }}>
        <ResponsiveHeatMap
          data={heatmapData}
          keys={["1D", "2D", "3D", "7D", "30D"]}
          indexBy="id"
          margin={{ top: 40, right: 60, bottom: 40, left: 60 }}
          colors={{ scheme: 'reds' }}  // Updated color configuration
          axisTop={{
            orient: "top",
            tickSize: 5,
            tickPadding: 5,
            tickRotation: 0,
          }}
          axisRight={null}
          axisLeft={{
            tickSize: 5,
            tickPadding: 5,
            tickRotation: 0,
          }}
          axisBottom={null}
        />
      </div>
    </div>
  );
}
