// src/components/CompanyPerformanceChart.jsx
import React from 'react';
import { ResponsiveLine } from '@nivo/line';

export default function CompanyPerformanceChart({ data }) {
  // Nivo expects series data in the format: [{ id: 'seriesName', data: [{ x, y }, ...] }]
  const chartData = [
    {
      id: 'Ensemble',
      data: data,
    },
  ];

  return (
    <div style={{ height: '300px' }}>
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
          legend: 'Forecast Date',
          legendOffset: 36,
          legendPosition: 'middle',
        }}
        axisLeft={{
          orient: 'left',
          tickSize: 5,
          tickPadding: 5,
          tickRotation: 0,
          legend: 'Predicted Value',
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
  );
}
