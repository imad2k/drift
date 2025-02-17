// src/pages/CompanyDetailsPage.jsx
import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getLogo, getTickerPredictions, getUserTickers } from '../api/companyApi';
import NavigationBar from '../components/NavigationBar';
import CompanyHeader from '../components/CompanyHeader';
import Predictions from '../components/Predictions';
import HeatMapComponent from '../components/HeatMapComponent';
import PerformanceChart from '../components/PerformanceChart';
import ModelAccuracy from '../components/ModelAccuracy';
import Footer from '../components/Footer';
import AddCompanyModal from '../components/AddCompanyModal';

export default function CompanyDetailsPage() {
  const { ticker } = useParams();
  const navigate = useNavigate();
  const username = 'Leila'; // Replace with dynamic auth info as needed

  const [logoUrl, setLogoUrl] = useState(null);
  const [predictionsData, setPredictionsData] = useState(null);
  const [tickers, setTickers] = useState([]);
  const [showModal, setShowModal] = useState(false);

  // For simplicity, we assume predictionsData contains:
  // { predictions: [...], heatmap: [...], performanceData: { modelName: [...] }, metrics: { accuracy, r2, mse, mae } }
  const performanceData = predictionsData ? predictionsData.performanceData : {};
  const metrics = predictionsData ? predictionsData.metrics : {};
  const heatmapData = predictionsData ? predictionsData.heatmap : [];

  useEffect(() => {
    async function fetchData() {
      const logo = await getLogo(ticker);
      setLogoUrl(logo);
      try {
        const predictionsResponse = await getTickerPredictions(username, ticker);
        setPredictionsData(predictionsResponse);
      } catch (error) {
        console.error('Error fetching ticker predictions:', error);
      }
    }
    fetchData();
  }, [ticker, username]);

  useEffect(() => {
    async function fetchTickers() {
      try {
        const tickersResponse = await getUserTickers(username);
        setTickers(tickersResponse.tickers || []);
      } catch (error) {
        console.error('Error fetching user tickers:', error);
      }
    }
    fetchTickers();
  }, [username]);

  const handleCompanySelect = (newTicker) => {
    navigate(`/company/${newTicker}`);
  };

  const handleOpenModal = () => setShowModal(true);
  const handleCloseModal = () => setShowModal(false);
  const handleLogout = () => {
    // Implement your logout logic here
    console.log('Logging out...');
  };

  // For this example, we use the ticker as the company name.
  const companyName = ticker;

  return (
    <div className="dashboard">
      <NavigationBar 
        tickers={tickers}
        currentTicker={ticker}
        onCompanySelect={handleCompanySelect}
        onOpenModal={handleOpenModal}
        onLogout={handleLogout}
      />
      <CompanyHeader companyName={companyName} logoUrl={logoUrl} />
      <div style={{ display: 'contents' }}>
        <Predictions predictions={predictionsData ? predictionsData.predictions : []} />
        {/* <HeatMapComponent heatmapData={heatmapData} /> */}
      </div>
      <div style={{ display: 'contents' }}>
        <PerformanceChart performanceData={performanceData} />
        <ModelAccuracy metrics={metrics} />
      </div>
      <Footer />
      {showModal && <AddCompanyModal username={username} onClose={handleCloseModal} />}
    </div>
  );
}
