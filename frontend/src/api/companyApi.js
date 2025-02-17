// src/api/companyApi.js
import axios from "axios";

/**
 * Base URL for your local backend that fetches user predictions from AWS RDS
 */
const API_BASE_URL = "http://127.0.0.1:5001";

/* ------------------------------------------------------------------
   1. Fetch a single logo via backend proxy
   ------------------------------------------------------------------ */
export async function getLogo(symbol) {
  try {
    const response = await axios.get(`${API_BASE_URL}/proxy/logo/${symbol}`);
    return response.data.logoUrl;
  } catch (error) {
    console.error("Error fetching logo:", error);
    return null;
  }
}

/* ------------------------------------------------------------------
   2. Fetch top picks via backend proxy
   ------------------------------------------------------------------ */
export async function getTopPicks() {
  try {
    const response = await axios.get(`${API_BASE_URL}/proxy/top-picks`);
    // Ensure we return an array
    return {
      data: Array.isArray(response.data) ? response.data : [],
      error: null
    };
  } catch (error) {
    console.error("getTopPicks error:", error);
    return {
      data: [],
      error: error.message
    };
  }
}

/* ------------------------------------------------------------------
   3. Fetch all user predictions from AWS RDS (via local API)
   ------------------------------------------------------------------ */
export async function getUserPredictions(username) {
  try {
    // Example: GET http://127.0.0.1:5001/user/{username}/predictions
    const response = await axios.get(`${API_BASE_URL}/user/${username}/predictions`);
    return response.data; // Should match { username, tickers, predictions } structure
  } catch (error) {
    console.error("Error fetching user predictions:", error);
    throw error;
  }
}

/* ------------------------------------------------------------------
   4. Fetch predictions for a specific ticker
   ------------------------------------------------------------------ */
export async function getTickerPredictions(username, ticker) {
  try {
    // Example: GET http://127.0.0.1:5001/user/{username}/predictions/{ticker}
    const response = await axios.get(`${API_BASE_URL}/user/${username}/predictions/${ticker}`);
    return response.data; // Should match { username, tickers, predictions } structure
  } catch (error) {
    console.error("Error fetching ticker predictions:", error);
    throw error;
  }
}

/* ------------------------------------------------------------------
   5. Add new ticker to user's watchlist
   ------------------------------------------------------------------ */
export async function addUserTicker(username, ticker, companyName) {
  try {
    const response = await axios.post(`${API_BASE_URL}/user/${username}/tickers`, {
      ticker,
      company_name: companyName,
    });
    return response.data;
  } catch (error) {
    console.error("Error adding ticker:", error);
    throw error;
  }
}

/* ------------------------------------------------------------------
   6. Get user's watched tickers
   ------------------------------------------------------------------ */
export async function getUserTickers(username) {
  try {
    const response = await axios.get(`${API_BASE_URL}/user/${username}/tickers`);
    return response.data;
  } catch (error) {
    console.error("Error fetching user tickers:", error);
    throw error;
  }
}
