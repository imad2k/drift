import os
import requests
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
EOD_API_KEY = os.getenv('EOD_API_KEY')
RDS_HOST = os.getenv('RDS_HOST')
RDS_PORT = int(os.getenv('RDS_PORT', 5432))
RDS_USER = os.getenv('RDS_USER')
RDS_PASSWORD = os.getenv('RDS_PASSWORD')


# stock_prediction_app/data/data_fetching.py


def fetch_fundamental_data(ticker, api_token):
    """
    Fetch fundamental data for the given ticker from EODHD.
    """
    url = f"https://eodhd.com/api/fundamentals/{ticker}?api_token={api_token}&fmt=json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch fundamental data for {ticker}. Status code: {response.status_code}")
        return {}

def fetch_news_sentiment(tickers, start_date, end_date):
    """
    Fetch news sentiment from EODHD for the given tickers and date range.
    """
    EOD_API_KEY = os.getenv('EOD_API_KEY')
    tickers_str = ','.join(tickers)
    url = (f"https://eodhd.com/api/sentiments?"
           f"s={tickers_str}&from={start_date}&to={end_date}"
           f"&api_token={EOD_API_KEY}&fmt=json")
    response = requests.get(url)
    
    print(f"Fetching news sentiment for tickers: {tickers}")
    print(f"URL: {url}")
    print(f"Response Status Code: {response.status_code}")
    
    if response.status_code != 200:
        print(f"Failed to fetch news sentiment: {response.text}")
        return {}
    
    sentiment_data = response.json()
    
    # Convert response to a usable DataFrame format
    sentiment_dfs = {}
    for ticker, data in sentiment_data.items():
        df = pd.DataFrame(data)
        df["ticker"] = ticker
        sentiment_dfs[ticker] = df
    
    # Concatenate all ticker data
    if sentiment_dfs:
        return pd.concat(sentiment_dfs.values(), ignore_index=True)
    else:
        return pd.DataFrame()

def fetch_economic_events(start_date, end_date, country="USA"):
    """
    Fetch economic events from EODHD for the given date range and country.
    """
    EOD_API_KEY = os.getenv('EOD_API_KEY')
    url = "https://eodhd.com/api/economic-events"
    params = {
        "api_token": EOD_API_KEY,
        "from": start_date,
        "to": end_date,
        "country": country,
        "limit": 1000,
        "offset": 0
    }
    response = requests.get(url, params=params)
    print(f"Fetching economic events data: {response.url}")
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching economic events data: {response.text}")
        return []

def fetch_macroeconomic_data():
    """
    Fetch macroeconomic data (example: UK10Y.GBOND from EODHD).
    """
    EOD_API_KEY = os.getenv('EOD_API_KEY')
    url = f"https://eodhd.com/api/eod/UK10Y.GBOND?api_token={EOD_API_KEY}&fmt=json"
    response = requests.get(url)
    
    print(f"URL: {url}")
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Text: {response.text}")
    print(f"Response Headers: {response.headers}")
    
    if response.status_code == 200:
        try:
            return response.json()
        except json.JSONDecodeError:
            print("Error: Response is not valid JSON.")
            return {}
    else:
        print(f"Error: Failed to fetch data. Status Code: {response.status_code}, Response: {response.text}")
        return {}

def get_intraday_data(ticker, start_date, end_date, interval, EOD_API_KEY):
    """
    Fetch intraday data for the specified ticker, date range, and interval from EODHD.
    """
    # Convert date strings to Unix timestamps
    start_timestamp = int(time.mktime(datetime.strptime(start_date, "%Y-%m-%d").timetuple()))
    end_timestamp = int(time.mktime(datetime.strptime(end_date, "%Y-%m-%d").timetuple()))
    
    url = (f"https://eodhd.com/api/intraday/{ticker}?api_token={EOD_API_KEY}"
           f"&interval={interval}&from={start_timestamp}&to={end_timestamp}&fmt=json")
    
    response = requests.get(url)
    
    # Debugging: Print response details for troubleshooting
    print(f"Fetching intraday data for {ticker}")
    print(f"URL: {url}")
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Text: {response.text}")
    
    if response.status_code == 200:
        try:
            return response.json()  # Parse and return JSON data
        except json.JSONDecodeError as e:
            print(f"JSON decoding error for {ticker}: {e}")
            return []
    else:
        print(f"Error fetching intraday data for {ticker}: {response.text}")
        return []
