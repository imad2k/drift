# stock_prediction_app/data/data_fetching.py
import os
import requests
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import re
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
    Fetch raw fundamentals JSON from EODHD for `ticker`, then fix any occurrences 
    of '}}SomeWord:{' -> '}}, "SomeWord": {' or variations with extra spaces. 
    Finally, return a Python dictionary (or {} if still invalid).
    """
    url = f"https://eodhd.com/api/fundamentals/{ticker}?api_token={api_token}&fmt=json"
    response = requests.get(url)

    if response.status_code != 200:
        return {}

    # 1) Get the raw text
    raw_text = response.text

    # 2) Regex to fix known patterns like '}}Word:{' (with possible spaces)
    pattern = r'(\}\})\s*([A-Za-z0-9_]+)\s*:\s*\{'
    replacement = r'}}, "\2": {'
    fixed_text = re.sub(pattern, replacement, raw_text)

    # 3) Attempt to parse
    try:
        data = json.loads(fixed_text)
        return data
    except json.JSONDecodeError:
        # Return an empty dict if we can't parse
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
    
    if response.status_code != 200:
        print(f"Failed to fetch news sentiment: {response.text}")
        return {}
    return response.json()

def fetch_economic_events(api_token, from_date, to_date, country="USA", limit=1000, offset=0):
    """
    Fetch economic events data from EODHD.
    """
    url = (f"https://eodhd.com/api/economic-events?api_token={api_token}"
           f"&from={from_date}&to={to_date}&country={country}&limit={limit}&offset={offset}")
    response = requests.get(url)
    
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
    
    if response.status_code == 200:
        try:
            data = response.json()
            if not data:
                print(f"No data returned for {ticker}")
            return data  # Parse and return JSON data
        except json.JSONDecodeError as e:
            print(f"JSON decoding error for {ticker}: {e}")
            return []
    else:
        print(f"Error fetching intraday data for {ticker}: {response.text}")
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
