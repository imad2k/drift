import os
from dotenv import load_dotenv
import json
import requests
from datetime import datetime

# Load the API key from the .env file
load_dotenv()

# Get the API key from the environment variables
EOD_API_KEY = os.getenv('EOD_API_KEY')
if EOD_API_KEY is None:
    raise ValueError("EOD_API_KEY is not set in the environment variables")

# Convert date to UNIX time
def date_to_unix(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp())

# This function will get historical data for a given ticker
def get_historical_data(ticker, start_date, end_date, EOD_API_KEY):
    url = f'https://eodhistoricaldata.com/api/eod/{ticker}.US'
    params = {
        'from': start_date,
        'to': end_date,
        'period': 'd',
        'api_token': EOD_API_KEY,
        'fmt': 'json'
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    historical_data = response.json()
    return json.dumps(historical_data, indent=2)

# This function will get intraday data for a given ticker
def get_intraday_data(ticker, start_date, end_date, interval, EOD_API_KEY):
    """
    Fetches intraday data (interval: 1m, 5m, 15m, 1h, etc.).
    Example: https://eodhistoricaldata.com/api/intraday/{ticker}
    """
    url = f"https://eodhistoricaldata.com/api/intraday/{ticker}"
    params = {
        'api_token': EOD_API_KEY,
        'interval': interval,
        'from': date_to_unix(start_date),
        'to': date_to_unix(end_date),
        'fmt': 'json'
    }
    # print(f"Request URL: {url}")
    # print(f"Request Params: {params}")
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    # print(json.dumps(resp.json(), indent=2))
    return json.dumps(resp.json(), indent=2)


def get_fundamentals_data(ticker):
    data = eod_api.get_fundamentals_data(ticker)
    
    #This is the general data
    company_code = data['General']['Code']
    company_name = data['General']['Name']
    company_sector = data['General']['Sector']
    
    highlights = data["Highlights"]
    valuation = data["Valuation"]
    technicals = data["Technicals"]
    earnings = data["Earnings"]
    financials = data["Financials"]
    
    pprint({'company highlights: ':highlights, 
            })
    
def get_daily_data(ticker, start_date, end_date, api_key):
    """
    Fetch daily historical price data (OHLCV) for the given ticker
    and date range from EOD Historical Data.
    """
    url = f"https://eodhistoricaldata.com/api/eod/{ticker}"
    params = {
        'from': start_date,
        'to': end_date,
        'api_token': api_key,
        'fmt': 'json'
    }

    response = requests.get(url, params=params)
    # Raises requests.exceptions.HTTPError if the status isn't 200
    response.raise_for_status()

    # Convert response content to JSON, then re-serialize with indentation
    return json.dumps(response.json(), indent=2)

# Example usage
# ticker = input("Enter the ticker symbol (e.g., 'AAPL.US'): ")
# start_date = input("Enter the start date (YYYY-MM-DD): ")
# end_date = input("Enter the end date (YYYY-MM-DD): ")
# interval = input("Enter intraday interval (e.g., '5m', '15m', '1h'): ")
#
# get_intraday_data(ticker, start_date, end_date, interval, EOD_API_KEY)

