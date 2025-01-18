import os
from eodhd import APIClient
from dotenv import load_dotenv
from pprint import pprint
import json
import requests


# Load the API key from the .env file
load_dotenv()

# Get the API key from the environment variables
EOD_API_KEY = os.getenv('EOD_API_KEY')
if EOD_API_KEY is None:
    raise ValueError("EOD_API_KEY is not set in the environment variables")

# Create an instance of the APIClient
eod_api = APIClient(api_key=EOD_API_KEY)


# This function will get all the relevant fundamental data for a given ticker
def get_fundamentals_data(ticker):
    data = eod_api.get_fundamentals_data(ticker)
    company_code = data['General']['Code']
    company_name = data['General']['Name']
    company_sector = data['General']['Sector']   
    highlights = data["Highlights"]
    valuation = data["Valuation"]
    technicals = data["Technicals"]
    earnings_trend = data["Earnings"]["Trend"]
    financials = data["Financials"]
    return json.dumps({
        'Code': company_code,
        'Name': company_name,
        'Sector': company_sector,
        'Highlights': highlights,
        'Valuation': valuation,
        'Technicals': technicals,
        'Earnings_trend': earnings_trend,
        'Financials': financials
    })
    
# This function will get historical data for a given ticker
def get_historical_data(ticker, start_date, end_date, EOD_API_KEY):
    url = f'https://eodhd.com/api/eod/{ticker}.US?from={start_date}&to={end_date}&period=d&api_token={EOD_API_KEY}&fmt=json'
    response = requests.get(url)
    historical_data = response.json()  # Parse the JSON content
    return json.dumps(historical_data, indent=2)  # Convert to JSON string
    
   
# Example usage
# historical_data_json = get_historical_data('AAPL', '2025-01-01', '2025-01-04', EOD_API_KEY)
# print(historical_data_json)


