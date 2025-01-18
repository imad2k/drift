import os
from eodhd import APIClient
import pandas as pd
from dotenv import load_dotenv
from pprint import pprint
import json

# Load the API key from the .env file
load_dotenv()

# Get the API key from the environment variables
EOD_API_KEY = os.getenv('EOD_API_KEY')
if EOD_API_KEY is None:
    raise ValueError("EOD_API_KEY is not set in the environment variables")

# Create an instance of the APIClient
eod_api = APIClient(api_key=EOD_API_KEY)

#get user input
ticker = input("Enter the ticker: ")
start_date = input("Enter the start date in the format YYYY-MM-DD: ")
end_date = input("Enter the end date in the format YYYY-MM-DD: ")


# This function will get all the relevant fundamental data for a given ticker
def get_fundamentals_data(ticker):
    data = eod_api.get_fundamentals_data(ticker)
    company_code = data['General']['Code']
    company_name = data['General']['Name']
    company_sector = data['General']['Sector']   
    highlights = data["Highlights"]
    valuation = data["Valuation"]
    technicals = data["Technicals"]
    earnings = data["Earnings"]
    financials = data["Financials"]
    return json.dumps({
        'Code': company_code,
        'Name': company_name,
        'Sector': company_sector,
        'Highlights': highlights,
        'Valuation': valuation,
        'Technicals': technicals,
        'Earnings': earnings,
        'Financials': financials
    })
    
# This function will get historical data for a given ticker
def get_historical_data(ticker, start_date, end_date):
    data = eod_api.get_historical_data(
        symbol=ticker,
        interval='d',
        iso8601_start=start_date,
        iso8601_end=end_date,
        results=300
    )
    return data.to_json(orient='records')
    

# Get historical data and print the JSON response
historical_data_json = get_historical_data(ticker, start_date, end_date)
# pprint(historical_data_json)

# Get fundamental data and print the JSON response
fundamental_data_json = get_fundamentals_data(ticker)
# print(fundamental_data_json)

