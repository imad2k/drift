import os
from eodhd import APIClient
import pandas as pd
from dotenv import load_dotenv
from pprint import pprint

# Load the API key from the .env file
load_dotenv()

# Get the API key from the environment variables
EOD_API_KEY = os.getenv('EOD_API_KEY')
if EOD_API_KEY is None:
    raise ValueError("EOD_API_KEY is not set in the environment variables")

# Create an instance of the APIClient
eod_api = APIClient(api_key=EOD_API_KEY)

ticker = input("Enter the ticker: ")

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
    

get_fundamentals_data(ticker)

