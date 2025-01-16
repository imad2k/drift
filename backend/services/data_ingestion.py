import os
from eodhd import APIClient
import pandas as pd
from dotenv import load_dotenv
from pprint import pprint

# Load the API key from the .env file
load_dotenv()

EOD_API_KEY = os.getenv('EOD_API_KEY')
if EOD_API_KEY is None:
    raise ValueError("EOD_API_KEY is not set in the environment variables")

eod_api = APIClient(api_key=EOD_API_KEY)

ticker = input("Enter the ticker: ")

def get_fundamentals_data(ticker):
    data = eod_api.get_fundamentals_data(ticker)
    pprint(data)

get_fundamentals_data(ticker)

