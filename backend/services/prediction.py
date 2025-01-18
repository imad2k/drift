import os
from openai import OpenAI
from dotenv import load_dotenv
from data_ingestion import get_fundamentals_data, get_historical_data
from pprint import pprint
import json

# Load the API key from the .env file
load_dotenv()

# Get the EOD API key from the environment variables
EOD_API_KEY = os.getenv('EOD_API_KEY')

# Get the API key from the environment variables
OpenAI_API_KEY = os.getenv('OPENAI_API_KEY')
if OpenAI_API_KEY is None:
    raise ValueError("OpenAI_API_KEY is not set in the environment variables")

# Create an instance of the OpenAI
openai = OpenAI(api_key=OpenAI_API_KEY)

# Prediction function
def predict_next_close(ticker, start_date, end_date):
    # Get historical data and parse the JSON response
    historical_data_json = get_historical_data(ticker, start_date, end_date, EOD_API_KEY)
    historical_data = json.loads(historical_data_json)

    # Get fundamental data and parse the JSON response
    fundamental_data_json = get_fundamentals_data(ticker)
    fundamental_data = json.loads(fundamental_data_json)
    
    # Prepare a prompt string with the relevant data
    # (Keep it concise to avoid token limit issues)
    prompt_str = "Historical close data:\n"
    for item in historical_data[-20:]:  # last 20 days for example
        prompt_str += f"{item['date']}: close={item['close']}\n"
    prompt_str += "\nPlease predict the next day's closing price."

    # Send to OpenAI for a standard ChatCompletion
    response = openai.chat.completions.create(
        model="o1-2024-12-17",
        messages=[
          {"role": "system", "content": "You are a world class expert financial analyst and helpful financial prediction model."},
          {"role": "user", "content": prompt_str}
        #   {"role": "user", "content": json.dumps(fundamental_data, indent=2)}
        ]
    )
    prediction_text = response.choices[0].message.content
    pprint(prediction_text)
    return prediction_text

# Get user input
ticker = input("Enter the ticker: ")
start_date = input("Enter the start date in the format YYYY-MM-DD: ")
end_date = input("Enter the end date in the format YYYY-MM-DD: ")
predict_next_close(ticker, start_date, end_date)
