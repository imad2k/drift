import requests, json, time
from datetime import datetime
import time





def get_intraday_data(ticker, start_date, end_date, interval, EOD_API_KEY):
    """
    Fetch intraday data for the specified ticker, date range, and interval.
    """
    # Convert date strings to Unix timestamps
    start_timestamp = int(time.mktime(datetime.strptime(start_date, "%Y-%m-%d").timetuple()))
    end_timestamp = int(time.mktime(datetime.strptime(end_date, "%Y-%m-%d").timetuple()))
    
    
    # Build the URL for the API call
    url = f"https://eodhd.com/api/intraday/{ticker}?api_token={EOD_API_KEY}&interval={interval}&from={start_timestamp}&to={end_timestamp}&fmt=json"
    
    # Make the API request
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
            return []  # Return an empty list if JSON decoding fails
    else:
        print(f"Error fetching intraday data for {ticker}: {response.text}")
        return []  # Return an empty list for non-200 responses

EOD_API_KEY = "62fc32b65388e2.02738383"
ticker = "AAPL.US"
start_date = "2023-01-01"
end_date = "2023-01-31"
interval = "1h"

data = get_intraday_data(ticker, start_date, end_date, interval, EOD_API_KEY)
print(data)  # Verify the fetched intraday data
