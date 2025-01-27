#!/usr/bin/env python
import os
import sys
import requests
import json
from datetime import datetime
from urllib.parse import urlencode
from dotenv import load_dotenv

load_dotenv()
# If you prefer, read from environment:
EOD_API_KEY = os.getenv('EOD_API_KEY', "YOUR_API_KEY_HERE")

def fetch_news_sentiment(tickers, start_date, end_date, api_key=EOD_API_KEY):
    """
    Calls EODHD's sentiment endpoint for a list of tickers in [start_date, end_date].
    Returns either a dict of {ticker -> [articles]} or a list, depending on the API response.
    """
    if not isinstance(tickers, list):
        tickers = [tickers]
    tickers_str = ",".join(tickers)
    
    base_url = "https://eodhd.com/api/sentiments?"
    params = {
        "s": tickers_str,
        "from": start_date,
        "to": end_date,
        "api_token": api_key,
        "fmt": "json"
    }
    url = base_url + urlencode(params)
    print(f"[DEBUG] GET {url}")
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"[ERROR] status={resp.status_code}, text={resp.text}")
        return {}
    try:
        data = resp.json()
        return data
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON decode error: {e}")
        return {}

def main():
    """
    Simple script to test the sentiment API in isolation.
    Usage: python test_sentiment.py [ticker] [start_date] [end_date]
    Example: python test_sentiment.py AAPL.US 2025-01-01 2025-01-27
    """
    if len(sys.argv) < 4:
        print("Usage: python test_sentiment.py TICKER START_DATE END_DATE")
        print("Example: python test_sentiment.py AAPL.US 2025-01-01 2025-01-27")
        sys.exit(1)

    ticker_arg = sys.argv[1]
    start_arg = sys.argv[2]
    end_arg = sys.argv[3]

    print(f"[INFO] Fetching sentiment for {ticker_arg}, from={start_arg}, to={end_arg}...")
    result = fetch_news_sentiment([ticker_arg], start_arg, end_arg)
    
    print("[INFO] Raw Response:")
    print(json.dumps(result, indent=2))
    
    # If it's a dict of {ticker -> [articles]}, let's see how many we got:
    if isinstance(result, dict):
        for tk, articles in result.items():
            print(f"\nTicker: {tk}, Article Count: {len(articles)}")
            # Optionally show the first few
            for i, article in enumerate(articles[:3]):
                print(f"  Article {i+1}: date={article.get('date')}, normalized={article.get('normalized')}")

if __name__ == "__main__":
    main()
