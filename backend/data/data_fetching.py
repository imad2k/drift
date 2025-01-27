# stock_prediction_app/data/data_fetching.py

import os
import requests
import time
import json
import re
from datetime import datetime, timedelta
from urllib.parse import urlencode
from dotenv import load_dotenv

load_dotenv()
EOD_API_KEY = os.getenv('EOD_API_KEY')

# If you use the same 'INDICATOR_CONFIGS' approach as in your script, 
# you can define it here or import from a separate file.

# ---------- Helpers ----------

def _fetch_json(url):
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"[WARN] Request failed: {resp.text}")
        return []
    try:
        data = resp.json()
        # a small sleep to avoid hitting rate-limits
        time.sleep(0.1)
        if isinstance(data, list):
            return data
        else:
            return []
    except json.JSONDecodeError:
        print(f"[ERROR] JSON decode error on {url}")
        return []

# ---------- Fundamentals ----------

def fetch_fundamental_data(ticker, api_token):
    """
    Example: https://eodhd.com/api/fundamentals/AAPL.US?api_token=XXX&fmt=json
    """
    url = f"https://eodhd.com/api/fundamentals/{ticker}?api_token={api_token}&fmt=json"
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"[WARN] fundamentals request {resp.text}")
        return {}

    raw_text = resp.text
    # Fix known broken patterns
    pattern = r'(\}\})\s*([A-Za-z0-9_]+)\s*:\s*\{'
    replacement = r'}}, "\2": {'
    fixed_text = re.sub(pattern, replacement, raw_text)
    try:
        data = json.loads(fixed_text)
        return data if data else {}
    except json.JSONDecodeError:
        return {}

# ---------- Daily EOD (adjusted close) ----------

def fetch_daily_eod(ticker, start_date, end_date, api_token):
    """
    Example usage:
      https://eodhd.com/api/eod/AAPL.US?from=YYYY-MM-DD&to=YYYY-MM-DD&fmt=json
    Returns a list of dicts with keys like: date, open, high, low, close, adjusted_close, volume, ...
    """
    params = {
        'from': start_date,
        'to': end_date,
        'api_token': api_token,
        'fmt': 'json'
    }
    url = f"https://eodhd.com/api/eod/{ticker}?{urlencode(params)}"
    return _fetch_json(url)

# ---------- Technical Indicators (Daily) ----------

def fetch_technical_data(ticker, start_date, end_date, api_token, buffer_days=20, indicators=None):
    """
    Indicators is a dict or list of what you want. 
    We'll adapt from your script's get_technical_data.
    Returns a list of dicts, each having 'date' plus technical fields.
    """
    from_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    earlier_dt = from_date_dt - timedelta(days=buffer_days)
    earlier_str = earlier_dt.strftime("%Y-%m-%d")

    # 1) Fetch daily EOD so we know each trading day
    params = {
        'from': earlier_str,
        'to': end_date,
        'api_token': api_token,
        'fmt': 'json'
    }
    base_url = f"https://eodhd.com/api/eod/{ticker}?{urlencode(params)}"
    price_data = _fetch_json(base_url)

    days_dict = {}
    for row in price_data:
        d = row.get('date')
        if d:
            days_dict[d] = {'date': d}

    if not indicators:
        # default list
        indicators = ['sma','ema','rsi','macd','atr','stddev','bbands','beta']

    # Now fetch each indicator
    for ind in indicators:
        # Just an example of how to pass extra params
        # For a robust approach, you'd define a dict of period/etc. 
        # This is kept simple here.
        tech_params = {
            'function': ind,
            'from': earlier_str,
            'to': end_date,
            'api_token': api_token,
            'fmt': 'json',
            'order': 'd'
        }
        url = f"https://eodhd.com/api/technical/{ticker}?{urlencode(tech_params)}"
        ind_data = _fetch_json(url)

        for point in ind_data:
            dt = point.get('date')
            if not dt:
                continue
            if dt not in days_dict:
                days_dict[dt] = {'date': dt}

            # Then parse out the relevant fields
            if ind == 'macd':
                days_dict[dt]['macd'] = point.get('macd')
                days_dict[dt]['macd_signal'] = point.get('signal')
                days_dict[dt]['macd_histogram'] = point.get('divergence')
            elif ind == 'bbands':
                days_dict[dt]['bbands_upper'] = point.get('uband')
                days_dict[dt]['bbands_middle'] = point.get('mband')
                days_dict[dt]['bbands_lower'] = point.get('lband')
            else:
                # single-value
                val = point.get(ind)
                days_dict[dt][ind] = val

    # Convert to list, filter between [start_date, end_date], ascending
    merged_list = []
    for dt_str in sorted(days_dict.keys()):
        if start_date <= dt_str <= end_date:
            merged_list.append(days_dict[dt_str])
    return merged_list

# ---------- Macro, News, Events ----------

def fetch_news_sentiment(tickers, start_date, end_date):
    if not isinstance(tickers, list):
        tickers = [tickers]
    tickers_str = ','.join(tickers)
    url = (
        f"https://eodhd.com/api/sentiments?"
        f"s={tickers_str}&from={start_date}&to={end_date}"
        f"&api_token={EOD_API_KEY}&fmt=json"
    )
    return _fetch_json(url)

def fetch_economic_events(api_token, from_date, to_date, country="USA", limit=1000, offset=0):
    url = (
        f"https://eodhd.com/api/economic-events?api_token={api_token}"
        f"&from={from_date}&to={to_date}&country={country}&limit={limit}&offset={offset}"
    )
    return _fetch_json(url)

def fetch_macroeconomic_data():
    """
    Example: fetch 10Y bond or something
    """
    url = f"https://eodhd.com/api/eod/UK10Y.GBOND?api_token={EOD_API_KEY}&fmt=json"
    return _fetch_json(url)

# ---------- Intraday (if still needed) ----------

def get_intraday_data(ticker, start_date, end_date, interval, api_key):
    """
    Not always needed if you're doing daily approach with adjusted close.
    """
    import time
    from datetime import datetime

    start_ts = int(time.mktime(datetime.strptime(start_date, "%Y-%m-%d").timetuple()))
    end_ts   = int(time.mktime(datetime.strptime(end_date, "%Y-%m-%d").timetuple()))

    url = (
        f"https://eodhd.com/api/intraday/{ticker}?api_token={api_key}"
        f"&interval={interval}&from={start_ts}&to={end_ts}&fmt=json"
    )
    return _fetch_json(url)
