import os
import requests
import json
import time
from datetime import datetime, timedelta
from urllib.parse import urlencode
from dotenv import load_dotenv

load_dotenv()
EOD_API_KEY = os.getenv('EOD_API_KEY')

# Which indicators to fetch, plus default params
INDICATOR_CONFIGS = {
    'sma':        {'period': 20},
    'ema':        {'period': 20},
    'wma':        {'period': 20},
    'volatility': {'period': 20},
    'stochastic': {'k': 14, 'd': 3},
    'rsi':        {'period': 14},
    'stddev':     {'period': 20},
    'stochrsi':   {'period': 14},  # might return fast_k / fast_d
    'slope':      {'period': 20},
    'dmi':        {'period': 14},
    'adx':        {'period': 14},
    'macd':       {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
    'atr':        {'period': 14},
    'cci':        {'period': 20},
    'sar':        {},
    'beta':       {'period': 14},
    'bbands':     {'period': 20}
}

def fetch_json(url):
    """Helper to GET JSON with basic error handling."""
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    time.sleep(0.1)  # simple rate-limiting
    return data if isinstance(data, list) else []

def validate(value):
    """Convert None/False/'false' => None."""
    return None if value in [None, False, "false"] else value

def get_technical_data(ticker, start_date, end_date, api_key, buffer_days=20):
    """
    Fetch daily price data and multiple technical indicators from EODHD,
    merge into a single list of dicts (one per date), 
    and filter to the desired [start_date, end_date] range.
    """
    # 1) Calculate "earlier" start date so we have enough data for lookback periods
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    earlier_dt = start_dt - timedelta(days=buffer_days)
    earlier_str = earlier_dt.strftime("%Y-%m-%d")

    # 2) Fetch daily EOD data to know all trading days
    params = {
        'from': earlier_str, 'to': end_date,
        'api_token': api_key, 'fmt': 'json'
    }
    base_url = f'https://eodhd.com/api/eod/{ticker}?{urlencode(params)}'
    price_data = fetch_json(base_url)

    # Build a dict keyed by date, ensuring we have every trading date
    days_dict = {}
    for row in price_data:
        d = row.get('date')
        if d:
            days_dict[d] = {'date': d}

    # 3) For each indicator, fetch data and merge into days_dict
    for indicator, extra_params in INDICATOR_CONFIGS.items():
        tech_params = {
            'order': 'd',
            'from': earlier_str,
            'to': end_date,
            'function': indicator,
            'api_token': api_key,
            'fmt': 'json'
        }
        tech_params.update(extra_params)
        url = f'https://eodhd.com/api/technical/{ticker}?{urlencode(tech_params)}'
        ind_data = fetch_json(url)

        # Merge each data point by date
        for point in ind_data:
            d = point.get('date')
            if not d:
                continue
            # If date doesn't exist in days_dict, add it
            if d not in days_dict:
                days_dict[d] = {'date': d}

            # Handle multi-field indicators specially
            if indicator == 'macd':
                days_dict[d]['macd'] = validate(point.get('macd'))
                days_dict[d]['macd_signal'] = validate(point.get('signal'))
                days_dict[d]['macd_histogram'] = validate(point.get('divergence'))
            elif indicator == 'bbands':
                days_dict[d]['bbands_upper'] = validate(point.get('uband'))
                days_dict[d]['bbands_middle'] = validate(point.get('mband'))
                days_dict[d]['bbands_lower'] = validate(point.get('lband'))
            elif indicator == 'stochastic':
                days_dict[d]['stochastic_k'] = validate(point.get('k_values'))
                days_dict[d]['stochastic_d'] = validate(point.get('d_values'))
            elif indicator == 'stochrsi':
                # Adjust to the fields EODHD actually returns 
                days_dict[d]['stochrsi_k'] = validate(point.get('fast_k'))
                days_dict[d]['stochrsi_d'] = validate(point.get('fast_d'))
            else:
                # Single-value indicator
                val = validate(point.get(indicator))
                days_dict[d][indicator] = val

    # 4) Convert days_dict -> list, sorted desc, and filter to [start_date, end_date]
    merged_list = []
    for d in sorted(days_dict.keys(), reverse=True):
        if start_date <= d <= end_date:
            merged_list.append(days_dict[d])
    return merged_list


if __name__ == "__main__":
    # Example usage
    ticker = "AAPL.US"
    start_date = "2023-11-01"   # the true start you want
    end_date = "2024-01-01"
    buffer_days = 20           # how many extra days to go back for lookback

    final_data = get_technical_data(ticker, start_date, end_date, EOD_API_KEY, buffer_days)
    print(json.dumps(final_data, indent=2))
