# stock_prediction_app/data/prepare_data.py

import os
import pandas as pd
import numpy as np

from data.data_fetching import (
    fetch_fundamental_data,
    fetch_news_sentiment,
    fetch_economic_events,
    fetch_macroeconomic_data,
    get_intraday_data
)

from data.data_processing import (
    aggregate_intraday,
    process_fundamental_data,
    process_economic_events,
    process_macroeconomic_data,
)

from dotenv import load_dotenv

# Load environment variables
load_dotenv()
EOD_API_KEY = os.getenv('EOD_API_KEY')

def prepare_merged_fund_data(ticker="AAPL.US", start_date="2021-01-01", end_date="2025-01-23", interval="1h"):
    """
    High-level function that:
      1) Fetches intraday data, fundamentals, macro, events, news for `ticker`.
      2) Aggregates & merges everything into a final DataFrame.
      3) Creates a 'target' column by shifting 'intraday_close_mean' by -1 day.
      4) Returns the final DataFrame (df_merged_fund).
    """
    if not EOD_API_KEY:
        raise ValueError("EOD_API_KEY environment variable not set")
        
    # 1) Intraday data
    intraday_json = get_intraday_data(ticker, start_date, end_date, interval, EOD_API_KEY)
    if not intraday_json:
        print(f"No intraday data returned for {ticker}")
        return pd.DataFrame()  # empty means we can't proceed

    df_intraday = pd.DataFrame(intraday_json)
    if df_intraday.empty:
        print(f"Empty intraday DataFrame for {ticker}")
        return pd.DataFrame()

    # 2) Aggregate intraday
    df_agg = aggregate_intraday(df_intraday)
    if df_agg.empty:
        print(f"Aggregated intraday is empty for {ticker}")
        return pd.DataFrame()

    # 3) Fundamentals
    fund_json = fetch_fundamental_data(ticker, EOD_API_KEY)
    df_fund = process_fundamental_data(fund_json)
    if not df_fund.empty:
        df_fund_rep = pd.concat([df_fund]*len(df_agg), ignore_index=True)
        df_fund_rep["date"] = df_agg["date"].values
    else:
        df_fund_rep = pd.DataFrame({"date": df_agg["date"]})

    # 4) Macro + Events
    events_data = fetch_economic_events(EOD_API_KEY, start_date, end_date)
    macro_data  = fetch_macroeconomic_data()

    df_events_merged = process_economic_events(events_data, df_agg)
    df_macro_merged  = process_macroeconomic_data(macro_data, df_events_merged)

    # 5) Merge fundamentals
    df_macro_merged["date"] = pd.to_datetime(df_macro_merged["date"])
    df_fund_rep["date"]     = pd.to_datetime(df_fund_rep["date"])
    df_merged_fund = pd.merge(df_macro_merged, df_fund_rep, on="date", how="left")

    # 6) News
    # Single ticker, so fetch only for that ticker
    news_dict = fetch_news_sentiment([ticker], start_date, end_date)
    if isinstance(news_dict, dict):
        articles = news_dict.get(ticker, [])
        if articles:
            news_df = pd.DataFrame(articles)
            if not news_df.empty and "normalized" in news_df.columns:
                news_df["date"] = pd.to_datetime(news_df["date"])
                news_agg = news_df.groupby(news_df["date"].dt.date)["normalized"].mean().reset_index()
                news_agg.rename(columns={"normalized":"avg_news_sentiment","date":"merge_date"}, inplace=True)
                news_agg["date"] = pd.to_datetime(news_agg["merge_date"])
                news_agg.drop("merge_date", axis=1, inplace=True)
                df_merged_fund = pd.merge(df_merged_fund, news_agg, on="date", how="left")
            else:
                df_merged_fund["avg_news_sentiment"] = 0
        else:
            df_merged_fund["avg_news_sentiment"] = 0

    # 7) Fill NAs
    df_merged_fund.fillna(method="ffill", inplace=True)
    df_merged_fund.fillna(0, inplace=True)

    # 8) Shift target
    if "intraday_close_mean" not in df_merged_fund.columns:
        print(f"No 'intraday_close_mean' found in merged data for {ticker}")
        return pd.DataFrame()

    df_merged_fund["target"] = df_merged_fund["intraday_close_mean"].shift(-1)
    df_merged_fund.dropna(subset=["target"], inplace=True)

    return df_merged_fund
