import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from openai import OpenAI
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Adjust these imports to match your actual file structure
from data_ingestion import get_daily_data, get_intraday_data


# Load environment variables
load_dotenv()
EOD_API_KEY = os.getenv('EOD_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not EOD_API_KEY:
    raise ValueError("EOD_API_KEY is not set in environment variables.")


def aggregate_intraday(intraday_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert intraday bars (datetime, OHLC, volume) into daily-level stats.
    Example: mean intraday volume, mean high, mean low, mean close.
    """
    # Parse the 'datetime' column
    intraday_df["datetime"] = pd.to_datetime(intraday_df["datetime"], errors="coerce")
    intraday_df.dropna(subset=["datetime"], inplace=True)

    # Group by date_only
    intraday_df["date_only"] = intraday_df["datetime"].dt.date
    grouped = intraday_df.groupby("date_only")

    # Aggregate
    agg_df = grouped.agg({
        'volume': 'mean',
        'high': 'mean',
        'low': 'mean',
        'close': 'mean'
    }).reset_index()

    # Rename
    agg_df.rename(columns={
        'volume': 'intraday_vol_mean',
        'high': 'intraday_high_mean',
        'low': 'intraday_low_mean',
        'close': 'intraday_close_mean',
        'date_only': 'intraday_date'
    }, inplace=True)

    return agg_df


def predict_next_close(ticker: str, start_date: str, end_date: str, intraday_interval: str = "1h") -> str:
    """
    1. Fetch daily data & intraday data for the given ticker & date range.
    2. Aggregate intraday to daily-level features; merge with daily.
    3. Shift next-day close as 'target'; train RandomForest, GradientBoosting, XGBoost.
    4. Predict final test sample's next close.
    5. Generate a textual explanation via OpenAI (if OPENAI_API_KEY is set).
    6. Return ONLY the textual explanation (or fallback string if something fails).
    """

    # 1) Fetch daily data
    daily_json = get_daily_data(ticker, start_date, end_date, EOD_API_KEY)
    daily_data = json.loads(daily_json)
    df_daily = pd.DataFrame(daily_data)

    if df_daily.empty:
        return "No daily data found for this date range."

    # Clean daily data
    df_daily["date"] = pd.to_datetime(df_daily["date"], errors="coerce")
    df_daily.dropna(subset=["date"], inplace=True)
    df_daily.sort_values("date", inplace=True)
    df_daily.reset_index(drop=True, inplace=True)

    # 2) Fetch intraday data & merge
    intraday_json = get_intraday_data(ticker, start_date, end_date, intraday_interval, EOD_API_KEY)
    intraday_data = json.loads(intraday_json)
    df_intraday = pd.DataFrame(intraday_data)

    if not df_intraday.empty:
        df_agg = aggregate_intraday(df_intraday)
        df_agg["intraday_date"] = pd.to_datetime(df_agg["intraday_date"])
        df_daily["daily_date_only"] = df_daily["date"].dt.date
        df_agg["agg_date_only"] = df_agg["intraday_date"].dt.date

        df_merged = pd.merge(
            df_daily,
            df_agg,
            how="left",
            left_on="daily_date_only",
            right_on="agg_date_only",
            suffixes=("", "_intraday")
        )
        df_merged.drop(columns=["daily_date_only", "agg_date_only"], errors="ignore", inplace=True)
    else:
        # If intraday is empty, still proceed with daily data
        df_merged = df_daily.copy()
        df_merged["intraday_vol_mean"] = np.nan
        df_merged["intraday_high_mean"] = np.nan
        df_merged["intraday_low_mean"] = np.nan
        df_merged["intraday_close_mean"] = np.nan

    # Fill intraday columns with 0 to avoid NaNs for GradientBoosting
    fill_cols = ["intraday_vol_mean", "intraday_high_mean", "intraday_low_mean", "intraday_close_mean"]
    for col in fill_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(0)  # direct assignment to avoid chained assignment

    # 3) Shift next-day close as target
    df_merged["target"] = df_merged["close"].shift(-1)
    df_merged.dropna(subset=["target"], inplace=True)
    df_merged.reset_index(drop=True, inplace=True)

    # Features
    feature_cols = [
        "open", "high", "low", "close", "volume",
        "intraday_vol_mean", "intraday_high_mean",
        "intraday_low_mean", "intraday_close_mean"
    ]
    feature_cols = [c for c in feature_cols if c in df_merged.columns]

    X = df_merged[feature_cols]
    y = df_merged["target"]

    # Train/test split
    data_len = len(df_merged)
    split_index = int(data_len * 0.8)
    if split_index < 1:
        return "Not enough data to perform a train/test split."

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    if X_train.empty or X_test.empty:
        return "Training or testing data is empty. Check your date range or data coverage."

    # 4) Train 3 ensemble models
    rf = RandomForestRegressor(random_state=42)
    gbm = GradientBoostingRegressor(random_state=42)
    xgb = XGBRegressor(random_state=42)

    rf.fit(X_train, y_train)
    gbm.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    gbm_pred = gbm.predict(X_test)
    xgb_pred = xgb.predict(X_test)

    if len(rf_pred) == 0:
        return "No predictions available from the test set."

    # We'll pick the last row in the test set as the "next day"
    final_prediction = np.mean([rf_pred[-1], gbm_pred[-1], xgb_pred[-1]])

    # 5) Generate textual explanation (OpenAI) or fallback message
    if not OPENAI_API_KEY:
        return f"Predicted next close: {final_prediction:.2f} (No OpenAI key provided)."

    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = (
            f"Based on historical daily and intraday data from {start_date} to {end_date}, "
            f"the predicted next closing price for {ticker} is {final_prediction:.2f}. "
            "Provide a short, concise explanation, limited to a few sentences."
        )
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a concise financial assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        explanation = response.choices[0].message.content
        return explanation

    except Exception as e:
        return f"OpenAI request failed: {e}"


if __name__ == "__main__":
    ticker = input("Enter the ticker symbol (e.g., 'AAPL.US'): ")
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")
    interval = input("Enter intraday interval (e.g., '5m', '15m', '1h'): ")

    # Run the pipeline and print the final OpenAI explanation (or fallback string)
    final_explanation = predict_next_close(ticker, start_date, end_date, intraday_interval=interval)
    print(final_explanation)
