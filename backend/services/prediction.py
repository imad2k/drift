import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pprint import pprint
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from data_ingestion import get_intraday_data
import sklearn

# Print scikit-learn version
print("scikit-learn version:", sklearn.__version__)

# Load the API key from the .env file
load_dotenv()

# Get the API key from the environment variables
EOD_API_KEY = os.getenv('EOD_API_KEY')
if EOD_API_KEY is None:
    raise ValueError("EOD_API_KEY is not set in the environment variables")

# ===========================
# 1) HELPER: Aggregate Intraday
#    Stats into Daily Frame
# ===========================
def aggregate_intraday(intraday_df):
    """
    Given a DataFrame of intraday bars (timestamp, open, high, low, close, volume),
    return aggregated stats per date. For example:
      - average intraday volume
      - day's high-low range average
      - etc.
    """
    intraday_df["timestamp"] = pd.to_datetime(intraday_df["timestamp"], unit='s')
    intraday_df["date"] = intraday_df["timestamp"].dt.date  # group by day

    # Group by date
    grouped = intraday_df.groupby("date")
    # Example aggregated features:
    agg_df = grouped.agg({
        'volume': 'mean',            # average intraday volume
        'high': 'mean',              # average of intraday highs
        'low': 'mean',               # average of intraday lows
        'close': 'mean'              # average of intraday close
    }).reset_index()

    # Rename columns to indicate these are intraday stats
    agg_df.rename(columns={
        'volume': 'intraday_vol_mean',
        'high': 'intraday_high_mean',
        'low': 'intraday_low_mean',
        'close': 'intraday_close_mean'
    }, inplace=True)

    return agg_df

def predict_next_close(ticker, start_date, end_date, intraday_interval):
    """
    Demonstration of an ensemble-based approach to predict next day's closing price.
    1. Fetch and parse intraday data
    2. Aggregate intraday data to daily statistics
    3. Build and train 3 ensemble models (RF, GBM, XGBoost)
    4. Average their predictions for a final forecast
    5. [Optional] Provide a textual summary via OpenAI
    """
    # -------------------------
    # 1) GET INTRADAY DATA
    # -------------------------
    intraday_json = get_intraday_data(ticker, start_date, end_date, intraday_interval, EOD_API_KEY)
    intraday_data = json.loads(intraday_json)
    df_intraday = pd.DataFrame(intraday_data)

    if df_intraday.empty:
        raise ValueError("No intraday data found for the specified range.")

    # -------------------------
    # 2) AGGREGATE INTRADAY DATA
    # -------------------------
    df_agg_intraday = aggregate_intraday(df_intraday)

    # -------------------------
    # 3) BUILD AND TRAIN MODELS
    # -------------------------
    # Feature engineering
    df_agg_intraday["target"] = df_agg_intraday["intraday_close_mean"].shift(-1)
    df_agg_intraday.dropna(inplace=True)

    X = df_agg_intraday[["intraday_vol_mean", "intraday_high_mean", "intraday_low_mean", "intraday_close_mean"]]
    y = df_agg_intraday["target"]

    # Split data into training and testing sets
    split_index = int(len(df_agg_intraday) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Initialize models
    rf = RandomForestRegressor(random_state=42)
    gbm = GradientBoostingRegressor(random_state=42)
    xgb = XGBRegressor(random_state=42)

    # Train models
    rf.fit(X_train, y_train)
    gbm.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    # Predict next day's close
    rf_pred = rf.predict(X_test)
    gbm_pred = gbm.predict(X_test)
    xgb_pred = xgb.predict(X_test)

    # Average predictions
    final_prediction = np.mean([rf_pred[-1], gbm_pred[-1], xgb_pred[-1]])

    # -------------------------
    # 4) [OPTIONAL] PROVIDE TEXTUAL SUMMARY VIA OPENAI
    # -------------------------
    summary_prompt = (
        f"Based on the historical intraday data from {start_date} to {end_date} for {ticker}, "
        f"the predicted next day's closing price is {final_prediction:.2f}. "
        "Provide a short explanation."
    )

    # Note: Adjust to your needs. Also ensure you have your OpenAI API key.
    try:
        response = openai.chat.completion.create(
            model="o1",
            messages=[
                {"role": "system", "content": "You are a concise financial assistant."},
                {"role": "user", "content": summary_prompt}
            ]
        )
        textual_explanation = response.choices[0].message.content
    except Exception as e:
        textual_explanation = (
            f"Could not generate summary from OpenAI: {e}\n"
            f"Final predicted price is {final_prediction:.2f}."
        )

    # Return numeric prediction or a structured result
    return {
        "rf_prediction": rf_pred[-1],
        "gbm_prediction": gbm_pred[-1],
        "xgb_prediction": xgb_pred[-1],
        "ensemble_prediction": final_prediction,
        "explanation": textual_explanation
    }

# Get user input
ticker = input("Enter the ticker symbol (e.g., 'AAPL.US'): ")
start_date = input("Enter the start date (YYYY-MM-DD): ")
end_date = input("Enter the end date (YYYY-MM-DD): ")
interval = input("Enter intraday interval (e.g., '5m', '15m', '1h'): ")

try:
    result = predict_next_close(ticker, start_date, end_date, intraday_interval=interval)
    print("Prediction Result:")
    pprint(result)
except Exception as e:
    print(f"Error: {e}")
