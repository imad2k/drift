import os
from openai import OpenAI
from dotenv import load_dotenv
from data_ingestion import get_fundamentals_data, get_historical_data
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pprint import pprint
# Scikit-learn ensemble regressors
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# XGBoost
from xgboost import XGBRegressor

# Load the API key from the .env file
load_dotenv()

# Get the EOD API key from the environment variables
EOD_API_KEY = os.getenv('EOD_API_KEY')
if EOD_API_KEY is None:
    raise ValueError("EOD_API_KEY is not set in the environment variables")

# Get the OpenAI API key from the environment variables
OpenAI_API_KEY = os.getenv('OPENAI_API_KEY')
if OpenAI_API_KEY is None:
    raise ValueError("OpenAI_API_KEY is not set in the environment variables")

# Create an instance of the OpenAI
openai = OpenAI(api_key=OpenAI_API_KEY)

def predict_next_close(ticker, start_date, end_date):
    """
    Demonstration of an ensemble-based approach to predict next day's closing price.
    1. Fetch and parse historical data
    2. Build and train 3 ensemble models (RF, GBM, XGBoost)
    3. Average their predictions for a final forecast
    4. [Optional] Provide a textual summary via OpenAI
    """
    # -------------------------
    # 1) GET HISTORICAL DATA
    # -------------------------
    historical_data_json = get_historical_data(ticker, start_date, end_date, EOD_API_KEY)
    historical_data = json.loads(historical_data_json)

    # Convert to DataFrame
    df = pd.DataFrame(historical_data)
    # Ensure correct types and sort by date
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)

    # Basic sanity check
    if len(df) < 5:
        raise ValueError("Not enough historical data to train a model.")

    # -------------------------
    # 2) [OPTIONAL] GET FUNDAMENTALS
    #    (Demonstration of how you might incorporate them)
    # -------------------------
    fundamental_data_json = get_fundamentals_data(ticker)
    fundamental_data = json.loads(fundamental_data_json)

    # -------------------------
    # 3) BUILD AND TRAIN MODELS
    # -------------------------
    # Feature engineering
    df["target"] = df["close"].shift(-1)
    df.dropna(inplace=True)

    X = df[["open", "high", "low", "close", "volume"]]
    y = df["target"]

    # Split data into training and testing sets
    split_index = int(len(df) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Initialize models
    rf = RandomForestRegressor()
    gbm = GradientBoostingRegressor()
    xgb = XGBRegressor()

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
        f"Based on the historical data from {start_date} to {end_date} for {ticker}, "
        f"the predicted next day's closing price is {final_prediction:.2f}. "
        "Provide a short explanation."
    )

    # Note: Adjust to your needs. Also ensure you have your OpenAI API key.
    try:
        response = openai.chat.completions.create(
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

    # Print (or log) both the numeric prediction and textual summary
    pprint({
        "ensemble_prediction": final_prediction,
        "explanation": textual_explanation
    })

    # Return numeric prediction or a structured result
    return {
        "prediction": final_prediction,
        "explanation": textual_explanation
    }

# Get user input
ticker = input("Enter the ticker: ")
start_date = input("Enter the start date in the format YYYY-MM-DD: ")
end_date = input("Enter the end date in the format YYYY-MM-DD: ")
predict_next_close(ticker, start_date, end_date)