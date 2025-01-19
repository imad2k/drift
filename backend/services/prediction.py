import os
import json
import pandas as pd
import numpy as np

from datetime import datetime
from pprint import pprint


from openai import OpenAI
from dotenv import load_dotenv
from data_ingestion import get_fundamentals_data, get_historical_data



# Scikit-learn models and tools
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# XGBoost
from xgboost import XGBRegressor


# ------------------------------------------------------------------
# 1) ENVIRONMENT SETUP
# ------------------------------------------------------------------
load_dotenv()

EOD_API_KEY = os.getenv('EOD_API_KEY')
if EOD_API_KEY is None:
    raise ValueError("EOD_API_KEY is not set in environment variables")

OpenAI_API_KEY = os.getenv('OPENAI_API_KEY')
if OpenAI_API_KEY is None:
    raise ValueError("OpenAI_API_KEY is not set in environment variables")

openai = OpenAI(api_key=OpenAI_API_KEY)

# ------------------------------------------------------------------
# 2) HELPER FUNCTION: Train and Evaluate a Single Model w/ TSCV
# ------------------------------------------------------------------
def train_and_evaluate_ts(model, X, y, n_splits=3):
    """
    Performs time-series split (walk-forward) cross-validation on the given model.
    Returns the mean MSE across all splits, and leaves the model trained
    on the entire dataset at the end.
    
    Parameters:
    -----------
    model : regressor instance (e.g., RandomForestRegressor)
    X, y  : full dataset features and targets (same length)
    n_splits : how many splits for TimeSeriesSplit
    
    Returns:
    --------
    avg_mse : float
        Average MSE across all splits
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mses = []

    # Walk-forward validation
    for train_index, val_index in tscv.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        mses.append(mse)

    # Final training on full data
    model.fit(X, y)
    avg_mse = np.mean(mses)
    return avg_mse

# ------------------------------------------------------------------
# 3) MAIN PREDICTION FUNCTION
# ------------------------------------------------------------------
def predict_next_close(ticker, start_date, end_date):
    """
    1. Fetch and parse historical + fundamental data.
    2. Incorporate advanced feature engineering (lagged prices, returns, fundamentals).
    3. Use time-series cross-validation to evaluate each model's performance.
    4. Create a weighted ensemble of RandomForest, GradientBoosting, XGBoost based on MSE.
    5. Predict next day's close using the ensemble.
    6. (Optional) Provide a textual summary via OpenAI.
    """

    # -------------------------
    # A) GET HISTORICAL DATA
    # -------------------------
    historical_data_json = get_historical_data(ticker, start_date, end_date, EOD_API_KEY)
    historical_data = json.loads(historical_data_json)
    
    df = pd.DataFrame(historical_data)
    if df.empty:
        raise ValueError("No historical data returned for this ticker/date range.")

    # Ensure correct typing
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Basic check
    if len(df) < 30:
        raise ValueError("Not enough historical data to perform robust modeling.")

    # -------------------------
    # B) GET FUNDAMENTALS & PARSE
    # -------------------------
    fundamental_data_json = get_fundamentals_data(ticker)
    fundamental_data = json.loads(fundamental_data_json)

    # Extract some numeric fields. Adjust as needed:
    highlights = fundamental_data.get("Highlights", {})
    valuation  = fundamental_data.get("Valuation", {})
    technicals = fundamental_data.get("Technicals", {})
    
    market_cap = highlights.get("MarketCapitalization", np.nan)
    trailing_pe = valuation.get("TrailingPE", np.nan)
    beta = technicals.get("Beta", np.nan)

    # If you have quarterly or yearly data, you'd merge them time-wise,
    # but here's a single snapshot approach (same fundamental for all rows):
    df["market_cap"] = market_cap
    df["trailing_pe"] = trailing_pe
    df["beta"] = beta

    # One-hot or numeric encode sector (optional)
    sector = fundamental_data.get("Sector", "Unknown")
    df["sector_code"] = pd.Categorical([sector]*len(df)).codes

    # -------------------------
    # C) ADVANCED FEATURE ENGINEERING
    # -------------------------

    # 1) SHIFT the target: next day close as "target"
    #    We'll drop the last row after shifting to avoid NaN target
    df["target"] = df["close"].shift(-1)

    # 2) Basic lag features
    df["lag1_close"] = df["close"].shift(1)
    df["lag2_close"] = df["close"].shift(2)

    # 3) Returns: close-to-close % change
    df["returns"] = df["close"].pct_change()  # This row's daily return
    df["lag1_returns"] = df["returns"].shift(1)
    df["lag2_returns"] = df["returns"].shift(2)

    # 4) Drop any resulting NaNs
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Features we'll use:
    feature_cols = [
        "open", "high", "low", "close", "volume",
        "market_cap", "trailing_pe", "beta", "sector_code",
        "lag1_close", "lag2_close",
        "returns", "lag1_returns", "lag2_returns"
    ]

    X = df[feature_cols]
    y = df["target"]

    # We want to do a final day prediction. The last row in df is "today",
    # and we want to predict "tomorrow." We'll separate that out:
    # We'll train on everything except the last row. The last row is for inference.
    X_for_prediction = X.iloc[[-1]].copy()
    X_train_full = X.iloc[:-1]
    y_train_full = y.iloc[:-1]

    # -------------------------
    # D) TIME-SERIES SPLIT & MODEL TRAINING
    #    We'll do walk-forward cross-validation for each model,
    #    then do Weighted Ensemble based on inverse MSE.
    # -------------------------
    models = {
        "rf": RandomForestRegressor(n_estimators=50, random_state=42),
        "gbm": GradientBoostingRegressor(n_estimators=50, random_state=42),
        "xgb": XGBRegressor(n_estimators=50, random_state=42)
    }

    mse_scores = {}
    for name, model in models.items():
        avg_mse = train_and_evaluate_ts(model, X_train_full, y_train_full, n_splits=3)
        mse_scores[name] = avg_mse
    
    # Weighted Ensemble: weight = 1 / MSE
    weights = {}
    for name in models:
        weights[name] = 1.0 / mse_scores[name] if mse_scores[name] != 0 else 1.0

    total_weight = sum(weights.values())

    # Final training set -> each model is *already trained* on full data
    # after calling train_and_evaluate_ts, but let's confirm we re-fit just to be safe:
    for name, model in models.items():
        model.fit(X_train_full, y_train_full)

    # Make predictions for the last row
    preds = {}
    for name, model in models.items():
        preds[name] = model.predict(X_for_prediction)[0]

    # Weighted average
    ensemble_prediction = 0.0
    for name, pred_val in preds.items():
        ensemble_prediction += (pred_val * (weights[name] / total_weight))

    # -------------------------
    # E) OPTIONAL: CREATE A TEXTUAL SUMMARY WITH OpenAI
    # -------------------------
    summary_prompt = (
        f"Based on time-series cross-validation and a weighted ensemble of RF, GBM, and XGB, "
        f"the predicted next closing price for {ticker} is {ensemble_prediction:.2f}. "
        "Give a concise explanation of how this estimate was derived."
    )

    try:
        # "o1" is your custom model. If not available, try "gpt-3.5-turbo" or similar.
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
            f"OpenAI summary request failed: {e}\n"
            f"Prediction is {ensemble_prediction:.2f}."
        )

    # -------------------------
    # F) PRINT + RETURN RESULTS
    # -------------------------
    results = {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "ensemble_prediction": ensemble_prediction,
        "mse_scores": mse_scores,
        "weights": weights,
        "raw_model_preds": preds,
        "explanation": textual_explanation
    }

    pprint(results)
    return results


# -------------------------
# G) DRIVER CODE (INTERACTIVE)
# -------------------------
if __name__ == "__main__":
    ticker = input("Enter the ticker: ")
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")

    predict_next_close(ticker, start_date, end_date)