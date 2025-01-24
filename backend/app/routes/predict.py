import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from dotenv import load_dotenv
import time  # if needed by get_intraday_data

# Data fetching
from data.data_fetching import (
    fetch_fundamental_data,
    fetch_news_sentiment,
    fetch_economic_events,
    fetch_macroeconomic_data,
    get_intraday_data
)

# Data processing
from data.data_processing import (
    aggregate_intraday,
    process_fundamental_data,
    process_economic_events,
    process_macroeconomic_data
)

# Model training (for the create_sequences function)
from models.model_training import create_sequences

# Database
from db.database import save_to_rds

# ML libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

load_dotenv()
EOD_API_KEY = os.getenv('EOD_API_KEY')

predict_bp = Blueprint('predict', __name__)

@predict_bp.route("/predict", methods=["POST"])
def predict_next_close():
    """
    Full pipeline endpoint using:
      1) Intraday data (aggregated daily)
      2) Fundamentals
      3) News sentiment
      4) Economic events
      5) Macroeconomic data
      6) Sliding-window LSTM + tabular models + ensemble
      7) Saves predictions to DB
      8) Returns final JSON of predictions

    If ANY of these data sources are empty, we raise an exception or skip that ticker.
    """
    try:
        # 1) Parse request
        request_data = request.get_json()
        tickers = request_data.get("tickers", ["AAPL.US"])
        start_date = request_data.get("start_date", "2020-01-01")
        end_date = request_data.get("end_date", "2021-01-01")
        interval = request_data.get("interval", "1h")

        # 2) Fetch external data that doesn't depend on a specific ticker:
        #    Economic events (global or country-level)
        events_data = fetch_economic_events(EOD_API_KEY, start_date, end_date)
        #    Macroeconomic data (e.g., bond yields or other indicators)
        macro_data = fetch_macroeconomic_data()

        # 3) Fetch news sentiment for ALL tickers in one call
        #    This typically returns a dict { ticker_symbol: [ {article1}, {article2} ...], ...}
        raw_news_sent = fetch_news_sentiment(tickers, start_date, end_date)
        # Convert that to a single DataFrame so we can merge by date & ticker
        news_df_list = []
        if isinstance(raw_news_sent, dict):
            for tk, articles in raw_news_sent.items():
                df_tk = pd.DataFrame(articles)
                df_tk["ticker"] = tk
                news_df_list.append(df_tk)
        if not news_df_list:
            # If absolutely no news, we raise an error or proceed with empty DataFrame
            raise ValueError("No news sentiment data fetched for any ticker. Endpoint requires it.")
        combined_news_df = pd.concat(news_df_list, ignore_index=True)

        # We'll build up final results for each ticker
        all_results = []

        # 4) Process each ticker
        for ticker in tickers:
            try:
                # ========== A) Intraday ==============
                intraday_json = get_intraday_data(ticker, start_date, end_date, interval, EOD_API_KEY)
                if not intraday_json:
                    raise ValueError(f"No intraday data for {ticker}. Can't proceed.")
                df_intraday = pd.DataFrame(intraday_json)
                if df_intraday.empty:
                    raise ValueError(f"Intraday dataframe empty for {ticker}.")

                # Aggregate to daily
                df_daily = aggregate_intraday(df_intraday)
                if df_daily.empty:
                    raise ValueError(f"Aggregation resulted in empty data for {ticker}.")

                # ========== B) Fundamentals ==============
                fundamental_json = fetch_fundamental_data(ticker, EOD_API_KEY)
                df_fund = process_fundamental_data(fundamental_json)
                if df_fund.empty:
                    # If truly no fundamental data, you can decide to skip or fill with zeros
                    raise ValueError(f"Fundamental data is empty for {ticker}.")
                # Replicate the single-row fundamentals across each daily row
                df_fund_rep = pd.concat([df_fund]*len(df_daily), ignore_index=True)
                df_fund_rep["date"] = df_daily["date"].values

                # ========== C) News for THIS ticker ==============
                # Filter from the combined dataframe
                ticker_news = combined_news_df[combined_news_df["ticker"] == ticker].copy()
                if ticker_news.empty:
                    raise ValueError(f"No news sentiment found for {ticker} (but other tickers had some).")
                # Convert whatever date/time col we have into daily aggregates
                # Typically 'date' or 'datetime' or something
                # Let's assume there's a 'date' column
                # We'll group by date, compute mean of 'normalized' sentiment
                ticker_news["date"] = pd.to_datetime(ticker_news["date"]).dt.date
                news_agg = ticker_news.groupby("date")["normalized"].mean().reset_index()
                news_agg.rename(columns={"normalized": "avg_news_sentiment"}, inplace=True)

                # Convert the daily intraday date to datetime so merges work
                df_daily["date"] = pd.to_datetime(df_daily["date"])

                # ========== D) Merge economic events ==============
                df_events_merged = process_economic_events(events_data, df_daily)

                # ========== E) Merge macro data ==============
                df_macro_merged = process_macroeconomic_data(macro_data, df_events_merged)

                # ========== F) Merge fundamentals ==============
                df_macro_merged["date"] = pd.to_datetime(df_macro_merged["date"])
                df_fund_rep["date"] = pd.to_datetime(df_fund_rep["date"])
                df_merged_fund = pd.merge(df_macro_merged, df_fund_rep, on="date", how="left")

                # ========== G) Merge news sentiment ==============
                # Also ensure all date columns are datetime64
                news_agg["date"] = pd.to_datetime(news_agg["date"])
                df_final = pd.merge(df_merged_fund, news_agg, on="date", how="left")

                # Fill any remaining nulls
                df_final.fillna(method="ffill", inplace=True)
                df_final.fillna(0, inplace=True)

                if df_final.empty:
                    raise ValueError(f"Final merged DataFrame is empty for {ticker}.")

                # ========== H) Create target ==============
                df_final["target"] = df_final["intraday_close_mean"].shift(-1)
                df_final.dropna(subset=["target"], inplace=True)
                if df_final.empty:
                    raise ValueError(f"After shifting for target, no data left for {ticker}.")

                # ========== I) Prepare features ==============
                feature_cols = [
                    "intraday_vol_mean", "intraday_high_mean", "intraday_low_mean", "intraday_close_mean",
                    "macro_open", "macro_high", "macro_low", "macro_close",
                    "avg_importance", "event_count",
                    "avg_news_sentiment",  # from news
                    "market_cap", "pe_ratio", "beta"  # from fundamentals
                ]
                # Only keep columns that exist
                feature_cols = [c for c in feature_cols if c in df_final.columns]
                if not feature_cols:
                    raise ValueError(f"No valid features for {ticker} after merges.")

                X_all = df_final[feature_cols].values
                y_all = df_final["target"].values

                # Train-test split
                split_idx = int(len(df_final) * 0.8)
                X_train, X_test = X_all[:split_idx], X_all[split_idx:]
                y_train, y_test = y_all[:split_idx], y_all[split_idx:]

                if len(X_train) < 10:
                    raise ValueError(f"Not enough rows to train models for {ticker}.")

                # ========== J) Train 5 Tabular Models ==========
                models = {
                    "RandomForest": RandomForestRegressor(random_state=42),
                    "GradientBoosting": GradientBoostingRegressor(random_state=42),
                    "XGBoost": XGBRegressor(random_state=42),
                    "CatBoost": CatBoostRegressor(verbose=0, random_state=42),
                    "LightGBM": LGBMRegressor(random_state=42),
                }
                tabular_preds = {}
                for model_name, model in models.items():
                    model.fit(X_train, y_train)
                    tabular_preds[model_name] = model.predict(X_test)

                # ========== K) LSTM with create_sequences (sliding window) ==========
                timesteps = 5  # how many time steps per sequence
                X_seq_train, y_seq_train = create_sequences(X_train, y_train, timesteps)
                X_seq_test, y_seq_test = create_sequences(X_test, y_test, timesteps)

                if X_seq_train.size == 0 or X_seq_test.size == 0:
                    raise ValueError(f"Not enough data to create LSTM sequences for {ticker} (window={timesteps}).")

                lstm_model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(X_seq_train.shape[1], X_seq_train.shape[2])),
                    LSTM(50, return_sequences=False),
                    Dense(1)
                ])
                lstm_model.compile(optimizer="adam", loss="mse")
                lstm_model.fit(X_seq_train, y_seq_train, epochs=10, batch_size=32, verbose=0)
                lstm_preds = lstm_model.predict(X_seq_test).flatten()

                # Because the tabular preds are NOT using sliding windows, we must align:
                #   - We remove the first 'timesteps' from each tabular prediction so
                #     they match the shape of y_seq_test
                for m_name in tabular_preds:
                    tabular_preds[m_name] = tabular_preds[m_name][timesteps:]
                y_test_final = y_test[timesteps:]

                # ========== L) Ensemble ==========
                all_preds = list(tabular_preds.values()) + [lstm_preds]
                # shape => (# samples, # models)
                stacked = np.column_stack(all_preds)
                ensemble_preds = stacked.mean(axis=1)

                # Evaluate
                mse_val = mean_squared_error(y_test_final, ensemble_preds)
                mae_val = mean_absolute_error(y_test_final, ensemble_preds)
                r2_val  = r2_score(y_test_final, ensemble_preds)

                # Extract final predictions (the last point in the test set)
                final_ensemble = float(ensemble_preds[-1])
                final_actual   = float(y_test_final[-1])

                # ========== M) Build record to save & return ==========
                record = {
                    "ticker": ticker,
                    "date": str(datetime.now().date()),
                    "rf_prediction": float(tabular_preds["RandomForest"][-1]),
                    "gbm_prediction": float(tabular_preds["GradientBoosting"][-1]),
                    "xgb_prediction": float(tabular_preds["XGBoost"][-1]),
                    "catboost_prediction": float(tabular_preds["CatBoost"][-1]),
                    "lightgbm_prediction": float(tabular_preds["LightGBM"][-1]),
                    "lstm_prediction": float(lstm_preds[-1]),
                    "ensemble_prediction": final_ensemble,
                    "mse": mse_val,
                    "mae": mae_val,
                    "r2_score": r2_val
                }
                all_results.append(record)

                # ========== N) Save to DB ==========
                save_to_rds([record])

            except Exception as e:
                current_app.logger.error(f"Error processing ticker {ticker}: {e}", exc_info=True)
                return jsonify({"error": f"Error for {ticker}: {e}"}), 500

        # Return all ticker results
        return jsonify(all_results), 200

    except Exception as e:
        current_app.logger.error(f"Top-level error in /predict: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
