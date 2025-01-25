import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from dotenv import load_dotenv

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
    process_macroeconomic_data,
)

# Model training utils
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
    Endpoint that:
      1) Fetches intraday data, fundamentals, news, events, macro.
      2) Aggregates & merges everything.
      3) Trains multiple models (RF, GB, XGB, CatBoost, LightGBM, LSTM).
      4) Ensembles, calculates metrics.
      5) Saves results to DB with save_to_rds().
      6) Returns final predictions & metrics as JSON.
    """
    try:
        request_data = request.get_json()
        tickers     = request_data.get("tickers", ["AAPL.US"])
        start_date  = request_data.get("start_date", "2020-01-01")
        end_date    = request_data.get("end_date", "2025-01-01")
        interval    = request_data.get("interval", "1h")

        # --------------------
        # Global fetch: events, macro, news
        # --------------------
        events_data = fetch_economic_events(EOD_API_KEY, start_date, end_date)
        macro_data  = fetch_macroeconomic_data()
        news_sent   = fetch_news_sentiment(tickers, start_date, end_date)  # if it returns a dict of ticker->articles

        # Convert raw news to a combined DF
        news_df_list = []
        if isinstance(news_sent, dict):
            for tk, articles in news_sent.items():
                df_tk = pd.DataFrame(articles)
                df_tk["ticker"] = tk
                news_df_list.append(df_tk)
        combined_news_df = pd.concat(news_df_list, ignore_index=True) if news_df_list else pd.DataFrame()

        all_results = []

        for ticker in tickers:
            try:
                # 1) Intraday data
                intraday_json = get_intraday_data(ticker, start_date, end_date, interval, EOD_API_KEY)
                if not intraday_json:
                    raise ValueError(f"No intraday data for {ticker}.")
                df_intraday = pd.DataFrame(intraday_json)
                if df_intraday.empty:
                    raise ValueError(f"Empty intraday data for {ticker}")

                # 2) Aggregate intraday to daily
                df_agg = aggregate_intraday(df_intraday)
                if df_agg.empty:
                    raise ValueError(f"Aggregated intraday is empty for {ticker}")

                # 3) Fundamentals
                fund_json = fetch_fundamental_data(ticker, EOD_API_KEY)
                df_fund   = process_fundamental_data(fund_json)
                if not df_fund.empty:
                    # replicate across each date row
                    df_fund_rep = pd.concat([df_fund]*len(df_agg), ignore_index=True)
                    df_fund_rep["date"] = df_agg["date"].values
                else:
                    # create a placeholder so merges won't fail
                    df_fund_rep = pd.DataFrame({"date": df_agg["date"]})

                # 4) Merge events + macro with aggregated intraday
                df_events_merged = process_economic_events(events_data, df_agg)
                df_macro_merged  = process_macroeconomic_data(macro_data, df_events_merged)

                # 5) Merge fundamentals
                df_macro_merged["date"] = pd.to_datetime(df_macro_merged["date"])
                df_fund_rep["date"]     = pd.to_datetime(df_fund_rep["date"])
                df_merged_fund = pd.merge(df_macro_merged, df_fund_rep, on="date", how="left")

                # 6) Merge news for this ticker
                ticker_news = combined_news_df[combined_news_df["ticker"] == ticker].copy()
                if not ticker_news.empty and "normalized" in ticker_news.columns:
                    ticker_news["date"] = pd.to_datetime(ticker_news["date"])
                    news_agg = ticker_news.groupby(ticker_news["date"].dt.date)["normalized"].mean().reset_index()
                    news_agg.rename(columns={"normalized":"avg_news_sentiment","date":"merge_date"}, inplace=True)
                    # reintroduce date as datetime
                    news_agg["date"] = pd.to_datetime(news_agg["merge_date"])
                    news_agg.drop("merge_date", axis=1, inplace=True)
                    df_merged_fund = pd.merge(df_merged_fund, news_agg, on="date", how="left")
                else:
                    # No news or no 'normalized' column
                    df_merged_fund["avg_news_sentiment"] = 0

                # fill nulls
                df_merged_fund.fillna(method="ffill", inplace=True)
                df_merged_fund.fillna(0, inplace=True)

                # 7) Create target (next day close)
                df_merged_fund["target"] = df_merged_fund["intraday_close_mean"].shift(-1)
                df_merged_fund.dropna(subset=["target"], inplace=True)
                if df_merged_fund.empty:
                    raise ValueError(f"No data after target shift for {ticker}")

                # 8) Feature selection
                feature_cols = [
                    "intraday_vol_mean","intraday_high_mean","intraday_low_mean","intraday_close_mean",
                    "macro_open","macro_high","macro_low","macro_close",
                    "avg_importance","event_count","avg_news_sentiment",
                    "market_cap","pe_ratio","beta"
                ]
                feature_cols = [c for c in feature_cols if c in df_merged_fund.columns]

                X_all = df_merged_fund[feature_cols].values
                y_all = df_merged_fund["target"].values

                # Train-test split
                split_idx = int(len(df_merged_fund)*0.8)
                X_train, X_test = X_all[:split_idx], X_all[split_idx:]
                y_train, y_test = y_all[:split_idx], y_all[split_idx:]

                if len(X_train) < 10:
                    raise ValueError(f"Not enough rows to train for {ticker}")

                # 9) Train multiple tabular models
                models = {
                    "RandomForest": RandomForestRegressor(random_state=42),
                    "GradientBoosting": GradientBoostingRegressor(random_state=42),
                    "XGBoost": XGBRegressor(random_state=42),
                    "CatBoost": CatBoostRegressor(verbose=0, random_state=42),
                    "LightGBM": LGBMRegressor(random_state=42),
                }
                tabular_preds = {}
                for m_name, m in models.items():
                    m.fit(X_train, y_train)
                    tabular_preds[m_name] = m.predict(X_test)

                # 10) LSTM
                timesteps = 5
                def create_seq(data_arr, target_arr, ts):
                    Xs, ys = [], []
                    for i in range(len(data_arr)-ts):
                        Xs.append(data_arr[i:i+ts])
                        ys.append(target_arr[i+ts])
                    return np.array(Xs), np.array(ys)

                X_seq_train, y_seq_train = create_seq(X_train, y_train, timesteps)
                X_seq_test,  y_seq_test  = create_seq(X_test,  y_test,  timesteps)

                if len(X_seq_train) and len(X_seq_test):
                    lstm_model = Sequential([
                        LSTM(50, return_sequences=True, input_shape=(timesteps, X_seq_train.shape[2])),
                        LSTM(50, return_sequences=False),
                        Dense(1)
                    ])
                    lstm_model.compile(optimizer="adam", loss="mse")
                    lstm_model.fit(X_seq_train, y_seq_train, epochs=10, batch_size=32, verbose=0)
                    lstm_preds = lstm_model.predict(X_seq_test).flatten()
                else:
                    lstm_preds = np.zeros(len(X_test)-timesteps)

                # Align tabular predictions with LSTM length
                for nm in tabular_preds:
                    tabular_preds[nm] = tabular_preds[nm][timesteps:]
                y_test_final = y_test[timesteps:]

                # 11) Ensemble
                model_arrays = list(tabular_preds.values()) + [lstm_preds]
                stacked = np.column_stack(model_arrays)
                ensemble_pred = stacked.mean(axis=1)

                # 12) Metrics
                mse_val = mean_squared_error(y_test_final, ensemble_pred)
                mae_val = mean_absolute_error(y_test_final, ensemble_pred)
                r2_val  = r2_score(y_test_final, ensemble_pred)

                # Final record
                record = {
                    "ticker": ticker,
                    "date": str(datetime.now().date()),
                    "rf_prediction": float(tabular_preds["RandomForest"][-1]),
                    "gbm_prediction": float(tabular_preds["GradientBoosting"][-1]),
                    "xgb_prediction": float(tabular_preds["XGBoost"][-1]),
                    "catboost_prediction": float(tabular_preds["CatBoost"][-1]),
                    "lightgbm_prediction": float(tabular_preds["LightGBM"][-1]),
                    "lstm_prediction": float(lstm_preds[-1]),
                    "ensemble_prediction": float(ensemble_pred[-1]),
                    "mse": mse_val,
                    "mae": mae_val,
                    "r2_score": r2_val
                }

                # 13) Save to DB
                save_to_rds([record])

                # 14) Append to results
                # If you prefer to see all predictions, store them in the JSON
                all_results.append(record)

            except Exception as e:
                current_app.logger.error(f"Error processing {ticker}: {e}", exc_info=True)
                return jsonify({"error": f"Error for {ticker}: {str(e)}"}), 500

        return jsonify(all_results), 200

    except Exception as e:
        current_app.logger.error(f"Top-level error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
