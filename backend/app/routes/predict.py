# stock_prediction_app/app/routes/predict.py



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

# Database
from db.database import save_to_rds

# ML libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

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
      3) Trains multiple tabular models (RF, GB, XGB, CatBoost, LightGBM).
      4) Ensembles, calculates metrics.
      5) Saves results to DB with save_to_rds().
      6) Returns final predictions & metrics as JSON.

    **LSTM is fully removed** from this route.
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
        news_sent   = fetch_news_sentiment(tickers, start_date, end_date)

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
                # 1) Intraday
                intraday_json = get_intraday_data(ticker, start_date, end_date, interval, EOD_API_KEY)
                if not intraday_json:
                    raise ValueError(f"No intraday data for {ticker}.")
                df_intraday = pd.DataFrame(intraday_json)
                if df_intraday.empty:
                    raise ValueError(f"Empty intraday data for {ticker}")

                # 2) Aggregate
                df_agg = aggregate_intraday(df_intraday)
                if df_agg.empty:
                    raise ValueError(f"Aggregated intraday is empty for {ticker}")

                # 3) Fundamentals
                fund_json = fetch_fundamental_data(ticker, EOD_API_KEY)
                df_fund   = process_fundamental_data(fund_json)
                if not df_fund.empty:
                    df_fund_rep = pd.concat([df_fund]*len(df_agg), ignore_index=True)
                    df_fund_rep["date"] = df_agg["date"].values
                else:
                    df_fund_rep = pd.DataFrame({"date": df_agg["date"]})

                # 4) Merge events + macro
                df_events_merged = process_economic_events(events_data, df_agg)
                df_macro_merged  = process_macroeconomic_data(macro_data, df_events_merged)

                # 5) Merge fundamentals
                df_macro_merged["date"] = pd.to_datetime(df_macro_merged["date"])
                df_fund_rep["date"]     = pd.to_datetime(df_fund_rep["date"])
                df_merged_fund = pd.merge(df_macro_merged, df_fund_rep, on="date", how="left")

                # 6) Merge news
                ticker_news = combined_news_df[combined_news_df["ticker"] == ticker].copy()
                if not ticker_news.empty and "normalized" in ticker_news.columns:
                    ticker_news["date"] = pd.to_datetime(ticker_news["date"])
                    news_agg = ticker_news.groupby(ticker_news["date"].dt.date)["normalized"].mean().reset_index()
                    news_agg.rename(columns={"normalized":"avg_news_sentiment","date":"merge_date"}, inplace=True)
                    news_agg["date"] = pd.to_datetime(news_agg["merge_date"])
                    news_agg.drop("merge_date", axis=1, inplace=True)
                    df_merged_fund = pd.merge(df_merged_fund, news_agg, on="date", how="left")
                else:
                    df_merged_fund["avg_news_sentiment"] = 0

                # Fill
                df_merged_fund.fillna(method="ffill", inplace=True)
                df_merged_fund.fillna(0, inplace=True)

                # 7) Next day close
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

                # 9) Tabular Models
                models = {
                    "RandomForest": RandomForestRegressor(random_state=42),
                    "GradientBoosting": GradientBoostingRegressor(random_state=42),
                    "XGBoost": XGBRegressor(random_state=42),
                    "CatBoost": CatBoostRegressor(verbose=0, random_state=42),
                    "LightGBM": LGBMRegressor(random_state=42),
                }
                preds_dict = {}
                for m_name, m in models.items():
                    m.fit(X_train, y_train)
                    preds_dict[m_name] = m.predict(X_test)

                # 10) Simple Ensemble
                stacked = np.column_stack(list(preds_dict.values()))
                ensemble_pred = stacked.mean(axis=1)

                # 11) Metrics
                mse_val = mean_squared_error(y_test, ensemble_pred)
                mae_val = mean_absolute_error(y_test, ensemble_pred)
                r2_val  = r2_score(y_test, ensemble_pred)

                # final record for DB
                record = {
                    "ticker": ticker,
                    "date": str(datetime.now().date()),
                    "rf_prediction": float(preds_dict["RandomForest"][-1]),
                    "gbm_prediction": float(preds_dict["GradientBoosting"][-1]),
                    "xgb_prediction": float(preds_dict["XGBoost"][-1]),
                    "catboost_prediction": float(preds_dict["CatBoost"][-1]),
                    "lightgbm_prediction": float(preds_dict["LightGBM"][-1]),
                    # remove LSTM column (None or just skip it)
                    "lstm_prediction": None,  # or remove from DB schema
                    "ensemble_prediction": float(ensemble_pred[-1]),
                    "mse": mse_val,
                    "mae": mae_val,
                    "r2_score": r2_val
                }

                # Save to DB
                save_to_rds([record])
                all_results.append(record)

            except Exception as e:
                current_app.logger.error(f"Error processing {ticker}: {e}", exc_info=True)
                return jsonify({"error": f"Error for {ticker}: {str(e)}"}), 500

        return jsonify(all_results), 200

    except Exception as e:
        current_app.logger.error(f"Top-level error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
