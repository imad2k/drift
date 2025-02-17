# stock_prediction_app/app/routes/predict.py

import os
import json
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from flask import Blueprint, request, jsonify, current_app
from dotenv import load_dotenv

# Database
from db.database import save_model_performance, save_predictions

# Data fetching
from data.data_fetching import (
    fetch_daily_eod,
    fetch_technical_data,
    get_intraday_data,
    fetch_fundamental_data,
    fetch_economic_events,
    fetch_macroeconomic_data,
    fetch_news_sentiment
)

# Data processing
from data.data_processing import (
    merge_daily_and_technical,
    remove_zero_volume,
    remove_outliers,
    aggregate_intraday,
    process_fundamental_data,
    process_economic_events,
    process_macroeconomic_data,
)

# Model training
from models.model_training import train_tabular_ensemble

load_dotenv()
EOD_API_KEY = os.getenv("EOD_API_KEY")

predict_bp = Blueprint("predict", __name__)

def add_trading_days(start_date, days_to_add):
    """
    Adds 'days_to_add' trading days to start_date, skipping weekends.
    E.g., if start_date is Friday and days_to_add=1, we land on Monday.
    """
    proposed = start_date
    added = 0
    while added < days_to_add:
        proposed += timedelta(days=1)
        # Monday=0 ... Friday=4, Saturday=5, Sunday=6
        if proposed.weekday() < 5:
            added += 1
    return proposed


from flask import Blueprint, jsonify, request
from ..modules.prediction import run_predictions_for_ticker

predict_bp = Blueprint("predict", __name__)

@predict_bp.route("/predict", methods=["POST"])
def predict_multi_windows():
    """Route handler for predictions"""
    try:
        # Keep route handling logic here
        # Use run_predictions_for_ticker for core prediction logic
        req = request.get_json()

        # Basic request params
        tickers = req.get("tickers", ["AAPL.US"])
        data_type = req.get("data_type", "daily")  # "daily" or "intraday"
        end_date = req.get("end_date", date.today().strftime("%Y-%m-%d"))
        interval = req.get("interval", "1h")  # for intraday only

        # Rolling windows & horizons
        windows = req.get("windows", {
            "2y": 730,
            "1y": 365,
            "6m": 180,
            "3m": 90
        })
        horizons = req.get("horizons", {
            "1d": 1,   # 1 trading day
            "2d": 2,   # 2 trading days
            "1w": 5,   # 5 trading days
            "2w": 10,  # etc.
        })
        buffer_days = req.get("buffer_days", 20)

        all_results = []

        # For the biggest window
        max_window_days = max(windows.values())
        earliest_dt = date.today() - timedelta(days=max_window_days)
        earliest_str = earliest_dt.strftime("%Y-%m-%d")

        # Fetch global data: events & macro
        events_data = fetch_economic_events(EOD_API_KEY, earliest_str, end_date)
        macro_data  = fetch_macroeconomic_data()

        for ticker in tickers:
            for w_label, w_days in windows.items():
                # Window start date for fetching data
                w_start_dt = date.today() - timedelta(days=w_days)
                w_start_str = w_start_dt.strftime("%Y-%m-%d")

                # Fetch daily or intraday data
                if data_type == "daily":
                    eod_list = fetch_daily_eod(ticker, w_start_str, end_date, EOD_API_KEY)
                    tech_list = fetch_technical_data(
                        ticker, w_start_str, end_date, EOD_API_KEY,
                        buffer_days=buffer_days
                    )
                    df_merged = merge_daily_and_technical(eod_list, tech_list)
                    vol_col = "volume"
                    price_col = "adjusted_close"
                else:
                    intraday_json = get_intraday_data(ticker, w_start_str, end_date, interval, EOD_API_KEY)
                    if not intraday_json:
                        current_app.logger.warning(f"No intraday data for {ticker}, window={w_label}")
                        continue
                    df_int = pd.DataFrame(intraday_json)
                    if df_int.empty:
                        current_app.logger.warning(f"Intraday empty for {ticker}")
                        continue
                    df_merged = aggregate_intraday(df_int)
                    vol_col = "intraday_vol_mean"
                    price_col = "intraday_close_mean"

                if df_merged.empty:
                    current_app.logger.warning(f"No data after fetch/merge for {ticker}, window={w_label}")
                    continue

                # Fundamentals
                fund_json = fetch_fundamental_data(ticker, EOD_API_KEY)
                df_fund = process_fundamental_data(fund_json)
                if not df_fund.empty:
                    # Repeat fundamentals to match length of df_merged
                    df_fund_rep = pd.concat([df_fund]*len(df_merged), ignore_index=True)
                    df_fund_rep["date"] = df_merged["date"].values
                    df_merged = pd.merge(df_merged, df_fund_rep, on="date", how="left")

                # Add events + macro
                df_merged = process_economic_events(events_data, df_merged)
                df_merged = process_macroeconomic_data(macro_data, df_merged)

                # News
                news_res = fetch_news_sentiment([ticker], w_start_str, end_date)
                if isinstance(news_res, dict) and ticker in news_res:
                    articles = news_res[ticker]
                elif isinstance(news_res, list):
                    articles = news_res
                else:
                    articles = []
                if articles:
                    df_news = pd.DataFrame(articles)
                    df_news["date"] = pd.to_datetime(df_news["date"]).dt.date
                    agg = df_news.groupby("date")["normalized"].mean().reset_index()
                    agg.rename(columns={"normalized":"avg_news_sentiment","date":"merge_date"}, inplace=True)
                    df_merged["merge_date"] = pd.to_datetime(df_merged["date"]).dt.date
                    df_merged = pd.merge(df_merged, agg, on="merge_date", how="left")
                    df_merged.drop("merge_date", axis=1, inplace=True)
                    df_merged["avg_news_sentiment"].fillna(0, inplace=True)
                else:
                    df_merged["avg_news_sentiment"] = 0

                if df_merged.empty:
                    continue

                # Remove zero-volume & outliers
                if vol_col in df_merged.columns:
                    df_merged = remove_zero_volume(df_merged, vol_col=vol_col)
                outlier_cols = [price_col, vol_col, "sma","ema","rsi","macd","stddev","bbands_upper","bbands_lower"]
                outlier_cols = [c for c in outlier_cols if c in df_merged.columns]
                df_merged = remove_outliers(df_merged, outlier_cols)

                df_merged.sort_values("date", inplace=True)
                df_merged.reset_index(drop=True, inplace=True)
                if len(df_merged) < 10:
                    current_app.logger.warning(f"Not enough data after cleaning for {ticker}, window={w_label}")
                    continue

                # For each horizon
                for h_label, h_shift in horizons.items():
                    df_horizon = df_merged.copy()
                    if price_col not in df_horizon.columns:
                        current_app.logger.warning(f"No price col {price_col} for horizon {h_label}")
                        continue

                    # SHIFT target by h_shift
                    df_horizon["target"] = df_horizon[price_col].shift(-h_shift)
                    df_horizon.dropna(subset=["target"], inplace=True)
                    if df_horizon.empty:
                        current_app.logger.warning(f"No data after target shift for {ticker}, horizon={h_label}")
                        continue

                    # Feature selection
                    exclude_cols = {"date","timestamp","target",price_col,"merge_date"}
                    feature_cols = [c for c in df_horizon.columns if c not in exclude_cols]
                    X_all = df_horizon[feature_cols].values
                    y_all = df_horizon["target"].values

                    if len(X_all) < 20:
                        continue

                    # Train/test split
                    split_idx = int(len(X_all) * 0.8)
                    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
                    y_train, y_test = y_all[:split_idx], y_all[split_idx:]

                    if len(X_train) < 10 or len(X_test) < 5:
                        continue

                    # Train ensemble
                    preds_dict, ensemble_pred, metrics = train_tabular_ensemble(X_train, y_train, X_test, y_test)
                    mse_val = float(metrics["mse"])
                    mae_val = float(metrics["mae"])
                    r2_val  = float(metrics["r2"])

                    # Final test sample
                    last_idx = -1
                    actual_val = float(y_test[last_idx])

                    # ==============
                    # Forecast Date
                    # ==============
                    last_row_date = df_horizon.iloc[split_idx + last_idx]["date"]
                    # Always convert to Python date
                    base_date_dt = pd.to_datetime(last_row_date).date()

                    # Force base date to be at least "today"
                    if base_date_dt < date.today():
                        base_date_dt = date.today()

                    # Now add trading days
                    forecast_date = add_trading_days(base_date_dt, h_shift)

                    # Save performance & predictions
                    perf_records = []
                    pred_records = []

                    # model-by-model
                    for m_name, model_preds in preds_dict.items():
                        # MSE, MAE, R2
                        m_mse = float(np.mean((model_preds - y_test)**2))
                        m_mae = float(np.mean(np.abs(model_preds - y_test)))
                        ss_res = np.sum((model_preds - y_test)**2)
                        ss_tot = np.sum((y_test - np.mean(y_test))**2)
                        if abs(ss_tot) < 1e-12:
                            m_r2 = 0.0
                        else:
                            m_r2 = float(1 - ss_res / ss_tot)

                        # Insert model performance
                        perf_rec = {
                            "model_name": m_name,
                            "train_window_start": w_start_str,
                            "train_window_end": end_date,
                            "r2_score": m_r2,
                            "mse": m_mse,
                            "mae": m_mae,
                            "training_date": datetime.now(),
                            "ticker": ticker,
                            "horizon": h_label,
                            "window_label": w_label,
                            "data_type": data_type
                        }
                        perf_ids = save_model_performance([perf_rec])
                        if perf_ids:
                            mp_id = perf_ids[0]
                            final_val = float(model_preds[last_idx])
                            pct_err = None
                            if abs(actual_val) > 1e-12:
                                pct_err = 100.0*(final_val - actual_val)/actual_val

                            # Insert predictions
                            pr = {
                                "created_at": datetime.now(),
                                "forecast_date": forecast_date,
                                "forecast_horizon": h_label,
                                "ticker": ticker,
                                "predicted_value": final_val,
                                "model_name": m_name,
                                "model_performance_id": mp_id,
                                "data_type": data_type,
                                "actual_value": actual_val,
                                "pct_error": pct_err
                            }
                            pred_records.append(pr)

                    # Ensemble
                    ens_pct_err = None
                    if abs(actual_val) > 1e-12:
                        ens_pct_err = 100.0*(ensemble_pred[last_idx] - actual_val)/actual_val

                    perf_ens = {
                        "model_name": "Ensemble",
                        "train_window_start": w_start_str,
                        "train_window_end": end_date,
                        "r2_score": r2_val,
                        "mse": mse_val,
                        "mae": mae_val,
                        "training_date": datetime.now(),
                        "ticker": ticker,
                        "horizon": h_label,
                        "window_label": w_label,
                        "data_type": data_type
                    }
                    ens_ids = save_model_performance([perf_ens])
                    if ens_ids:
                        ens_id = ens_ids[0]
                        final_ens_val = float(ensemble_pred[last_idx])
                        pred_ens = {
                            "created_at": datetime.now(),
                            "forecast_date": forecast_date,
                            "forecast_horizon": h_label,
                            "ticker": ticker,
                            "predicted_value": final_ens_val,
                            "model_name": "Ensemble",
                            "model_performance_id": ens_id,
                            "data_type": data_type,
                            "actual_value": actual_val,
                            "pct_error": ens_pct_err
                        }
                        pred_records.append(pred_ens)

                    # store predictions in DB
                    if pred_records:
                        save_predictions(pred_records)

                    # add info to final results
                    run_info = {
                        "ticker": ticker,
                        "window": w_label,
                        "horizon": h_label,
                        "ensemble_mse": mse_val,
                        "ensemble_mae": mae_val,
                        "ensemble_r2": r2_val,
                        "final_ensemble_pred": float(ensemble_pred[last_idx]),
                        "actual_final_value": actual_val,
                        "ensemble_pct_err": ens_pct_err,
                        "forecast_date": str(forecast_date),
                        "data_type": data_type
                    }
                    all_results.append(run_info)

        return jsonify(all_results), 200

    except Exception as ex:
        current_app.logger.error(f"[ERROR] top-level in /predict: {ex}", exc_info=True)
        return jsonify({"error": str(ex)}), 500
