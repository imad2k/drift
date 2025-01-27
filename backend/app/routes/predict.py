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

@predict_bp.route("/predict", methods=["POST"])
def predict_multi_windows():
    """
    A polished endpoint that:
      1) Reads JSON params to decide daily vs. intraday
      2) Loops over multiple rolling windows + horizons
      3) Fetches & merges data
      4) Removes outliers & zero-volume rows
      5) Trains multiple tabular models + ensemble
      6) Saves each model's performance & final prediction
      7) Returns JSON summary
    """

    try:
        req = request.get_json()

        # Required / optional fields
        tickers = req.get("tickers", ["AAPL.US"])
        data_type = req.get("data_type", "daily")  # "daily" or "intraday"
        start_date = req.get("start_date")  # might not be used if we're doing rolling
        end_date = req.get("end_date", date.today().strftime("%Y-%m-%d"))
        interval = req.get("interval", "1h")  # for intraday
        # Optional user-provided or we define them:
        windows = req.get("windows", {
            "2y": 730,
            "1y": 365,
            "6m": 180,
            "3m": 90
        })
        horizons = req.get("horizons", {
            "1d": 1,
            "1w": 5,
            "2w": 10,
            "4w": 20
        })
        buffer_days = req.get("buffer_days", 20)  # for daily technical fetch

        all_results = []

        # Fetch macro & events once for the broadest date needed
        max_window_days = max(windows.values())
        earliest_dt = date.today() - timedelta(days=max_window_days)
        earliest_str = earliest_dt.strftime("%Y-%m-%d")

        events_data = fetch_economic_events(EOD_API_KEY, earliest_str, end_date)
        macro_data  = fetch_macroeconomic_data()

        for ticker in tickers:
            # --------------
            # MULTI-WINDOW
            # --------------
            for w_label, w_days in windows.items():
                # Compute rolling start_date from "today - w_days"
                w_start_dt = date.today() - timedelta(days=w_days)
                w_start_str = w_start_dt.strftime("%Y-%m-%d")

                # If user gave an explicit start_date, you can choose to override or not:
                # e.g., we might do: actual_start = min(w_start_str, start_date) if start_date else w_start_str
                # but let's just use the rolling approach:
                actual_start = w_start_str

                # fetch & merge data
                if data_type == "daily":
                    # 1) fetch daily EOD
                    eod_list = fetch_daily_eod(ticker, actual_start, end_date, EOD_API_KEY)
                    # 2) fetch technical
                    tech_list = fetch_technical_data(ticker, actual_start, end_date, EOD_API_KEY,
                                                     buffer_days=buffer_days)
                    # 3) merge
                    df_merged = merge_daily_and_technical(eod_list, tech_list)
                    vol_col = "volume"
                    price_col = "adjusted_close"
                else:
                    # data_type == "intraday"
                    intraday_json = get_intraday_data(ticker, actual_start, end_date, interval, EOD_API_KEY)
                    if not intraday_json:
                        current_app.logger.warning(f"No intraday data for {ticker} in window {w_label}")
                        continue
                    df_int = pd.DataFrame(intraday_json)
                    if df_int.empty:
                        current_app.logger.warning(f"Intraday is empty for {ticker}")
                        continue
                    df_merged = aggregate_intraday(df_int)
                    vol_col = "intraday_vol_mean"
                    price_col = "intraday_close_mean"

                if df_merged.empty:
                    current_app.logger.warning(f"No data after fetch/merge for {ticker}, window {w_label}")
                    continue

                # 4) Fundamentals
                fund_json = fetch_fundamental_data(ticker, EOD_API_KEY)
                df_fund = process_fundamental_data(fund_json)
                if not df_fund.empty:
                    # replicate fundamentals row for each day
                    df_fund_rep = pd.concat([df_fund]*len(df_merged), ignore_index=True)
                    df_fund_rep["date"] = df_merged["date"].values
                    df_merged = pd.merge(df_merged, df_fund_rep, on="date", how="left")

                # 5) events + macro
                df_merged = process_economic_events(events_data, df_merged)
                df_merged = process_macroeconomic_data(macro_data, df_merged)

                # 6) news
                news_res = fetch_news_sentiment([ticker], actual_start, end_date)
                # user might need to parse that if it's {ticker->list}
                if isinstance(news_res, dict) and ticker in news_res:
                    articles = news_res[ticker]
                elif isinstance(news_res, list):
                    articles = news_res
                else:
                    articles = []
                # simple approach: average or skip
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

                # 7) remove zero-volume & outliers
                if vol_col in df_merged.columns:
                    df_merged = remove_zero_volume(df_merged, vol_col=vol_col)

                # choose some columns for outlier removal
                outlier_cols = [price_col, vol_col, "sma","ema","rsi","macd","stddev","bbands_upper","bbands_lower"]
                outlier_cols = [c for c in outlier_cols if c in df_merged.columns]
                df_merged = remove_outliers(df_merged, outlier_cols)

                # sort
                df_merged.sort_values("date", inplace=True)
                df_merged.reset_index(drop=True, inplace=True)
                if len(df_merged) < 10:
                    current_app.logger.warning(f"Not enough data after cleaning for {ticker}, window={w_label}")
                    continue

                # --------------
                # MULTI-HORIZON
                # --------------
                for h_label, h_shift in horizons.items():
                    df_horizon = df_merged.copy()
                    if price_col not in df_horizon.columns:
                        current_app.logger.warning(f"No price col {price_col} in df for horizon {h_label}")
                        continue

                    # SHIFT target
                    df_horizon["target"] = df_horizon[price_col].shift(-h_shift)
                    df_horizon.dropna(subset=["target"], inplace=True)
                    if df_horizon.empty:
                        current_app.logger.warning(f"No data after target shift for {ticker}, horizon={h_label}")
                        continue

                    # feature selection
                    # exclude columns we don't want as features
                    exclude_cols = set(["date","timestamp","target",price_col,"merge_date"])
                    feature_cols = []
                    for c in df_horizon.columns:
                        if c not in exclude_cols:
                            feature_cols.append(c)

                    X_all = df_horizon[feature_cols].values
                    y_all = df_horizon["target"].values

                    if len(X_all) < 20:
                        continue

                    # train-test split
                    split_idx = int(len(X_all)*0.8)
                    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
                    y_train, y_test = y_all[:split_idx], y_all[split_idx:]

                    if len(X_train) < 10 or len(X_test) < 5:
                        continue

                    # --------------
                    # Train ensemble
                    # --------------
                    preds_dict, ensemble_pred, metrics = train_tabular_ensemble(X_train, y_train, X_test, y_test)
                    mse_val = float(metrics["mse"])
                    mae_val = float(metrics["mae"])
                    r2_val  = float(metrics["r2"])

                    # store each model
                    perf_records = []
                    pred_records = []

                    # 1) model-by-model
                    for m_name, model_preds in preds_dict.items():
                        m_mse = float(np.mean((model_preds - y_test)**2))
                        m_mae = float(np.mean(np.abs(model_preds - y_test)))
                        m_r2  = float(1 - np.sum((model_preds - y_test)**2)/ np.sum((y_test - np.mean(y_test))**2))

                        # performance row
                        perf_rec = {
                            "model_name": m_name,
                            "train_window_start": actual_start,
                            "train_window_end": end_date,
                            "r2_score": m_r2,
                            "mse": m_mse,
                            "mae": m_mae,
                            "training_date": datetime.now(),
                            "ticker": ticker,
                            "horizon": h_label,
                            "window_label": w_label
                        }
                        perf_ids = save_model_performance([perf_rec])  # returns [id]
                        if perf_ids:
                            mp_id = perf_ids[0]
                            # predictions: store last test sample
                            final_val = float(model_preds[-1])
                            pr = {
                                "prediction_date": datetime.now(),
                                "forecast_horizon": h_label,
                                "ticker": ticker,
                                "predicted_value": final_val,
                                "model_name": m_name,
                                "model_performance_id": mp_id
                            }
                            pred_records.append(pr)

                    # 2) ensemble
                    perf_ens = {
                        "model_name": "Ensemble",
                        "train_window_start": actual_start,
                        "train_window_end": end_date,
                        "r2_score": r2_val,
                        "mse": mse_val,
                        "mae": mae_val,
                        "training_date": datetime.now(),
                        "ticker": ticker,
                        "horizon": h_label,
                        "window_label": w_label
                    }
                    ens_ids = save_model_performance([perf_ens])
                    if ens_ids:
                        ens_id = ens_ids[0]
                        final_ens_val = float(ensemble_pred[-1])
                        pr_ens = {
                            "prediction_date": datetime.now(),
                            "forecast_horizon": h_label,
                            "ticker": ticker,
                            "predicted_value": final_ens_val,
                            "model_name": "Ensemble",
                            "model_performance_id": ens_id
                        }
                        pred_records.append(pr_ens)

                    # store predictions
                    if pred_records:
                        save_predictions(pred_records)

                    # add to final_results JSON
                    run_info = {
                        "ticker": ticker,
                        "window": w_label,
                        "horizon": h_label,
                        "ensemble_mse": mse_val,
                        "ensemble_mae": mae_val,
                        "ensemble_r2": r2_val,
                        "final_ensemble_pred": float(ensemble_pred[-1])
                    }
                    all_results.append(run_info)

        return jsonify(all_results), 200

    except Exception as ex:
        current_app.logger.error(f"[ERROR] top-level in /predict: {ex}", exc_info=True)
        return jsonify({"error": str(ex)}), 500
