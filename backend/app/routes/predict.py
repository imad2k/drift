# stock_prediction_app/app/routes/predict.py

import os
import json
import pandas as pd
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from dotenv import load_dotenv

# We must import "time" because the get_intraday_data function uses time.mktime
import time

# Import fetch functions from data_fetching
from data.data_fetching import (
    fetch_fundamental_data,
    fetch_news_sentiment,
    fetch_economic_events,
    fetch_macroeconomic_data,
    get_intraday_data
)

# Import all relevant processing/aggregation functions
from data.data_processing import (
    process_external_data,
    aggregate_intraday,
    process_fundamental_data,
    process_economic_events,
    process_macroeconomic_data
)

# Import model training utilities (hyperparameter_tuning, train_and_predict, create_sequences, etc.)
from models.model_training import (
    hyperparameter_tuning,
    train_and_predict,
    create_sequences
)

# Import our function to save to RDS
from db.database import save_to_rds

# Import relevant models (used in final route code)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

load_dotenv()
EOD_API_KEY = os.getenv('EOD_API_KEY')
RDS_HOST = os.getenv('RDS_HOST')
RDS_PORT = int(os.getenv('RDS_PORT', 5432))
RDS_USER = os.getenv('RDS_USER')
RDS_PASSWORD = os.getenv('RDS_PASSWORD')

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        current_app.logger.info(f"Received data: {data}")

        # Add your prediction logic here
        # Example response
        response = {"message": "Prediction received", "data": data}
        current_app.logger.info(f"Response: {response}")
        return jsonify(response)
    except Exception as e:
        current_app.logger.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({"error": "Internal Server Error"}), 500

@predict_bp.route("/predict", methods=["POST"])
def predict_next_close():
    """
    Main prediction endpoint. 
    1) Fetch intraday, macro, and economic events data.
    2) Merge & do basic feature engineering.
    3) Train multiple models (RF, GB, XGB, CatBoost, LightGBM, LSTM).
    4) Ensemble the predictions.
    5) Save & return results.
    """
    try:
        request_data = request.get_json()
        tickers = request_data.get("tickers", ["AAPL.US"])
        start_date = request_data.get("start_date", "2020-11-01")
        end_date = request_data.get("end_date", "2025-01-18")
        interval = request_data.get("interval", "1h")

        results = []

        for ticker in tickers:
            try:
                # 1. Fetch intraday data
                intraday_json = get_intraday_data(ticker, start_date, end_date, interval, EOD_API_KEY)
                # The original code expects this to be JSON already parseable, but let's ensure it's correct:
                if isinstance(intraday_json, str):
                    # If for any reason it's a string, load it:
                    intraday_data = json.loads(intraday_json)
                else:
                    intraday_data = intraday_json

                df_intraday = pd.DataFrame(intraday_data)
                if df_intraday.empty:
                    return jsonify({"error": f"No intraday data returned for {ticker}"}), 500

                # 2. Fetch macroeconomic data & economic events
                macroeconomic_data = fetch_macroeconomic_data()
                economic_events = fetch_economic_events(start_date, end_date)

                # 3. Merge macro data into intraday
                df_merged_macro = process_macroeconomic_data(macroeconomic_data, df_intraday)

                # 4. Merge economic event data
                df_processed = process_economic_events(economic_events, df_merged_macro)

                # 5. Basic feature engineering: we create a target = next day's close
                #    But first, let's aggregate intraday if needed:
                #    (In the original final route code, we didn't do an explicit aggregator 
                #     before merging macro, but let's remain consistent with the script's approach.)
                # Actually, the final route code doesn't call aggregate_intraday on df_intraday 
                # *before* merging with macro. It's using raw intraday for merges. We'll follow that logic.

                # Then we shift -1 for the target
                df_processed["target"] = df_processed["intraday_close_mean"].shift(-1)
                df_processed.dropna(inplace=True)

                # 6. Prepare features (the original code picks specific columns)
                X = df_processed[[
                    "intraday_vol_mean", "intraday_high_mean", "intraday_low_mean",
                    "intraday_close_mean", "macro_open", "macro_high", "macro_low", "macro_close",
                    "avg_importance", "event_count"
                ]]
                y = df_processed["target"]

                # 7. Train-test split
                split_index = int(len(df_processed) * 0.8)
                X_train, X_test = X[:split_index], X[split_index:]
                y_train, y_test = y[:split_index], y[split_index:]

                # 8. Train models (default hyperparams here, consistent with your final route)
                models = {
                    "RandomForest": RandomForestRegressor(random_state=42),
                    "GradientBoosting": GradientBoostingRegressor(random_state=42),
                    "XGBoost": XGBRegressor(random_state=42),
                    "CatBoost": CatBoostRegressor(verbose=0, random_state=42),
                    "LightGBM": LGBMRegressor(random_state=42),
                }
                predictions = {}
                for model_name, model in models.items():
                    model.fit(X_train, y_train)
                    predictions[model_name] = model.predict(X_test)

                # 9. LSTM model
                #    Note: The original code sets two LSTM layers, 
                #    plus `return_sequences=True` on the first, then final Dense(1).
                lstm_model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                    LSTM(50, return_sequences=False),
                    Dense(1)
                ])
                lstm_model.compile(optimizer="adam", loss="mse")
                # Expand dims for LSTM input [samples, timesteps, features], here timesteps = X_train.shape[1]
                lstm_model.fit(
                    X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1),
                    y_train.values,
                    epochs=10,
                    batch_size=32,
                    verbose=0
                )
                lstm_predictions = lstm_model.predict(X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1))

                # 10. Ensemble
                ensemble_prediction = (
                    sum(predictions.values()) + lstm_predictions.flatten()
                ) / (len(predictions) + 1)

                # 11. Metrics
                mse = mean_squared_error(y_test, ensemble_prediction)
                mae = mean_absolute_error(y_test, ensemble_prediction)
                r2 = r2_score(y_test, ensemble_prediction)

                # 12. Save final result (like original code)
                prediction_record = {
                    "ticker": ticker,
                    "date": str(datetime.now().date()),
                    "rf_prediction": float(predictions["RandomForest"][-1]),
                    "gbm_prediction": float(predictions["GradientBoosting"][-1]),
                    "xgb_prediction": float(predictions["XGBoost"][-1]),
                    "catboost_prediction": float(predictions["CatBoost"][-1]),
                    "lightgbm_prediction": float(predictions["LightGBM"][-1]),
                    "lstm_prediction": float(lstm_predictions[-1]),
                    "ensemble_prediction": float(ensemble_prediction[-1]),
                    "mse": mse,
                    "mae": mae,
                    "r2_score": r2
                }
                results.append(prediction_record)

                # 13. Save to DB (same function from original code)
                save_to_rds([prediction_record])

            except Exception as e:
                return jsonify({"error": f"Error for {ticker}: {str(e)}"}), 500

        return jsonify(results), 200

    except Exception as e:
        # In your original code, you have a small typo: `return 'jsonify{"error": str(e)}, 500'`
        # We'll fix it properly:
        return jsonify({"error": str(e)}), 500
