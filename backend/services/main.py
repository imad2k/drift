import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import pg8000
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

# Load environment variables
EOD_API_KEY = os.getenv('EOD_API_KEY')
RDS_HOST = os.getenv('RDS_HOST')
RDS_PORT = int(os.getenv('RDS_PORT', 5432))
RDS_USER = os.getenv('RDS_USER')
RDS_PASSWORD = os.getenv('RDS_PASSWORD')

# Create the Flask app object
app = Flask(__name__)

@app.route("/")
def index():
    return "Stock Prediction App is Running!"

@app.route("/test_db_connection", methods=["GET"])
def test_db_connection():
    try:
        connection = pg8000.connect(
            host=RDS_HOST,
            port=RDS_PORT,
            user=RDS_USER,
            password=RDS_PASSWORD
        )
        connection.close()
        return "Database connection successful!", 200
    except Exception as e:
        return f"Database connection failed: {str(e)}", 500

@app.route("/predict", methods=["POST"])
def predict_next_close():
    try:
        # Get input data
        request_data = request.get_json()
        tickers = request_data.get("tickers", ["AAPL.US"])
        start_date = request_data.get("start_date", "2020-11-01")
        end_date = request_data.get("end_date", "2025-01-18")
        interval = request_data.get("interval", "1h")

        results = []

        for ticker in tickers:
            try:
                # Fetch intraday data
                intraday_json = get_intraday_data(ticker, start_date, end_date, interval, EOD_API_KEY)
                intraday_data = json.loads(intraday_json)

                if not intraday_data or "errors" in intraday_data:
                    print(f"Error fetching data for {ticker}: {intraday_data}")
                    return jsonify({"error": f"No valid intraday data for {ticker}."}), 400

                # Convert JSON data to DataFrame
                df_intraday = pd.DataFrame(intraday_data)

                # Process intraday data
                df_agg_intraday = aggregate_intraday(df_intraday)

                # Feature engineering
                df_agg_intraday["target"] = df_agg_intraday["intraday_close_mean"].shift(-1)
                df_agg_intraday.dropna(inplace=True)

                X = df_agg_intraday[["intraday_vol_mean", "intraday_high_mean", "intraday_low_mean", "intraday_close_mean"]]
                y = df_agg_intraday["target"]

                # Train-test split
                split_index = int(len(df_agg_intraday) * 0.8)
                X_train, X_test = X[:split_index], X[split_index:]
                y_train, y_test = y[:split_index], y[split_index:]

                # Train models
                rf = RandomForestRegressor(random_state=42)
                gbm = GradientBoostingRegressor(random_state=42)
                xgb = XGBRegressor(random_state=42)

                rf.fit(X_train, y_train)
                gbm.fit(X_train, y_train)
                xgb.fit(X_train, y_train)

                # Predict
                rf_pred = rf.predict(X_test)
                gbm_pred = gbm.predict(X_test)
                xgb_pred = xgb.predict(X_test)

                final_prediction = np.mean([rf_pred[-1], gbm_pred[-1], xgb_pred[-1]])

                # Prepare prediction result
                prediction = {
                    "ticker": ticker,
                    "date": str(datetime.now().date()),
                    "intraday_vol_mean": float(df_agg_intraday.iloc[-1]['intraday_vol_mean']),
                    "intraday_high_mean": float(df_agg_intraday.iloc[-1]['intraday_high_mean']),
                    "intraday_low_mean": float(df_agg_intraday.iloc[-1]['intraday_low_mean']),
                    "intraday_close_mean": float(df_agg_intraday.iloc[-1]['intraday_close_mean']),
                    "rf_prediction": float(rf_pred[-1]),
                    "gbm_prediction": float(gbm_pred[-1]),
                    "xgb_prediction": float(xgb_pred[-1]),
                    "ensemble_prediction": float(final_prediction),
                    "actual_close": float(y_test.iloc[-1]),
                    "rf_diff": float(rf_pred[-1] - y_test.iloc[-1]),
                    "gbm_diff": float(gbm_pred[-1] - y_test.iloc[-1]),
                    "xgb_diff": float(xgb_pred[-1] - y_test.iloc[-1]),
                    "ensemble_diff": float(final_prediction - y_test.iloc[-1]),
                    "rf_mae": float(np.mean(np.abs(rf_pred - y_test))),
                    "gbm_mae": float(np.mean(np.abs(gbm_pred - y_test))),
                    "xgb_mae": float(np.mean(np.abs(xgb_pred - y_test))),
                    "ensemble_mae": float(np.mean(np.abs(final_prediction - y_test)))
                }

                results.append(prediction)
                save_to_rds([prediction])

            except Exception as e:
                return jsonify({"error": f"Error for {ticker}: {str(e)}"}), 500

        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_intraday_data(ticker, start_date, end_date, interval, api_key):
    # Convert date strings to Unix timestamps
    start_timestamp = int(time.mktime(datetime.strptime(start_date, "%Y-%m-%d").timetuple()))
    end_timestamp = int(time.mktime(datetime.strptime(end_date, "%Y-%m-%d").timetuple()))
    
    url = f"https://eodhistoricaldata.com/api/intraday/{ticker}?api_token={api_key}&interval={interval}&from={start_timestamp}&to={end_timestamp}&fmt=json"
    response = requests.get(url)
    print(f"URL: {url}")
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Text: {response.text}")
    return response.text

def aggregate_intraday(intraday_df):
    intraday_df["timestamp"] = pd.to_datetime(intraday_df["timestamp"], unit='s')
    intraday_df["date"] = intraday_df["timestamp"].dt.date
    grouped = intraday_df.groupby("date")
    agg_df = grouped.agg({
        'volume': 'mean',
        'high': 'mean',
        'low': 'mean',
        'close': 'mean'
    }).reset_index()
    agg_df.rename(columns={
        'volume': 'intraday_vol_mean',
        'high': 'intraday_high_mean',
        'low': 'intraday_low_mean',
        'close': 'intraday_close_mean'
    }, inplace=True)
    return agg_df

def save_to_rds(predictions):
    connection = pg8000.connect(
        host=RDS_HOST,
        port=RDS_PORT,
        user=RDS_USER,
        password=RDS_PASSWORD
    )
    try:
        with connection.cursor() as cursor:
            for prediction in predictions:
                # Fetch the ticker_id
                cursor.execute("SELECT id FROM tickers WHERE ticker_symbol = %s", (prediction['ticker'],))
                ticker_id = cursor.fetchone()
                
                if ticker_id:
                    ticker_id = ticker_id[0]
                else:
                    # Insert the new ticker into the tickers table
                    cursor.execute("INSERT INTO tickers (ticker_symbol) VALUES (%s) RETURNING id", (prediction['ticker'],))
                    ticker_id = cursor.fetchone()[0]
                    print(f"Inserted new ticker {prediction['ticker']} with id {ticker_id}")

                print(f"Inserting prediction for ticker_id: {ticker_id}")

                sql = """
                INSERT INTO predictions (
                    ticker_id, date, intraday_vol_mean, intraday_high_mean, intraday_low_mean, intraday_close_mean,
                    random_forest_pred, gradient_boosting_pred, xgboost_pred, ensemble_pred, actual_close,
                    rf_diff, gbm_diff, xgb_diff, ensemble_diff, rf_mae, gbm_mae, xgb_mae, ensemble_mae
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """
                cursor.execute(sql, (
                    ticker_id,
                    prediction['date'],
                    float(prediction['intraday_vol_mean']),
                    float(prediction['intraday_high_mean']),
                    float(prediction['intraday_low_mean']),
                    float(prediction['intraday_close_mean']),
                    float(prediction['rf_prediction']),
                    float(prediction['gbm_prediction']),
                    float(prediction['xgb_prediction']),
                    float(prediction['ensemble_prediction']),
                    float(prediction['actual_close']),
                    float(prediction['rf_diff']),
                    float(prediction['gbm_diff']),
                    float(prediction['xgb_diff']),
                    float(prediction['ensemble_diff']),
                    float(prediction['rf_mae']),
                    float(prediction['gbm_mae']),
                    float(prediction['xgb_mae']),
                    float(prediction['ensemble_mae'])
                ))
        connection.commit()
    except Exception as e:
        print(f"Error saving to database: {str(e)}")
    finally:
        connection.close()