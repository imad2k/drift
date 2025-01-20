import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import pymysql
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

EOD_API_KEY = os.getenv('EOD_API_KEY')
RDS_HOST = os.getenv('RDS_HOST')
RDS_PORT = int(os.getenv('RDS_PORT', 3306))
RDS_USER = os.getenv('RDS_USER')
RDS_PASSWORD = os.getenv('RDS_PASSWORD')
RDS_DB = os.getenv('RDS_DB')

def get_intraday_data(ticker, start_date, end_date, interval, api_key):
    url = f"https://eodhistoricaldata.com/api/intraday/{ticker}?api_token={api_key}&interval={interval}&from={start_date}&to={end_date}&fmt=json"
    response = requests.get(url)
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
    connection = pymysql.connect(
        host=RDS_HOST,
        port=RDS_PORT,
        user=RDS_USER,
        password=RDS_PASSWORD,
        db=RDS_DB
    )
    try:
        with connection.cursor() as cursor:
            for prediction in predictions:
                sql = """
                INSERT INTO predictions (
                    ticker_id, date, intraday_vol_mean, intraday_high_mean, intraday_low_mean, intraday_close_mean,
                    random_forest_pred, gradient_boosting_pred, xgboost_pred, ensemble_pred, actual_close,
                    rf_diff, gbm_diff, xgb_diff, ensemble_diff, rf_mae, gbm_mae, xgb_mae, ensemble_mae
                ) VALUES (
                    (SELECT id FROM tickers WHERE ticker_symbol = %s), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """
                cursor.execute(sql, (
                    prediction['ticker'],
                    prediction['date'],
                    prediction['intraday_vol_mean'],
                    prediction['intraday_high_mean'],
                    prediction['intraday_low_mean'],
                    prediction['intraday_close_mean'],
                    prediction['rf_prediction'],
                    prediction['gbm_prediction'],
                    prediction['xgb_prediction'],
                    prediction['ensemble_prediction'],
                    prediction['actual_close'],
                    prediction['rf_diff'],
                    prediction['gbm_diff'],
                    prediction['xgb_diff'],
                    prediction['ensemble_diff'],
                    prediction['rf_mae'],
                    prediction['gbm_mae'],
                    prediction['xgb_mae'],
                    prediction['ensemble_mae']
                ))
        connection.commit()
    finally:
        connection.close()

def predict_next_close(ticker, start_date, end_date, intraday_interval):
    intraday_json = get_intraday_data(ticker, start_date, end_date, intraday_interval, EOD_API_KEY)
    intraday_data = json.loads(intraday_json)
    df_intraday = pd.DataFrame(intraday_data)

    if df_intraday.empty:
        raise ValueError("No intraday data found for the specified range.")

    df_agg_intraday = aggregate_intraday(df_intraday)
    df_agg_intraday["target"] = df_agg_intraday["intraday_close_mean"].shift(-1)
    df_agg_intraday.dropna(inplace=True)

    X = df_agg_intraday[["intraday_vol_mean", "intraday_high_mean", "intraday_low_mean", "intraday_close_mean"]]
    y = df_agg_intraday["target"]

    split_index = int(len(df_agg_intraday) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    rf = RandomForestRegressor(random_state=42)
    gbm = GradientBoostingRegressor(random_state=42)
    xgb = XGBRegressor(random_state=42)

    rf.fit(X_train, y_train)
    gbm.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    gbm_pred = gbm.predict(X_test)
    xgb_pred = xgb.predict(X_test)

    final_prediction = np.mean([rf_pred[-1], gbm_pred[-1], xgb_pred[-1]])

    predictions = [{
        'ticker': ticker,
        'date': str(datetime.now().date()),
        'intraday_vol_mean': df_agg_intraday.iloc[-1]['intraday_vol_mean'],
        'intraday_high_mean': df_agg_intraday.iloc[-1]['intraday_high_mean'],
        'intraday_low_mean': df_agg_intraday.iloc[-1]['intraday_low_mean'],
        'intraday_close_mean': df_agg_intraday.iloc[-1]['intraday_close_mean'],
        'rf_prediction': rf_pred[-1],
        'gbm_prediction': gbm_pred[-1],
        'xgb_prediction': xgb_pred[-1],
        'ensemble_prediction': final_prediction,
        'actual_close': y_test.iloc[-1],
        'rf_diff': rf_pred[-1] - y_test.iloc[-1],
        'gbm_diff': gbm_pred[-1] - y_test.iloc[-1],
        'xgb_diff': xgb_pred[-1] - y_test.iloc[-1],
        'ensemble_diff': final_prediction - y_test.iloc[-1],
        'rf_mae': mean_absolute_error(y_test, rf_pred),
        'gbm_mae': mean_absolute_error(y_test, gbm_pred),
        'xgb_mae': mean_absolute_error(y_test, xgb_pred),
        'ensemble_mae': mean_absolute_error(y_test, [final_prediction] * len(y_test))
    }]

    save_to_rds(predictions)

    return {
        "statusCode": 200,
        "body": json.dumps(predictions)
    }

def lambda_handler(event, context):
    tickers = event.get('tickers', ['AAPL.US'])
    start_date = event.get('start_date', '2019-01-01')
    end_date = event.get('end_date', '2025-01-18')
    interval = event.get('interval', '1h')

    for ticker in tickers:
        try:
            result = predict_next_close(ticker, start_date, end_date, interval)
            print(f"Prediction Result for {ticker}:")
            pprint(result)
        except Exception as e:
            print(f"Error for {ticker}: {e}")

    return {
        "statusCode": 200,
        "body": json.dumps({"message": "Predictions completed"})
    }