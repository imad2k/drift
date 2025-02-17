# stock_prediction_app/db/database.py

import os
import pg8000
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

RDS_HOST = os.getenv('RDS_HOST')
RDS_PORT = int(os.getenv('RDS_PORT', 5432))
RDS_USER = os.getenv('RDS_USER')
RDS_PASSWORD = os.getenv('RDS_PASSWORD')
RDS_DATABASE = os.getenv('RDS_DATABASE', 'postgres')


def test_db_connection():
    """
    Simple test to confirm DB connectivity.
    """
    try:
        conn = pg8000.connect(
            host=RDS_HOST,
            port=RDS_PORT,
            user=RDS_USER,
            password=RDS_PASSWORD,
            database=RDS_DATABASE
        )
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            _ = cur.fetchone()
        conn.close()
        return True
    except Exception as e:
        print(f"[ERROR] DB connection failed: {e}")
        return False


def save_model_performance(perf_records):
    """
    Insert one or more model performance records into `model_performance`.
    Return list of newly inserted model_performance_id values.
    
    Matches schema:
      model_performance_id SERIAL PRIMARY KEY,
      model_name           TEXT,
      train_window_start   DATE,
      train_window_end     DATE,
      r2_score             DOUBLE PRECISION,
      mse                  DOUBLE PRECISION,
      mae                  DOUBLE PRECISION,
      training_date        TIMESTAMP,
      ticker               TEXT,
      horizon              TEXT,
      window_label         TEXT,
      data_type            TEXT
    """
    if not perf_records:
        return []

    inserted_ids = []
    conn = None
    try:
        conn = pg8000.connect(
            host=RDS_HOST,
            port=RDS_PORT,
            user=RDS_USER,
            password=RDS_PASSWORD,
            database=RDS_DATABASE
        )
        insert_sql = """
            INSERT INTO model_performance (
                model_name,
                train_window_start,
                train_window_end,
                r2_score,
                mse,
                mae,
                training_date,
                ticker,
                horizon,
                window_label,
                data_type
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING model_performance_id
        """
        with conn.cursor() as cur:
            for rec in perf_records:
                cur.execute(
                    insert_sql,
                    (
                        rec.get("model_name"),
                        rec.get("train_window_start"),
                        rec.get("train_window_end"),
                        rec.get("r2_score"),
                        rec.get("mse"),
                        rec.get("mae"),
                        rec.get("training_date", datetime.now()),
                        rec.get("ticker"),
                        rec.get("horizon"),
                        rec.get("window_label"),
                        rec.get("data_type")
                    )
                )
                new_id = cur.fetchone()[0]
                inserted_ids.append(new_id)
        conn.commit()
    except Exception as e:
        print(f"[ERROR] save_model_performance: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

    return inserted_ids


def save_predictions(pred_records):
    """
    Insert predictions into `predictions` table.
    Matches schema:
      prediction_id         SERIAL PRIMARY KEY,
      created_at            TIMESTAMP NOT NULL,
      forecast_date         DATE NOT NULL,
      forecast_horizon      TEXT NOT NULL,
      ticker                TEXT NOT NULL,
      predicted_value       DOUBLE PRECISION NOT NULL,
      model_name            TEXT NOT NULL,
      model_performance_id  INT,
      data_type             TEXT NOT NULL,
      actual_value          DOUBLE PRECISION,
      pct_error             DOUBLE PRECISION
    """
    if not pred_records:
        return

    conn = None
    try:
        conn = pg8000.connect(
            host=RDS_HOST,
            port=RDS_PORT,
            user=RDS_USER,
            password=RDS_PASSWORD,
            database=RDS_DATABASE
        )
        insert_sql = """
            INSERT INTO predictions (
                created_at,
                forecast_date,
                forecast_horizon,
                ticker,
                predicted_value,
                model_name,
                model_performance_id,
                data_type,
                actual_value,
                pct_error
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        with conn.cursor() as cur:
            for rec in pred_records:
                cur.execute(
                    insert_sql,
                    (
                        rec["created_at"],
                        rec["forecast_date"],  # date object or date string
                        rec.get("forecast_horizon"),
                        rec["ticker"],
                        rec["predicted_value"],
                        rec["model_name"],
                        rec.get("model_performance_id"),
                        rec.get("data_type"),
                        rec.get("actual_value"),
                        rec.get("pct_error")
                    )
                )
        conn.commit()
    except Exception as e:
        print(f"[ERROR] save_predictions: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


def update_predictions_actuals(ticker, forecast_date, horizon, data_type, closing_price):
    """
    Update the actual_value and pct_error for any predictions that match 
    (ticker, forecast_date, horizon, data_type) and currently have actual_value=NULL.

    :param ticker: e.g. "AAPL.US"
    :param forecast_date: e.g. "2025-01-31" or a datetime.date object
    :param horizon: e.g. "1d", "2d", etc.
    :param data_type: e.g. "daily" or "intraday"
    :param closing_price: The official close price you fetched post-market
    """
    conn = None
    try:
        conn = pg8000.connect(
            host=RDS_HOST,
            port=RDS_PORT,
            user=RDS_USER,
            password=RDS_PASSWORD,
            database=RDS_DATABASE
        )

        # If closing_price is zero, we set pct_error to NULL to avoid division by zero
        update_sql = """
        UPDATE predictions
           SET actual_value = %s,
               pct_error = CASE 
                             WHEN %s <> 0
                             THEN 100.0 * (predicted_value - %s) / %s
                             ELSE NULL
                           END
         WHERE ticker = %s
           AND forecast_date = %s
           AND forecast_horizon = %s
           AND data_type = %s
           AND actual_value IS NULL
        """
        with conn.cursor() as cur:
            cur.execute(
                update_sql,
                (
                    closing_price,      # actual_value
                    closing_price,      # for the CASE check
                    closing_price,      # predicted_value - X
                    closing_price,      # denominator
                    ticker,
                    forecast_date,
                    horizon,
                    data_type
                )
            )
        conn.commit()
    except Exception as e:
        print(f"[ERROR] update_predictions_actuals: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
