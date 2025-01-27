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
    Return a list of newly inserted IDs (assuming usage of RETURNING).
    
    The table might have columns:
      id SERIAL,
      model_name TEXT,
      train_window_start DATE,
      train_window_end DATE,
      r2_score DOUBLE PRECISION,
      mse DOUBLE PRECISION,
      mae DOUBLE PRECISION,
      training_date TIMESTAMP,
      ticker TEXT,
      horizon TEXT,
      window_label TEXT,
      data_type TEXT
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
            RETURNING id
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
    Insert predictions into 'predictions' table. 
    The table might have columns:
      id SERIAL,
      prediction_date TIMESTAMP,
      forecast_horizon TEXT,
      ticker TEXT,
      predicted_value DOUBLE PRECISION,
      model_name TEXT,
      model_performance_id INT,
      data_type TEXT,
      actual_value DOUBLE PRECISION,
      pct_error DOUBLE PRECISION
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
                prediction_date,
                forecast_horizon,
                ticker,
                predicted_value,
                model_name,
                model_performance_id,
                data_type,
                actual_value,
                pct_error
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        with conn.cursor() as cur:
            for rec in pred_records:
                cur.execute(
                    insert_sql,
                    (
                        rec["prediction_date"],
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
