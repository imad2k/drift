# stock_prediction_app/db/database.py
import os
import pg8000
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

RDS_HOST = os.getenv('RDS_HOST')
RDS_PORT = int(os.getenv('RDS_PORT', 5432))
RDS_USER = os.getenv('RDS_USER')
RDS_PASSWORD = os.getenv('RDS_PASSWORD')

# Optionally, if you have a specific DB name:
RDS_DATABASE = os.getenv('RDS_DATABASE', 'postgres')

def test_db_connection():
    """
    Attempts a simple SELECT 1 query to confirm that we can connect
    to the database. Returns True if successful, False otherwise.
    """
    try:
        connection = pg8000.connect(
            host=RDS_HOST,
            port=RDS_PORT,
            user=RDS_USER,
            password=RDS_PASSWORD,
            database=RDS_DATABASE
        )
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1;")
            _ = cursor.fetchone()
        connection.close()
        return True
    except Exception as e:
        print(f"Database connection test failed: {e}")
        return False


def save_to_rds(predictions_list):
    """
    Inserts a list of prediction dictionaries into the 'predictions' table.
    Each dict should have columns that match your schema:
        ticker, prediction_date, rf_prediction, gbm_prediction, xgb_prediction,
        catboost_prediction, lightgbm_prediction, tft_prediction, ensemble_prediction,
        mse, mae, r2_score
    """
    if not predictions_list:
        print("No predictions to save.")
        return
    
    connection = None
    try:
        connection = pg8000.connect(
            host=RDS_HOST,
            port=RDS_PORT,
            user=RDS_USER,
            password=RDS_PASSWORD,
            database=RDS_DATABASE
        )
        insert_sql = """
            INSERT INTO predictions (
                ticker,
                prediction_date,
                rf_prediction,
                gbm_prediction,
                xgb_prediction,
                catboost_prediction,
                lightgbm_prediction,
                tft_prediction,
                ensemble_prediction,
                mse,
                mae,
                r2_score
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        with connection.cursor() as cursor:
            for pred_dict in predictions_list:
                cursor.execute(
                    insert_sql,
                    (
                        pred_dict["ticker"],
                        pred_dict["date"],
                        pred_dict.get("rf_prediction"),
                        pred_dict.get("gbm_prediction"),
                        pred_dict.get("xgb_prediction"),
                        pred_dict.get("catboost_prediction"),
                        pred_dict.get("lightgbm_prediction"),
                        pred_dict.get("tft_prediction"),            # New Prediction
                        pred_dict.get("ensemble_prediction"),
                        pred_dict.get("mse"),
                        pred_dict.get("mae"),
                        pred_dict.get("r2_score")
                    )
                )
        connection.commit()
        print(f"Saved {len(predictions_list)} predictions to database.")
    except Exception as e:
        print(f"DB Insertion Error: {e}")
        if connection:
            connection.rollback()
    finally:
        if connection:
            connection.close()