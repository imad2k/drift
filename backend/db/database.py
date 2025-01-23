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
    Inserts a list of predictions into the 'predictions' table.
    
    Each item in 'predictions_list' is expected to be a dictionary that contains:
      {
        "ticker": <str>,
        "date": <str or date>,
        "rf_prediction": <float>,
        "gbm_prediction": <float>,
        "xgb_prediction": <float>,
        "catboost_prediction": <float>,
        "lightgbm_prediction": <float>,
        "lstm_prediction": <float>,
        "ensemble_prediction": <float>,
        "mse": <float>,
        "mae": <float>,
        "r2_score": <float>
        // anything else you want to insert, but be sure to adapt your SQL
      }

    Make sure your database table 'predictions' has matching columns:
      ticker, prediction_date, rf_prediction, gbm_prediction, xgb_prediction,
      catboost_prediction, lightgbm_prediction, lstm_prediction, ensemble_prediction,
      mse, mae, r2_score
    (Adjust as needed if your schema is different.)
    """
    # Quick sanity check
    if not predictions_list:
        print("No predictions to save.")
        return
    
    # Connect
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
                lstm_prediction,
                ensemble_prediction,
                mse,
                mae,
                r2_score
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        with connection.cursor() as cursor:
            for pred_dict in predictions_list:
                # Extract fields; fallback to None if missing
                ticker = pred_dict.get("ticker")
                prediction_date = pred_dict.get("date")
                rf_val = pred_dict.get("rf_prediction")
                gbm_val = pred_dict.get("gbm_prediction")
                xgb_val = pred_dict.get("xgb_prediction")
                catboost_val = pred_dict.get("catboost_prediction")
                lightgbm_val = pred_dict.get("lightgbm_prediction")
                lstm_val = pred_dict.get("lstm_prediction")
                ensemble_val = pred_dict.get("ensemble_prediction")
                mse_val = pred_dict.get("mse")
                mae_val = pred_dict.get("mae")
                r2_val = pred_dict.get("r2_score")
                
                cursor.execute(
                    insert_sql,
                    (
                        ticker, prediction_date, 
                        rf_val, gbm_val, xgb_val, catboost_val,
                        lightgbm_val, lstm_val, ensemble_val,
                        mse_val, mae_val, r2_val
                    )
                )
        # Commit
        connection.commit()
        print(f"Saved {len(predictions_list)} predictions to database.")
    
    except Exception as e:
        print(f"Error saving to database: {str(e)}")
        if connection:
            connection.rollback()
    finally:
        if connection:
            connection.close()
