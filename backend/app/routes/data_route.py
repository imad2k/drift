from flask import Blueprint, jsonify, current_app, request
from ..modules.prediction import run_predictions_for_ticker
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

db_bp = Blueprint("db", __name__)

DB_CONFIG = {
    'host': os.getenv('RDS_HOST'),
    'port': int(os.getenv('RDS_PORT')),
    'user': os.getenv('RDS_USER'),
    'password': os.getenv('RDS_PASSWORD'),
    'database': 'postgres'
}

def get_db_connection():
    """Create database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        current_app.logger.error(f"Database connection error: {e}")
        return None

@db_bp.route("/user/<username>/predictions", methods=["GET"])
def get_user_predictions(username):
    """Get all predictions for user's tickers"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # First verify user exists
            cur.execute("""
                SELECT id FROM users 
                WHERE username = %s
            """, (username,))
            
            user = cur.fetchone()
            if not user:
                return jsonify({"error": "User not found"}), 404

            # Get user's tickers
            cur.execute("""
                SELECT t.symbol 
                FROM users u
                JOIN user_tickers ut ON u.id = ut.user_id
                JOIN tickers t ON ut.ticker = t.symbol
                WHERE u.username = %s AND ut.enabled = true
            """, (username,))
            
            user_tickers = [row['symbol'] for row in cur.fetchall()]
            
            if not user_tickers:
                return jsonify({"message": "No tickers found for user"}), 404

            # Get predictions with performance metrics
            cur.execute("""
                SELECT 
                    p.prediction_id,
                    p.created_at,
                    p.forecast_date,
                    p.forecast_horizon,
                    p.ticker,
                    p.predicted_value,
                    p.model_name,
                    p.data_type,
                    p.actual_value,
                    p.pct_error,
                    mp.r2_score,
                    mp.mse,
                    mp.mae,
                    mp.window_label,
                    mp.train_window_start,
                    mp.train_window_end
                FROM predictions p
                JOIN model_performance mp ON p.model_performance_id = mp.model_performance_id
                WHERE p.ticker = ANY(%s)
                ORDER BY p.created_at DESC
            """, (user_tickers,))
            
            predictions = cur.fetchall()
            
            # Format datetime objects to strings
            for pred in predictions:
                pred['created_at'] = pred['created_at'].isoformat() if pred['created_at'] else None
                pred['forecast_date'] = pred['forecast_date'].isoformat() if pred['forecast_date'] else None
                pred['train_window_start'] = pred['train_window_start'].isoformat() if pred['train_window_start'] else None
                pred['train_window_end'] = pred['train_window_end'].isoformat() if pred['train_window_end'] else None
            
        conn.close()
        
        return jsonify({
            "username": username,
            "tickers": user_tickers,
            "predictions": predictions
        }), 200

    except Exception as e:
        current_app.logger.error(f"Error fetching predictions: {e}")
        return jsonify({"error": str(e)}), 500

@db_bp.route("/user/<username>/predictions/<ticker>", methods=["GET"])
def get_ticker_predictions(username, ticker):
    """Get predictions for specific ticker"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Verify user has access to ticker
            cur.execute("""
                SELECT 1 
                FROM users u
                JOIN user_tickers ut ON u.id = ut.user_id
                WHERE u.username = %s AND ut.ticker = %s AND ut.enabled = true
            """, (username, ticker))
            
            if not cur.fetchone():
                return jsonify({"error": "Unauthorized access to ticker"}), 403

            # Get predictions with performance metrics
            cur.execute("""
                SELECT 
                    p.prediction_id,
                    p.created_at,
                    p.forecast_date,
                    p.forecast_horizon,
                    p.predicted_value,
                    p.model_name,
                    p.data_type,
                    p.actual_value,
                    p.pct_error,
                    mp.r2_score,
                    mp.mse,
                    mp.mae,
                    mp.window_label,
                    mp.train_window_start,
                    mp.train_window_end
                FROM predictions p
                JOIN model_performance mp ON p.model_performance_id = mp.model_performance_id
                WHERE p.ticker = %s
                ORDER BY p.created_at DESC
            """, (ticker,))
            
            predictions = cur.fetchall()
            
            # Format datetime objects to strings
            for pred in predictions:
                pred['created_at'] = pred['created_at'].isoformat() if pred['created_at'] else None
                pred['forecast_date'] = pred['forecast_date'].isoformat() if pred['forecast_date'] else None
                pred['train_window_start'] = pred['train_window_start'].isoformat() if pred['train_window_start'] else None
                pred['train_window_end'] = pred['train_window_end'].isoformat() if pred['train_window_end'] else None
            
        conn.close()
        
        return jsonify({
            "username": username,
            "ticker": ticker,
            "predictions": predictions
        }), 200

    except Exception as e:
        current_app.logger.error(f"Error fetching predictions: {e}")
        return jsonify({"error": str(e)}), 500

@db_bp.route("/user/<username>/tickers", methods=["GET"])
def get_user_tickers(username):
    """Get all tickers for a user"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT t.symbol, t.enabled 
                FROM users u
                JOIN user_tickers ut ON u.id = ut.user_id
                JOIN tickers t ON ut.ticker = t.symbol
                WHERE u.username = %s
                ORDER BY t.symbol
            """, (username,))
            
            tickers = cur.fetchall()
            
        conn.close()
        
        return jsonify({
            "username": username,
            "tickers": tickers
        }), 200

    except Exception as e:
        current_app.logger.error(f"Error fetching user tickers: {e}")
        return jsonify({"error": str(e)}), 500

@db_bp.route("/user/<username>/tickers", methods=["POST"])
def add_user_ticker(username):
    """Add new ticker to user's watchlist"""
    try:
        data = request.get_json()
        ticker = data.get('ticker')
        company_name = data.get('company_name')

        if not ticker:
            return jsonify({"error": "Ticker symbol required"}), 400

        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get user ID
            cur.execute("SELECT id FROM users WHERE username = %s", (username,))
            user = cur.fetchone()
            if not user:
                return jsonify({"error": "User not found"}), 404

            # Insert or update ticker
            cur.execute("""
                INSERT INTO tickers (symbol, enabled)
                VALUES (%s, true)
                ON CONFLICT (symbol) 
                DO UPDATE SET enabled = true
            """, (ticker,))

            # Link ticker to user
            cur.execute("""
                INSERT INTO user_tickers (user_id, ticker, enabled)
                VALUES (%s, %s, true)
                ON CONFLICT (user_id, ticker) 
                DO UPDATE SET enabled = true
            """, (user['id'], ticker))

            conn.commit()

        # Trigger predictions for new ticker
        # You'll need to implement this part based on your prediction logic
        try:
            run_predictions_for_ticker(ticker)
        except Exception as e:
            current_app.logger.error(f"Error running predictions: {e}")
            # Continue even if predictions fail - we can update later

        return jsonify({
            "message": "Ticker added successfully",
            "ticker": ticker
        }), 201

    except Exception as e:
        current_app.logger.error(f"Error adding ticker: {e}")
        return jsonify({"error": str(e)}), 500