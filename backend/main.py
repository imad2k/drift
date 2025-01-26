import os
import logging
from flask import Flask, jsonify
from dotenv import load_dotenv
import multiprocessing as mp

# Your existing blueprint import
from app.routes.predict import predict_bp

# Import the new TFT blueprint (assuming it's in app/routes/train_tft_route.py)
from app.routes.train_tft_route import tft_bp

# Database connection checker
from db.database import test_db_connection

load_dotenv()  # Load variables from .env

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route("/")
def index():
    app.logger.info("Root endpoint was called")
    return "Stock Prediction App is Running!"

# Register blueprint(s)
app.register_blueprint(predict_bp)     # The old /predict routes
app.register_blueprint(tft_bp)         # The new /train_tft route

# Error handler for internal server errors
@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Server Error: {error}")
    return jsonify({"error": "Internal Server Error"}), 500

# Test DB connection
@app.route("/test-db", methods=["GET"])
def test_db():
    """
    Simple endpoint to verify database connectivity.
    """
    if test_db_connection():
        return jsonify({"message": "DB connection successful"}), 200
    else:
        return jsonify({"error": "DB connection failed"}), 500

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    app.run(host="127.0.0.1", port=5001, debug=True, use_reloader=False)
