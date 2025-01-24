import os
import logging
from flask import Flask, jsonify
from dotenv import load_dotenv
from app.routes.predict import predict_bp  # Use absolute import
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
app.register_blueprint(predict_bp)

# Error handler for internal server errors
@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Server Error: {error}")
    return jsonify({"error": "Internal Server Error"}), 500

#test db connection
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
    # Create and run the Flask app
    port = os.getenv("PORT", 5001)
    app.run(host="0.0.0.0", port=int(port), debug=True)

