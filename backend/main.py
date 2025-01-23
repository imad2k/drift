import os
from flask import Flask
from dotenv import load_dotenv
from app.routes.predict import predict_bp  # Use absolute import

load_dotenv()  # Load variables from .env

def create_app():
    """Application Factory."""
    app = Flask(__name__)

    @app.route("/")
    def index():
        return "Stock Prediction App is Running!"

    # Register blueprint(s)
    app.register_blueprint(predict_bp)

    return app

if __name__ == "__main__":
    # Create and run the Flask app
    flask_app = create_app()
    port = os.getenv("PORT", 5001)
    flask_app.run(host="0.0.0.0", port=int(port))

