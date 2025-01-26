# stock_prediction_app/routes/train_tft_route.py

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from dotenv import load_dotenv

# 1) Import the new helper function
from data.prepare_data import prepare_merged_fund_data  
# 2) Import your existing TFT training script
from models.tft_training import train_tft_on_aggdf  

load_dotenv()
EOD_API_KEY = os.getenv('EOD_API_KEY')

tft_bp = Blueprint('tft_bp', __name__)

@tft_bp.route("/train_tft", methods=["POST"])
def train_tft_route():
    """
    POST /train_tft
    JSON body: {
        "ticker": "AAPL.US",
        "start_date": "2021-01-01",
        "end_date": "2025-01-23",
        "interval": "1h"
    }
    Fetches & merges data into df_merged_fund with a 'target' column,
    then trains a TemporalFusionTransformer, returning metrics & sample predictions.
    """
    try:
        data = request.get_json()
        ticker = data.get("ticker", "AAPL.US")
        start_date = data.get("start_date", "2021-01-01")
        end_date   = data.get("end_date", "2025-01-23")
        interval   = data.get("interval", "1h")

        # 1) Fetch & merge data into df_merged_fund (must have 'target' column).
        df_merged_fund = prepare_merged_fund_data(ticker, start_date, end_date, interval)

        # 2) Train TFT & get predictions
        tft_model, tft_metrics, preds = train_tft_on_aggdf(df_merged_fund, ticker)

        return jsonify({
            "status": "success",
            "ticker": ticker,
            "metrics": tft_metrics,                   # dict with train_loss, val_loss
            "sample_predictions": preds[:10].tolist() # show first 10 predictions
        }), 200

    except Exception as e:
        current_app.logger.error(f"TFT training error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500