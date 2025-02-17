import logging
from datetime import datetime

async def run_predictions_for_ticker(ticker):
    """Run predictions for a given ticker"""
    try:
        logging.info(f"Running predictions for {ticker}")
        return {
            "ticker": ticker,
            "status": "pending",
            "timestamp": datetime.now().isoformat(),
            "message": "Prediction queued"
        }
    except Exception as e:
        logging.error(f"Prediction error for {ticker}: {str(e)}")
        raise Exception(f"Failed to run predictions for {ticker}: {str(e)}")