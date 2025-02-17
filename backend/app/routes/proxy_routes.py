from flask import Blueprint, jsonify
import os
import requests

proxy_bp = Blueprint("proxy", __name__)

ALPACA_API_KEY = os.getenv("APACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("APACA_SECRET_KEY")

@proxy_bp.route("/proxy/logo/<symbol>")
def get_logo(symbol):
    try:
        response = requests.get(
            f"https://data.alpaca.markets/v1beta1/logos/{symbol}",
            headers={
                "APCA-API-KEY-ID": ALPACA_API_KEY,
                "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
            }
        )
        response.raise_for_status()
        return jsonify({"logoUrl": response.url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@proxy_bp.route("/proxy/top-picks")
def get_top_picks():
    try:
        response = requests.get(
            "https://data.alpaca.markets/v1beta1/screener/stocks/most-actives",
            headers={
                "APCA-API-KEY-ID": ALPACA_API_KEY,
                "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
            }
        )
        response.raise_for_status()
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500