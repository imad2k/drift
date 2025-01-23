import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pg8000
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
EOD_API_KEY = os.getenv('EOD_API_KEY')
RDS_HOST = os.getenv('RDS_HOST')
RDS_PORT = int(os.getenv('RDS_PORT', 5432))
RDS_USER = os.getenv('RDS_USER')
RDS_PASSWORD = os.getenv('RDS_PASSWORD')

# Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return "Stock Prediction App is Running!"

# Fetch data from external APIs
def fetch_fundamental_data(ticker, api_token):
    url = f"https://eodhd.com/api/fundamentals/{ticker}?api_token={api_token}&fmt=json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch fundamental data for {ticker}. Status code: {response.status_code}")
        return {}

def fetch_news_sentiment(tickers, start_date, end_date):
    tickers_str = ','.join(tickers)
    url = f"https://eodhd.com/api/sentiments?s={tickers_str}&from={start_date}&to={end_date}&api_token={EOD_API_KEY}&fmt=json"
    response = requests.get(url)
    
    print(f"Fetching news sentiment for tickers: {tickers}")
    print(f"URL: {url}")
    print(f"Response Status Code: {response.status_code}")
    
    if response.status_code != 200:
        print(f"Failed to fetch news sentiment: {response.text}")
        return {}
    
    sentiment_data = response.json()
    
    # Convert response to a usable DataFrame format
    sentiment_dfs = {}
    for ticker, data in sentiment_data.items():
        df = pd.DataFrame(data)
        df["ticker"] = ticker
        sentiment_dfs[ticker] = df
    
    # Concatenate all ticker data
    if sentiment_dfs:
        return pd.concat(sentiment_dfs.values(), ignore_index=True)
    else:
        return pd.DataFrame()


def fetch_economic_events(start_date, end_date, country="USA"):
    url = f"https://eodhd.com/api/economic-events"
    params = {
        "api_token": EOD_API_KEY,
        "from": start_date,
        "to": end_date,
        "country": country,
        "limit": 1000,
        "offset": 0
    }
    response = requests.get(url, params=params)
    print(f"Fetching economic events data: {response.url}")
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching economic events data: {response.text}")
        return []


def fetch_macroeconomic_data():
    url = f"https://eodhd.com/api/eod/UK10Y.GBOND?api_token={EOD_API_KEY}&fmt=json"
    response = requests.get(url)
    
    print(f"URL: {url}")
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Text: {response.text}")
    print(f"Response Headers: {response.headers}")
    
    if response.status_code == 200:
        try:
            return response.json()
        except json.JSONDecodeError:
            print("Error: Response is not valid JSON.")
            return {}
    else:
        print(f"Error: Failed to fetch data. Status Code: {response.status_code}, Response: {response.text}")
        return {}

def get_intraday_data(ticker, start_date, end_date, interval, EOD_API_KEY):
    """
    Fetch intraday data for the specified ticker, date range, and interval.
    """
    # Convert date strings to Unix timestamps
    start_timestamp = int(time.mktime(datetime.strptime(start_date, "%Y-%m-%d").timetuple()))
    end_timestamp = int(time.mktime(datetime.strptime(end_date, "%Y-%m-%d").timetuple()))
    
    
    # Build the URL for the API call
    url = f"https://eodhd.com/api/intraday/{ticker}?api_token={EOD_API_KEY}&interval={interval}&from={start_timestamp}&to={end_timestamp}&fmt=json"
    
    # Make the API request
    response = requests.get(url)
    
    # Debugging: Print response details for troubleshooting
    print(f"Fetching intraday data for {ticker}")
    print(f"URL: {url}")
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Text: {response.text}")
    
    if response.status_code == 200:
        try:
            return response.json()  # Parse and return JSON data
        except json.JSONDecodeError as e:
            print(f"JSON decoding error for {ticker}: {e}")
            return []  # Return an empty list if JSON decoding fails
    else:
        print(f"Error fetching intraday data for {ticker}: {response.text}")
        return []  # Return an empty list for non-200 responses

# Process external data
def process_external_data(fundamental_data, news_sentiment, economic_events, macroeconomic_data):
    # Initialize the data dictionary
    data = {}

    # Process Fundamental Data
    if fundamental_data:
        highlights = fundamental_data.get("Highlights", {})
        valuation = fundamental_data.get("Valuation", {})
        esg_scores = fundamental_data.get("ESGScores", {})
        technicals = fundamental_data.get("Technicals", {})
        
        data["market_cap"] = highlights.get("MarketCapitalization", np.nan)
        data["pe_ratio"] = highlights.get("PERatio", np.nan)
        data["dividend_yield"] = highlights.get("DividendYield", np.nan)
        data["return_on_assets"] = highlights.get("ReturnOnAssets", np.nan)
        data["return_on_equity"] = highlights.get("ReturnOnEquity", np.nan)
        data["beta"] = technicals.get("Beta", np.nan)
        data["esg_environment_score"] = esg_scores.get("EnvironmentScore", np.nan)
        data["esg_social_score"] = esg_scores.get("SocialScore", np.nan)
        data["esg_governance_score"] = esg_scores.get("GovernanceScore", np.nan)

    # Process News Sentiment
    if news_sentiment:
        data["news_sentiment_score"] = np.mean([item.get("normalized", 0) for item in news_sentiment])
        data["news_sentiment_count"] = len(news_sentiment)

    # Process Economic Events
    if economic_events:
        high_impact_events = [event for event in economic_events if event.get("impact") == "High"]
        data["economic_event_high_impact_count"] = len(high_impact_events)
        data["economic_event_average_change"] = np.mean(
            [event.get("change_percentage", 0) for event in economic_events]
        )

    # Process Macroeconomic Data
    if macroeconomic_data:
        gdp_growth = next((item for item in macroeconomic_data if item.get("indicator") == "GDP Growth"), {})
        inflation_rate = next((item for item in macroeconomic_data if item.get("indicator") == "Inflation Rate"), {})
        unemployment_rate = next((item for item in macroeconomic_data if item.get("indicator") == "Unemployment Rate"), {})
        
        data["gdp_growth_rate"] = gdp_growth.get("value", np.nan)
        data["inflation_rate"] = inflation_rate.get("value", np.nan)
        data["unemployment_rate"] = unemployment_rate.get("value", np.nan)

    # Convert the data dictionary to a DataFrame
    return pd.DataFrame([data])


# Aggregate intraday data
def aggregate_intraday(intraday_df):
    intraday_df["timestamp"] = pd.to_datetime(intraday_df["timestamp"], unit='s')
    intraday_df["date"] = intraday_df["timestamp"].dt.date
    grouped = intraday_df.groupby("date")
    agg_df = grouped.agg({
        'volume': 'mean',
        'high': 'mean',
        'low': 'mean',
        'close': 'mean'
    }).reset_index()
    agg_df.rename(columns={
        'volume': 'intraday_vol_mean',
        'high': 'intraday_high_mean',
        'low': 'intraday_low_mean',
        'close': 'intraday_close_mean'
    }, inplace=True)
    return agg_df

def process_fundamental_data(fundamental_data):
    if not fundamental_data:
        return pd.DataFrame()

    try:
        highlights = fundamental_data.get("Highlights", {})
        valuation = fundamental_data.get("Valuation", {})
        shares_stats = fundamental_data.get("SharesStats", {})
        technicals = fundamental_data.get("Technicals", {})
        splits_dividends = fundamental_data.get("SplitsDividends", {})
        esg_scores = fundamental_data.get("ESGScores", {})
        earnings = fundamental_data.get("Earnings", {}).get("History", [])
        financials = fundamental_data.get("Financials", {})

        processed_data = {
            # General Information
            "market_cap": highlights.get("MarketCapitalization"),
            "ebitda": highlights.get("EBITDA"),
            "pe_ratio": highlights.get("PERatio"),
            "peg_ratio": highlights.get("PEGRatio"),
            "eps": highlights.get("EarningsShare"),
            "book_value": highlights.get("BookValue"),
            "dividend_yield": highlights.get("DividendYield"),
            "profit_margin": highlights.get("ProfitMargin"),
            "operating_margin": highlights.get("OperatingMargin"),
            "return_on_assets": highlights.get("ReturnOnAssets"),
            "return_on_equity": highlights.get("ReturnOnEquity"),
            "revenue": highlights.get("Revenue"),
            "revenue_per_share": highlights.get("RevenuePerShare"),
            "gross_profit": highlights.get("GrossProfit"),
            "diluted_eps": highlights.get("DilutedEps"),
            "quarterly_earnings_growth": highlights.get("QuarterlyEarningsGrowthYOY"),

            # Valuation Metrics
            "trailing_pe": valuation.get("TrailingPE"),
            "forward_pe": valuation.get("ForwardPE"),
            "price_to_sales": valuation.get("PriceSales"),
            "price_to_book": valuation.get("PriceBook"),
            "enterprise_value_revenue": valuation.get("EnterpriseValueRevenue"),
            "enterprise_value_ebitda": valuation.get("EnterpriseValueEbitda"),

            # Share Statistics
            "shares_outstanding": shares_stats.get("SharesOutstanding"),
            "shares_float": shares_stats.get("SharesFloat"),
            "percent_held_by_insiders": shares_stats.get("PercentInsiders"),
            "percent_held_by_institutions": shares_stats.get("PercentInstitutions"),

            # Technical Indicators
            "beta": technicals.get("Beta"),
            "52_week_high": technicals.get("52WeekHigh"),
            "52_week_low": technicals.get("52WeekLow"),
            "50_day_moving_avg": technicals.get("50DayMA"),
            "200_day_moving_avg": technicals.get("200DayMA"),

            # Splits and Dividends
            "dividend_rate": splits_dividends.get("ForwardAnnualDividendRate"),
            "dividend_payout_ratio": splits_dividends.get("PayoutRatio"),
            "last_dividend_date": splits_dividends.get("LastSplitDate"),

            # ESG Scores (optional)
            "esg_score": esg_scores.get("Total"),
            "esg_environment_score": esg_scores.get("EnvironmentScore"),
            "esg_social_score": esg_scores.get("SocialScore"),
            "esg_governance_score": esg_scores.get("GovernanceScore"),

            # Latest Earnings Data (optional)
            "latest_quarter_eps": earnings[0].get("eps") if earnings else None,

            # Financial Metrics
            "total_assets": financials.get("Balance_Sheet", {}).get("totalAssets"),
            "total_liabilities": financials.get("Balance_Sheet", {}).get("totalLiabilities"),
            "total_equity": financials.get("Balance_Sheet", {}).get("totalEquity"),
            "cash_flow_operating": financials.get("Cash_Flow", {}).get("totalCashFromOperatingActivities"),
        }

        return pd.DataFrame([processed_data])

    except Exception as e:
        print(f"Error processing fundamental data: {e}")
        return pd.DataFrame()

def process_economic_events(events, intraday_data):
    """
    Process economic events data and merge with intraday data.
    :param events: List of economic events.
    :param intraday_data: DataFrame of intraday data.
    :return: Merged DataFrame with economic event features.
    """
    if not events:
        return intraday_data

    events_df = pd.DataFrame(events)
    events_df['date'] = pd.to_datetime(events_df['date'])
    
    # Aggregate economic events data by date
    events_agg = events_df.groupby('date').agg({
        'importance': 'mean',  # Average importance of events
        'actual': 'count'      # Count of events
    }).rename(columns={'importance': 'avg_importance', 'actual': 'event_count'}).reset_index()
    
    # Merge with intraday data
    intraday_data['date'] = pd.to_datetime(intraday_data['date'])
    merged_data = pd.merge(intraday_data, events_agg, on='date', how='left')
    merged_data['avg_importance'].fillna(0, inplace=True)
    merged_data['event_count'].fillna(0, inplace=True)
    
    return merged_data


def process_macroeconomic_data(macro_data, intraday_data):
    """
    Processes macroeconomic data and merges it with the intraday data.

    :param macro_data: List of macroeconomic data dictionaries
    :param intraday_data: DataFrame of intraday data
    :return: DataFrame merged with macroeconomic features
    """
    macro_df = pd.DataFrame(macro_data)
    macro_df['date'] = pd.to_datetime(macro_df['date'])  # Ensure date is in datetime format
    macro_df = macro_df[['date', 'open', 'high', 'low', 'close']]  # Select relevant features
    macro_df.rename(columns={
        'open': 'macro_open',
        'high': 'macro_high',
        'low': 'macro_low',
        'close': 'macro_close'
    }, inplace=True)

    # Merge macroeconomic data with intraday data
    intraday_data['date'] = pd.to_datetime(intraday_data['date'])
    merged_data = pd.merge(intraday_data, macro_df, on='date', how='left')
    merged_data.fillna(0, inplace=True)  # Fill missing values with 0 or another strategy

    return merged_data



def hyperparameter_tuning(model, param_grid, X_train, y_train, search_type="random"):
    try:
        if search_type == "random":
            search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=3, random_state=42, scoring="neg_mean_squared_error")
        elif search_type == "grid":
            search = GridSearchCV(model, param_grid=param_grid, cv=3, scoring="neg_mean_squared_error")
        else:
            raise ValueError("search_type must be 'random' or 'grid'")
        search.fit(X_train, y_train)
        return search.best_estimator_
    except Exception as e:
        print(f"Hyperparameter tuning failed: {e}")
        return model

    
    
# Train and predict models
def train_and_predict(X_train, y_train, X_test, y_test):
    models = {
        "RandomForest": (RandomForestRegressor(), {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 10]}),
        "GradientBoosting": (GradientBoostingRegressor(), {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]}),
        "XGBoost": (XGBRegressor(), {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]}),
        "CatBoost": (CatBoostRegressor(verbose=0), {"iterations": [100, 200], "depth": [4, 6, 10]}),
        "LightGBM": (LGBMRegressor(), {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1]})
    }
    predictions = {}

    # Train and predict for each model
    for name, (model, param_grid) in models.items():
        print(f"Training {name}...")
        best_model = hyperparameter_tuning(model, param_grid, X_train, y_train)
        predictions[name] = best_model.predict(X_test)

    # LSTM model
    print("Training LSTM...")
    timesteps = 10
    X_train_seq, y_train_seq = create_sequences(X_train.values, y_train.values, timesteps)
    X_test_seq, y_test_seq = create_sequences(X_test.values, y_test.values, timesteps)

    lstm_model = Sequential([
        LSTM(50, activation="relu", input_shape=(timesteps, X_train_seq.shape[2])),
        Dense(1)
    ])
    lstm_model.compile(optimizer="adam", loss="mse")
    lstm_model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, verbose=0)

    predictions["LSTM"] = lstm_model.predict(X_test_seq).flatten()

    # Ensemble prediction
    ensemble_pred = np.mean(list(predictions.values()), axis=0)

    # Metrics
    mse = mean_squared_error(y_test, ensemble_pred)
    mae = mean_absolute_error(y_test, ensemble_pred)
    r2 = r2_score(y_test, ensemble_pred)
    percent_error = np.mean((ensemble_pred - y_test) / y_test) * 100

    return predictions, {"mse": mse, "mae": mae, "r2": r2, "percent_error": percent_error}


# Helper: Create LSTM sequences
def create_sequences(data, target, timesteps):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i + timesteps])
        y.append(target[i + timesteps])
    return np.array(X), np.array(y)


# Save predictions to RDS
def save_to_rds(predictions, metrics):
    connection = pg8000.connect(
        host=RDS_HOST,
        port=RDS_PORT,
        user=RDS_USER,
        password=RDS_PASSWORD
    )
    try:
        with connection.cursor() as cursor:
            for model_name, pred_values in predictions.items():
                for i, pred in enumerate(pred_values):
                    sql = """
                    INSERT INTO predictions (
                        model_name, prediction, actual, mse, mae, r2, percent_error
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(sql, (
                        model_name,
                        float(pred),
                        float(metrics["actual"][i]),
                        float(metrics["mse"]),
                        float(metrics["mae"]),
                        float(metrics["r2"]),
                        float(metrics["percent_error"])
                    ))
        connection.commit()
    except Exception as e:
        print(f"Error saving to database: {str(e)}")
    finally:
        connection.close()


# Main prediction endpoint
@app.route("/predict", methods=["POST"])
def predict_next_close():
    try:
        request_data = request.get_json()
        tickers = request_data.get("tickers", ["AAPL.US"])
        start_date = request_data.get("start_date", "2020-11-01")
        end_date = request_data.get("end_date", "2025-01-18")
        interval = request_data.get("interval", "1h")

        results = []

        for ticker in tickers:
            try:
                # Fetch intraday data
                intraday_json = get_intraday_data(ticker, start_date, end_date, interval, EOD_API_KEY)
                intraday_data = json.loads(intraday_json)
                df_intraday = pd.DataFrame(intraday_data)

                # Fetch and process macroeconomic and economic events data
                macroeconomic_data = fetch_macroeconomic_data()
                economic_events = fetch_economic_events(start_date, end_date)
                processed_intraday = process_macroeconomic_data(macroeconomic_data, df_intraday)
                processed_data = process_economic_events(economic_events, processed_intraday)

                # Feature engineering
                processed_data['target'] = processed_data['intraday_close_mean'].shift(-1)
                processed_data.dropna(inplace=True)

                # Add economic event features to the training data
                X = processed_data[[
                    "intraday_vol_mean", "intraday_high_mean", "intraday_low_mean",
                    "intraday_close_mean", "macro_open", "macro_high", "macro_low", "macro_close",
                    "avg_importance", "event_count"
                ]]
                y = processed_data["target"]

                # Train-test split
                split_index = int(len(processed_data) * 0.8)
                X_train, X_test = X[:split_index], X[split_index:]
                y_train, y_test = y[:split_index], y[split_index:]

                # Train models
                models = {
                    "RandomForest": RandomForestRegressor(random_state=42),
                    "GradientBoosting": GradientBoostingRegressor(random_state=42),
                    "XGBoost": XGBRegressor(random_state=42),
                    "CatBoost": CatBoostRegressor(verbose=0, random_state=42),
                    "LightGBM": LGBMRegressor(random_state=42),
                }
                predictions = {}

                for model_name, model in models.items():
                    model.fit(X_train, y_train)
                    predictions[model_name] = model.predict(X_test)

                # LSTM Model
                lstm_model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                    LSTM(50, return_sequences=False),
                    Dense(1)
                ])
                lstm_model.compile(optimizer="adam", loss="mse")
                lstm_model.fit(np.expand_dims(X_train, axis=-1), y_train, epochs=10, batch_size=32, verbose=0)
                lstm_predictions = lstm_model.predict(np.expand_dims(X_test, axis=-1))

                # Ensemble
                ensemble_prediction = np.mean(list(predictions.values()) + [lstm_predictions.flatten()], axis=0)

                # Metrics
                mse = mean_squared_error(y_test, ensemble_prediction)
                mae = mean_absolute_error(y_test, ensemble_prediction)
                r2 = r2_score(y_test, ensemble_prediction)

                # Save Results
                prediction = {
                    "ticker": ticker,
                    "date": str(datetime.now().date()),
                    "rf_prediction": float(predictions["RandomForest"][-1]),
                    "gbm_prediction": float(predictions["GradientBoosting"][-1]),
                    "xgb_prediction": float(predictions["XGBoost"][-1]),
                    "catboost_prediction": float(predictions["CatBoost"][-1]),
                    "lightgbm_prediction": float(predictions["LightGBM"][-1]),
                    "lstm_prediction": float(lstm_predictions[-1]),
                    "ensemble_prediction": float(ensemble_prediction[-1]),
                    "mse": mse,
                    "mae": mae,
                    "r2_score": r2
                }
                results.append(prediction)
                save_to_rds([prediction])

            except Exception as e:
                return jsonify({"error": f"Error for {ticker}: {str(e)}"}), 500

        return jsonify(results), 200

    except Exception as e:
        return 'jsonify{"error": str(e)}, 500'



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
