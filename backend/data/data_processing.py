# stock_prediction_app/processing/data_processing.py

import pandas as pd
import numpy as np

def process_external_data(fundamental_data, news_sentiment, economic_events, macroeconomic_data):
    """
    Original function from your script. It returns a single-row DataFrame
    containing aggregated external data (fundamentals, sentiment, etc.).
    Even if not currently used in the final route, we keep it.
    """
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
    if isinstance(news_sentiment, (pd.DataFrame, list)):
        # If news_sentiment is a DataFrame or list of dict
        if isinstance(news_sentiment, pd.DataFrame):
            # Possibly use 'normalized' or something
            data["news_sentiment_score"] = news_sentiment.get("normalized", pd.Series([0])).mean()
            data["news_sentiment_count"] = len(news_sentiment)
        else:
            # list-based approach
            data["news_sentiment_score"] = np.mean([item.get("normalized", 0) for item in news_sentiment])
            data["news_sentiment_count"] = len(news_sentiment)

    # Process Economic Events
    if isinstance(economic_events, list) and len(economic_events) > 0:
        high_impact_events = [ev for ev in economic_events if ev.get("impact") == "High"]
        data["economic_event_high_impact_count"] = len(high_impact_events)
        data["economic_event_average_change"] = np.mean([
            ev.get("change_percentage", 0) for ev in economic_events
        ])

    # Process Macroeconomic Data
    if isinstance(macroeconomic_data, list) and len(macroeconomic_data) > 0:
        # Example: look for certain indicators
        gdp_growth = next((item for item in macroeconomic_data if item.get("indicator") == "GDP Growth"), {})
        inflation_rate = next((item for item in macroeconomic_data if item.get("indicator") == "Inflation Rate"), {})
        unemployment_rate = next((item for item in macroeconomic_data if item.get("indicator") == "Unemployment Rate"), {})
        
        data["gdp_growth_rate"] = gdp_growth.get("value", np.nan)
        data["inflation_rate"] = inflation_rate.get("value", np.nan)
        data["unemployment_rate"] = unemployment_rate.get("value", np.nan)

    return pd.DataFrame([data])


def aggregate_intraday(intraday_df):
    """
    Aggregate intraday data from your original script.
    Groups by date and calculates mean volume/high/low/close.
    """
    intraday_df["timestamp"] = pd.to_datetime(intraday_df["timestamp"], unit='s')
    intraday_df["date"] = intraday_df["timestamp"].dt.date
    grouped = intraday_df.groupby("date").agg({
        'volume': 'mean',
        'high': 'mean',
        'low': 'mean',
        'close': 'mean'
    }).reset_index()
    grouped.rename(columns={
        'volume': 'intraday_vol_mean',
        'high': 'intraday_high_mean',
        'low': 'intraday_low_mean',
        'close': 'intraday_close_mean'
    }, inplace=True)
    return grouped


def process_fundamental_data(fundamental_data):
    """
    Processes fundamental data into a single-row DataFrame. 
    Kept exactly from your script.
    """
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
    Original code merges event data with intraday data on 'date'.
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
    Merges macro data (open, high, low, close) with intraday data on 'date'.
    """
    macro_df = pd.DataFrame(macro_data)
    if 'date' in macro_df.columns:
        macro_df['date'] = pd.to_datetime(macro_df['date'])
        macro_df = macro_df[['date', 'open', 'high', 'low', 'close']]
        macro_df.rename(columns={
            'open': 'macro_open',
            'high': 'macro_high',
            'low': 'macro_low',
            'close': 'macro_close'
        }, inplace=True)
    else:
        # If the macro_data doesn't have 'date' or is empty, we'll return intraday_data unmodified
        return intraday_data

    intraday_data['date'] = pd.to_datetime(intraday_data['date'])
    merged_data = pd.merge(intraday_data, macro_df, on='date', how='left')
    merged_data.fillna(0, inplace=True)
    return merged_data
