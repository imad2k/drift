# stock_prediction_app/data/data_processing.py

import pandas as pd
import numpy as np

def process_external_data(fundamental_data, news_sentiment, economic_events, macroeconomic_data):
    """
    Returns a single-row DataFrame containing aggregated external data
    (fundamentals, sentiment, etc.). Even if not used directly in the final
    route, we keep it for advanced scenarios.
    """
    data = {}

    # -------------------
    # Fundamental Data
    # -------------------
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

    # -------------------
    # News Sentiment
    # -------------------
    if news_sentiment:
        if isinstance(news_sentiment, pd.DataFrame):
            data["news_sentiment_score"] = news_sentiment.get("normalized", pd.Series([0])).mean()
            data["news_sentiment_count"] = len(news_sentiment)
        elif isinstance(news_sentiment, list):
            data["news_sentiment_score"] = np.mean([item.get("normalized", 0) for item in news_sentiment])
            data["news_sentiment_count"] = len(news_sentiment)
    else:
        data["news_sentiment_score"] = np.nan
        data["news_sentiment_count"] = 0

    # -------------------
    # Economic Events
    # -------------------
    if isinstance(economic_events, list) and economic_events:
        high_impact = [ev for ev in economic_events if ev.get("impact") == "High"]
        data["economic_event_high_impact_count"] = len(high_impact)
        data["economic_event_average_change"] = np.mean([ev.get("change_percentage", 0) for ev in economic_events])
    else:
        data["economic_event_high_impact_count"] = 0
        data["economic_event_average_change"] = np.nan

    # -------------------
    # Macroeconomic Data
    # -------------------
    # If you want to do advanced summarization, you could parse macro here.
    # Right now, we just store a placeholder row.
    if macroeconomic_data:
        data["macro_data_found"] = True
    else:
        data["macro_data_found"] = False

    return pd.DataFrame([data])


def aggregate_intraday(intraday_df):
    """
    Aggregates intraday data by date, computing mean volume/high/low/close.
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


def process_fundamental_data(fundamental_json):
    """
    Processes the fundamental data JSON into a single-row DataFrame.
    """
    try:
        highlights = fundamental_json.get("Highlights", {})
        valuation = fundamental_json.get("Valuation", {})
        shares_stats = fundamental_json.get("SharesStats", {})
        technicals = fundamental_json.get("Technicals", {})
        splits_dividends = fundamental_json.get("SplitsDividends", {})
        esg_scores = fundamental_json.get("ESGScores", {})
        earnings = fundamental_json.get("Earnings", {}).get("History", [])
        financials = fundamental_json.get("Financials", {})

        combined_data = {
            # General
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

            # Valuation
            "trailing_pe": valuation.get("TrailingPE"),
            "forward_pe": valuation.get("ForwardPE"),
            "price_to_sales": valuation.get("PriceSales"),
            "price_to_book": valuation.get("PriceBook"),
            "enterprise_value_revenue": valuation.get("EnterpriseValueRevenue"),
            "enterprise_value_ebitda": valuation.get("EnterpriseValueEbitda"),

            # Shares
            "shares_outstanding": shares_stats.get("SharesOutstanding"),
            "shares_float": shares_stats.get("SharesFloat"),
            "percent_held_by_insiders": shares_stats.get("PercentInsiders"),
            "percent_held_by_institutions": shares_stats.get("PercentInstitutions"),

            # Technicals
            "beta": technicals.get("Beta"),
            "52_week_high": technicals.get("52WeekHigh"),
            "52_week_low": technicals.get("52WeekLow"),
            "50_day_moving_avg": technicals.get("50DayMA"),
            "200_day_moving_avg": technicals.get("200DayMA"),

            # Splits / Dividends
            "dividend_rate": splits_dividends.get("ForwardAnnualDividendRate"),
            "dividend_payout_ratio": splits_dividends.get("PayoutRatio"),
            "last_dividend_date": splits_dividends.get("LastSplitDate"),

            # ESG
            "esg_score": esg_scores.get("Total"),
            "esg_environment_score": esg_scores.get("EnvironmentScore"),
            "esg_social_score": esg_scores.get("SocialScore"),
            "esg_governance_score": esg_scores.get("GovernanceScore"),

            # Earnings
            "latest_quarter_eps": earnings[0].get("eps") if earnings else None,

            # Financials
            "total_assets": financials.get("Balance_Sheet", {}).get("totalAssets"),
            "total_liabilities": financials.get("Balance_Sheet", {}).get("totalLiabilities"),
            "total_equity": financials.get("Balance_Sheet", {}).get("totalEquity"),
            "cash_flow_operating": financials.get("Cash_Flow", {}).get("totalCashFromOperatingActivities"),
        }

        df_fund = pd.DataFrame([combined_data])
        if df_fund.empty:
            print("Fundamental data DataFrame is empty.")
        return df_fund

    except Exception as e:
        print(f"Error processing fundamental data: {e}")
        return pd.DataFrame()


def process_economic_events(events, intraday_data):
    """
    Merges aggregated economic events by date with intraday data (already aggregated).
    """
    if not events:
        return intraday_data

    events_df = pd.DataFrame(events)
    if 'date' not in events_df.columns:
        return intraday_data

    events_df['date'] = pd.to_datetime(events_df['date'])
    events_agg = events_df.groupby('date').agg({
        'importance': 'mean',
        'actual': 'count'
    }).rename(columns={'importance': 'avg_importance', 'actual': 'event_count'}).reset_index()

    intraday_data['date'] = pd.to_datetime(intraday_data['date'])
    merged_data = pd.merge(intraday_data, events_agg, on='date', how='left')
    merged_data['avg_importance'].fillna(0, inplace=True)
    merged_data['event_count'].fillna(0, inplace=True)
    
    return merged_data


def process_macroeconomic_data(macro_data, intraday_data):
    """
    Merges daily macro data (open/high/low/close) with daily intraday data.
    """
    macro_df = pd.DataFrame(macro_data)
    if 'date' not in macro_df.columns:
        # Return unmodified if no valid macro data
        return intraday_data

    macro_df['date'] = pd.to_datetime(macro_df['date'])
    macro_df = macro_df[['date', 'open', 'high', 'low', 'close']]
    macro_df.rename(columns={
        'open': 'macro_open',
        'high': 'macro_high',
        'low': 'macro_low',
        'close': 'macro_close'
    }, inplace=True)

    intraday_data['date'] = pd.to_datetime(intraday_data['date'])
    merged_data = pd.merge(intraday_data, macro_df, on='date', how='left')
    merged_data.fillna(0, inplace=True)

    return merged_data
