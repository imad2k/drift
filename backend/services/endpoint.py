import requests
import pandas as pd
import json
import re

# def fetch_and_fix_fundamentals(ticker, api_token):
#     url = f"https://eodhd.com/api/fundamentals/{ticker}?api_token={api_token}&fmt=json"
#     response = requests.get(url)
#     if response.status_code == 200:
#         raw_text = response.text
#         print("DEBUG: Raw fundamentals length =", len(raw_text))

#         # 1) Regex to fix `}}Words:{` -> `}}, "Words":{`
#         pattern = r'(\}\})\s*([A-Za-z0-9_]+)\s*:\s*\{'
#         replacement = r'}}, "\2": {'
#         fixed_text = re.sub(pattern, replacement, raw_text)

#         # 2) Attempt JSON parse
#         try:
#             data = json.loads(fixed_text)
#             return data
#         except json.JSONDecodeError as e:
#             print("Still invalid JSON after fix. Error:", e)
#             # Optionally return raw_text so you can debug further
#             return {}

#     else:
#         print(f"HTTP Error: {response.status_code}")
#         return {}

# # Usage
# api_key = "..."
# ticker = "AAPL.US"
# fund_data = fetch_and_fix_fundamentals(ticker, api_key)
# if fund_data:
#     print("Got fundamentals (possibly fixed)!")
# else:
#     print("No valid fundamentals after fix.")



# def process_fundamental_data(fundamental_json):
#     """
#     Process the fundamental data JSON into a DataFrame.
#     """
#     try:
#         # Debugging: Print the structure of the JSON data
#         print("Fundamental JSON structure:")
#         print(json.dumps(fundamental_json, indent=2))
        
#         highlights = fundamental_json.get("Highlights", {})
#         valuation = fundamental_json.get("Valuation", {})
#         shares_stats = fundamental_json.get("SharesStats", {})
#         technicals = fundamental_json.get("Technicals", {})
#         splits_dividends = fundamental_json.get("SplitsDividends", {})
#         esg_scores = fundamental_json.get("ESGScores", {})
#         earnings = fundamental_json.get("Earnings", {}).get("History", [])
#         financials = fundamental_json.get("Financials", {})

#         # Debugging: Print the contents of each section
#         print("Highlights:")
#         print(json.dumps(highlights, indent=2))
#         print("Valuation:")
#         print(json.dumps(valuation, indent=2))
#         print("Shares Stats:")
#         print(json.dumps(shares_stats, indent=2))
#         print("Technicals:")
#         print(json.dumps(technicals, indent=2))
#         print("Splits and Dividends:")
#         print(json.dumps(splits_dividends, indent=2))
#         print("ESG Scores:")
#         print(json.dumps(esg_scores, indent=2))
#         print("Earnings:")
#         print(json.dumps(earnings, indent=2))
#         print("Financials:")
#         print(json.dumps(financials, indent=2))

#         # Combine the extracted fields into a single dictionary
#         combined_data = {
#             # General Information
#             "market_cap": highlights.get("MarketCapitalization"),
#             "ebitda": highlights.get("EBITDA"),
#             "pe_ratio": highlights.get("PERatio"),
#             "peg_ratio": highlights.get("PEGRatio"),
#             "eps": highlights.get("EarningsShare"),
#             "book_value": highlights.get("BookValue"),
#             "dividend_yield": highlights.get("DividendYield"),
#             "profit_margin": highlights.get("ProfitMargin"),
#             "operating_margin": highlights.get("OperatingMargin"),
#             "return_on_assets": highlights.get("ReturnOnAssets"),
#             "return_on_equity": highlights.get("ReturnOnEquity"),
#             "revenue": highlights.get("Revenue"),
#             "revenue_per_share": highlights.get("RevenuePerShare"),
#             "gross_profit": highlights.get("GrossProfit"),
#             "diluted_eps": highlights.get("DilutedEps"),
#             "quarterly_earnings_growth": highlights.get("QuarterlyEarningsGrowthYOY"),

#             # Valuation Metrics
#             "trailing_pe": valuation.get("TrailingPE"),
#             "forward_pe": valuation.get("ForwardPE"),
#             "price_to_sales": valuation.get("PriceSales"),
#             "price_to_book": valuation.get("PriceBook"),
#             "enterprise_value_revenue": valuation.get("EnterpriseValueRevenue"),
#             "enterprise_value_ebitda": valuation.get("EnterpriseValueEbitda"),

#             # Share Statistics
#             "shares_outstanding": shares_stats.get("SharesOutstanding"),
#             "shares_float": shares_stats.get("SharesFloat"),
#             "percent_held_by_insiders": shares_stats.get("PercentInsiders"),
#             "percent_held_by_institutions": shares_stats.get("PercentInstitutions"),

#             # Technical Indicators
#             "beta": technicals.get("Beta"),
#             "52_week_high": technicals.get("52WeekHigh"),
#             "52_week_low": technicals.get("52WeekLow"),
#             "50_day_moving_avg": technicals.get("50DayMA"),
#             "200_day_moving_avg": technicals.get("200DayMA"),

#             # Splits and Dividends
#             "dividend_rate": splits_dividends.get("ForwardAnnualDividendRate"),
#             "dividend_payout_ratio": splits_dividends.get("PayoutRatio"),
#             "last_dividend_date": splits_dividends.get("LastSplitDate"),

#             # ESG Scores (optional)
#             "esg_score": esg_scores.get("Total"),
#             "esg_environment_score": esg_scores.get("EnvironmentScore"),
#             "esg_social_score": esg_scores.get("SocialScore"),
#             "esg_governance_score": esg_scores.get("GovernanceScore"),

#             # Latest Earnings Data (optional)
#             "latest_quarter_eps": earnings[0].get("eps") if earnings else None,

#             # Financial Metrics
#             "total_assets": financials.get("Balance_Sheet", {}).get("totalAssets"),
#             "total_liabilities": financials.get("Balance_Sheet", {}).get("totalLiabilities"),
#             "total_equity": financials.get("Balance_Sheet", {}).get("totalEquity"),
#             "cash_flow_operating": financials.get("Cash_Flow", {}).get("totalCashFromOperatingActivities"),
#         }

#         # Debugging: Print the combined data
#         print("Combined data:")
#         print(json.dumps(combined_data, indent=2))
        
#         # Convert the combined dictionary into a DataFrame
#         df_fund = pd.DataFrame([combined_data])
        
#         if df_fund.empty:
#             print("Fundamental data DataFrame is empty.")
        
#         return df_fund
#     except Exception as e:
#         print(f"Error processing fundamental data: {e}")
#         return pd.DataFrame()

# Test the functions
EOD_API_KEY = "62fc32b65388e2.02738383"
ticker = "AAPL.US"


import os
import re
import json
import requests
import pandas as pd

################################################################################
# 1. Fetch & Fix Function
################################################################################
def fetch_and_fix_fundamentals(ticker, api_token):
    """
    Fetch raw fundamentals JSON from EODHD for `ticker`, then fix any occurrences 
    of '}}SomeWord:{' -> '}}, "SomeWord": {' or variations with extra spaces. 
    Finally, return a Python dictionary (or {} if still invalid).
    """
    url = f"https://eodhd.com/api/fundamentals/{ticker}?api_token={api_token}&fmt=json"
    print(f"Requesting URL: {url}")
    response = requests.get(url)

    if response.status_code != 200:
        print(f"HTTP Error: {response.status_code} - {response.reason}")
        return {}

    # 1) Get the raw text
    raw_text = response.text
    print("DEBUG: Raw fundamentals length =", len(raw_text))
    # (Optional) print a snippet to see if it's obviously malformed
    print("DEBUG: Raw fundamentals snippet:")
    print(raw_text[:1000])  # first 1000 chars

    # 2) Regex to fix known patterns like '}}Word:{' (with possible spaces)
    # This pattern finds:
    pattern = r'(\}\})\s*([A-Za-z0-9_]+)\s*:\s*\{'
    replacement = r'}}, "\2": {'
    fixed_text = re.sub(pattern, replacement, raw_text)

    # Possibly, there could be multiple weird spots; re.sub fixes them all at once

    # 3) Attempt to parse
    try:
        data = json.loads(fixed_text)
        print("DEBUG: JSON successfully parsed after fix!")
        return data
    except json.JSONDecodeError as e:
        print("ERROR: Still invalid JSON after fix. JSONDecodeError details:")
        print(e)
        # Return an empty dict if we can't parse
        return {}

################################################################################
# 2. The process_fundamental_data Function
################################################################################
def process_fundamental_data(fundamental_json):
    """
    Process the fundamental data JSON into a DataFrame.
    Uses debug prints to show sections of data.
    """
    import json

    try:
        # Debugging: Print the structure of the JSON data
        print("\n--- process_fundamental_data Debug ---")
        print("Fundamental JSON structure (final):")
        print(json.dumps(fundamental_json, indent=2))
        
        highlights = fundamental_json.get("Highlights", {})
        valuation = fundamental_json.get("Valuation", {})
        shares_stats = fundamental_json.get("SharesStats", {})
        technicals = fundamental_json.get("Technicals", {})
        splits_dividends = fundamental_json.get("SplitsDividends", {})
        esg_scores = fundamental_json.get("ESGScores", {})
        earnings = fundamental_json.get("Earnings", {}).get("History", [])
        financials = fundamental_json.get("Financials", {})

        # Debug: Print the contents of each section
        print("Highlights:", json.dumps(highlights, indent=2))
        print("Valuation:", json.dumps(valuation, indent=2))
        print("Shares Stats:", json.dumps(shares_stats, indent=2))
        print("Technicals:", json.dumps(technicals, indent=2))
        print("Splits and Dividends:", json.dumps(splits_dividends, indent=2))
        print("ESG Scores:", json.dumps(esg_scores, indent=2))
        print("Earnings:", json.dumps(earnings, indent=2))
        print("Financials:", json.dumps(financials, indent=2))

        combined_data = {
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

            # ESG Scores
            "esg_score": esg_scores.get("Total"),
            "esg_environment_score": esg_scores.get("EnvironmentScore"),
            "esg_social_score": esg_scores.get("SocialScore"),
            "esg_governance_score": esg_scores.get("GovernanceScore"),

            # Latest Earnings Data
            "latest_quarter_eps": earnings[0].get("eps") if (isinstance(earnings, list) and len(earnings) > 0 and isinstance(earnings[0], dict)) else None,

            # Financial Metrics
            "total_assets": financials.get("Balance_Sheet", {}).get("totalAssets"),
            "total_liabilities": financials.get("Balance_Sheet", {}).get("totalLiabilities"),
            "total_equity": financials.get("Balance_Sheet", {}).get("totalEquity"),
            "cash_flow_operating": financials.get("Cash_Flow", {}).get("totalCashFromOperatingActivities"),
        }

        # Print combined data
        print("\nCombined data dictionary:")
        print(json.dumps(combined_data, indent=2))

        df_fund = pd.DataFrame([combined_data])
        if df_fund.empty:
            print("\n>>> Fundamental data DataFrame is empty.\n")
        return df_fund

    except Exception as e:
        print(f"Error processing fundamental data: {e}")
        return pd.DataFrame()
    
    


################################################################################
# 3. Main Test
################################################################################
if __name__ == "__main__":
    # You can store your API key in an environment variable or paste it here.
    EOD_API_KEY = "62fc32b65388e2.02738383"
    ticker = "AAPL.US"  # change as needed

    print(f"\n===== Fetching & Fixing Fundamentals for: {ticker} =====")
    raw_data = fetch_and_fix_fundamentals(ticker, EOD_API_KEY)

    print("\n===== Processing Fundamental Data =====")
    df_fund = process_fundamental_data(raw_data)

    print("\n===== Final DataFrame =====")
    print(df_fund)
    print("\nDone.")


# # Fetch fundamental data
# fundamental_data = fetch_fundamental_data(ticker, EOD_API_KEY)

# # Process the fetched data
# df_fund = process_fundamental_data(fundamental_data)

# # Print the processed DataFrame
# print("Processed DataFrame:")
# print(df_fund)