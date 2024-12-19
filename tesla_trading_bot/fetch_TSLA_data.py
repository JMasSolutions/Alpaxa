import csv
import requests
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from helpful_functions.finbert_utils import estimate_sentiment
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


# Function to fetch news sentiment using Alpaca News API with rate limiting and retries
def get_sentiment_with_rate_limit(symbol: str, start: str, end: str, api_key, secret_key, retries: int = 3):
    """
    Fetch sentiment scores for a stock symbol using Alpaca API.
    Implements rate limiting with retries to avoid 429 errors.
    """
    url = "https://data.alpaca.markets/v1beta1/news"  # Alpaca news endpoint
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }
    params = {"symbols": symbol, "start": start, "end": end}

    # Retry logic to handle rate limits or API errors
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:  # Successful request
                news_items = response.json().get("news", [])
                news_headlines = [item.get("headline", "No headline") for item in news_items]
                if news_headlines:  # Sentiment estimation only if headlines exist
                    probability, sentiment = estimate_sentiment(news_headlines)
                    return probability if sentiment == "positive" else -probability
                return 0.0  # Default to neutral sentiment if no headlines exist
            elif response.status_code == 429:  # Rate limit exceeded
                print(f"Rate limit hit. Sleeping for 60 seconds... (Attempt {attempt+1}/{retries})")
                time.sleep(60)  # Wait for 60 seconds before retrying
            else:
                print(f"Error fetching news for {start}-{end}: {response.status_code}")
                return 0.0  # Default to neutral sentiment on failure
        except Exception as e:
            print(f"Exception on attempt {attempt+1}: {e}")
            time.sleep(60)  # Wait before retrying
    return 0.0  # Return neutral sentiment after retries are exhausted


# Function to prepare the dataset
def prepare_stock_data(output_file, api_key, secret_key):
    """
    Fetches and processes stock data with technical indicators, sentiment analysis,
    and target creation. Saves the processed data as a CSV file.
    """
    # Check if the processed data file already exists
    if os.path.exists(output_file):
        print(f"Loading data from {output_file}...")
        data = pd.read_csv(output_file, index_col=0, parse_dates=True)
    else:
        print("Fetching data...")
        # Step 1: Fetch stock data using Yahoo Finance for TSLA, USD/JPY, and VIX
        tickers = ["TSLA", "JPY=X", "^VIX"]
        data = yf.download(tickers, interval="1mo", period="max")

        # Step 2: Add Technical Indicators for TSLA
        print("Calculating technical indicators...")
        data['SMA_14'] = data['Adj Close']['TSLA'].rolling(window=14).mean()
        data['EMA_14'] = ta.ema(data['Adj Close']['TSLA'], length=14)
        data['RSI'] = ta.rsi(data['Adj Close']['TSLA'], length=14)
        data['MACD'] = ta.macd(data['Adj Close']['TSLA']).iloc[:, 0]

        # Step 3: Add Market Indicators (USD/JPY exchange rate and VIX index)
        print("Adding market indicators...")
        data['USD_JPY'] = data['Adj Close']['JPY=X']
        data['VIX'] = data['Adj Close']['^VIX']

        # Step 4: Add Returns and Price Change for TSLA
        print("Calculating returns and price change...")
        data['Monthly_Return'] = data['Adj Close']['TSLA'].pct_change()
        data['Price_Change'] = (data['Adj Close']['TSLA'] - data['Open']['TSLA']) / data['Open']['TSLA']

        # Step 5: Fetch sentiment data for TSLA with parallel processing
        print("Fetching sentiment data with rate limiting...")
        sentiment_scores = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    get_sentiment_with_rate_limit,
                    "TSLA",
                    date.strftime("%Y-%m-%d"),
                    (date + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d"),
                    api_key,
                    secret_key,
                ): date
                for date in data.index
            }
            for future in as_completed(futures):
                sentiment_scores.append(future.result())
                time.sleep(1)

        # Step 6: Add sentiment scores to the DataFrame
        data["Sentiment_Score"] = sentiment_scores

        # Step 7: Create Target Variable (Binary: Up/Down)
        data['Target'] = (data['Adj Close']['TSLA'].shift(-1) > data['Adj Close']['TSLA']).astype(int)

        # Step 8: Drop rows with NaN values
        data = data.dropna()

        # Step 9: Save the processed DataFrame to a CSV file
        data.to_csv(output_file)
        print(f"Data saved to {output_file}.")

    return data


# Main block
if __name__ == "__main__":
    # Path to the CSV file containing Alpaca API credentials
    file_path = "/Users/joanmascastella/Documents/ALPAXA/API_KEYS.csv"
    output_file = "data/tsla_monthly_sentiment_data.csv"

    # Read API credentials
    with open(file_path, "r") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        row = next(csv_reader)
        endpoint, api_key, secret_key = [value.strip() for value in row]

    # Prepare the stock data
    processed_data = prepare_stock_data(output_file, api_key, secret_key)

    # Display relevant columns from the DataFrame
    print("\nFinal DataFrame:")
    print(processed_data[['Adj Close', 'SMA_14', 'RSI', 'MACD', 'USD_JPY', 'VIX', 'Monthly_Return', 'Sentiment_Score', 'Target']])