import csv
import requests
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from finbert_utils import estimate_sentiment
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Path to CSV File for Alpaca API keys
file_path = "/Users/joanmascastella/Documents/ALPAXA/API_KEYS.csv"
output_file = "/data/tsla_monthly_sentiment_data.csv"  # Output CSV file path

# Read API credentials
with open(file_path, "r") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header row
    row = next(csv_reader)
    endpoint, api_key, secret_key = [value.strip() for value in row]  # Strip whitespace

# Function to fetch news sentiment with rate limiting and retries
def get_sentiment_with_rate_limit(symbol: str, start: str, end: str, retries: int = 3):
    url = "https://data.alpaca.markets/v1beta1/news"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }
    params = {"symbols": symbol, "start": start, "end": end}

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                news_items = response.json().get("news", [])
                news_headlines = [item.get("headline", "No headline") for item in news_items]
                if news_headlines:
                    probability, sentiment = estimate_sentiment(news_headlines)
                    return probability if sentiment == "positive" else -probability
                return 0.0  # Default neutral sentiment if no news
            elif response.status_code == 429:
                print(f"Rate limit hit. Sleeping for 60 seconds... (Attempt {attempt+1}/{retries})")
                time.sleep(60)  # Wait before retrying
            else:
                print(f"Error fetching news for {start}-{end}: {response.status_code}")
                return 0.0
        except Exception as e:
            print(f"Exception on attempt {attempt+1}: {e}")
            time.sleep(60)
    return 0.0  # Return neutral sentiment after retries exhausted

# Fetch or load data
if os.path.exists(output_file):
    print(f"Loading data from {output_file}...")
    data = pd.read_csv(output_file, index_col=0, parse_dates=True)
else:
    print("Fetching data...")
    # Fetch stock data
    tickers = ["TSLA", "JPY=X", "^VIX"]
    data = yf.download(tickers, interval="1mo", period="max")

    # Add Technical Indicators
    data['SMA_14'] = data['Adj Close']['TSLA'].rolling(window=14).mean()
    data['EMA_14'] = ta.ema(data['Adj Close']['TSLA'], length=14)
    data['RSI'] = ta.rsi(data['Adj Close']['TSLA'], length=14)
    data['MACD'] = ta.macd(data['Adj Close']['TSLA']).iloc[:, 0]

    # Add Market Indicators
    data['USD_JPY'] = data['Adj Close']['JPY=X']
    data['VIX'] = data['Adj Close']['^VIX']

    # Add Returns
    data['Monthly_Return'] = data['Adj Close']['TSLA'].pct_change()
    data['Price_Change'] = (data['Adj Close']['TSLA'] - data['Open']['TSLA']) / data['Open']['TSLA']

    # Parallelized fetching of sentiment scores with rate limiting
    print("Fetching sentiment data with rate limiting...")
    sentiment_scores = []
    with ThreadPoolExecutor(max_workers=4) as executor:  # Limit workers to avoid rate-limiting
        futures = {
            executor.submit(
                get_sentiment_with_rate_limit,
                "TSLA",
                date.strftime("%Y-%m-%d"),
                (date + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d"),
            ): date
            for date in data.index
        }
        for future in as_completed(futures):
            sentiment_scores.append(future.result())
            time.sleep(1)

    # Add sentiment scores to the DataFrame
    data["Sentiment_Score"] = sentiment_scores

    # Create Target Variable (Binary: Up/Down)
    data['Target'] = (data['Adj Close']['TSLA'].shift(-1) > data['Adj Close']['TSLA']).astype(int)

    # Drop NaN values
    data = data.dropna()

    # Save DataFrame to CSV
    data.to_csv(output_file)
    print(f"Data saved to {output_file}.")

# Display final DataFrame
print(data[['Adj Close', 'SMA_14', 'RSI', 'MACD', 'USD_JPY', 'VIX', 'Monthly_Return', 'Sentiment_Score', 'Target']])