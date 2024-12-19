import csv
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        data = yf.download(tickers, interval="1d", period="max")

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


        # Step 6: Create Target Variable (Binary: Up/Down)
        data['Target'] = (data['Adj Close']['TSLA'].shift(-1) > data['Adj Close']['TSLA']).astype(int)

        # Step 7: Drop rows with NaN values
        data = data.dropna()

        # Step 8: Save the processed DataFrame to a CSV file
        data.to_csv(output_file)
        print(f"Data saved to {output_file}.")

    return data


# Main block
if __name__ == "__main__":
    # Path to the CSV file containing Alpaca API credentials
    file_path = "/Users/joanmascastella/Documents/ALPAXA/API_KEYS.csv"
    output_file = "data/tsla_daily_data.csv"

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
    print(processed_data[['Adj Close', 'SMA_14', 'RSI', 'MACD', 'USD_JPY', 'VIX', 'Monthly_Return', 'Target']])