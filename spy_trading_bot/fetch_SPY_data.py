import csv
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to prepare the dataset
def prepare_stock_data(output_file):
    """
    Fetches and processes stock data with technical indicators, sentiment analysis,
    and target creation. Saves the processed data as a CSV file.
    """
    # Check if the processed data file already exists
    if os.path.exists(output_file):
        print(f"Loading data from {output_file}...")
        data = pd.read_csv(output_file, index_col=0, parse_dates=True, date_format='%Y-%m-%d')
    else:
        print("Fetching data...")
        # Step 1: Fetch stock data using Yahoo Finance for SPY, USD/JPY, VIX, Gold, and Oil
        tickers = ["SPY", "JPY=X", "^VIX", "GC=F", "CL=F"]
        data = yf.download(tickers, interval="1d", period="max")

        # Step 2: Add Technical Indicators for SPY
        print("Calculating technical indicators...")
        data['SMA_14'] = data['Adj Close']['SPY'].rolling(window=14).mean()
        data['EMA_14'] = ta.ema(data['Adj Close']['SPY'], length=14)
        data['RSI'] = ta.rsi(data['Adj Close']['SPY'], length=14)
        data['MACD'] = ta.macd(data['Adj Close']['SPY']).iloc[:, 0]
        bb = ta.bbands(data['Adj Close']['SPY'], length=20)
        data['BB_upper'], data['BB_middle'], data['BB_lower'] = bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2]
        data['ATR'] = ta.atr(data['High']['SPY'], data['Low']['SPY'], data['Close']['SPY'], length=14)
        stoch = ta.stoch(data['High']['SPY'], data['Low']['SPY'], data['Close']['SPY'])
        data['Stoch_K'], data['Stoch_D'] = stoch.iloc[:, 0], stoch.iloc[:, 1]

        # Step 3: Add Market Indicators (USD/JPY exchange rate, VIX index, Gold, and Oil prices)
        print("Adding market indicators...")
        data['USD_JPY'] = data['Adj Close']['JPY=X']
        data['VIX'] = data['Adj Close']['^VIX']
        data['Gold'] = data['Adj Close']['GC=F']
        data['Oil'] = data['Adj Close']['CL=F']

        # Step 4: Add Returns and Price Change for SPY
        print("Calculating returns and price change...")
        data['Monthly_Return'] = data['Adj Close']['SPY'].pct_change()
        data['Price_Change'] = (data['Adj Close']['SPY'] - data['Open']['SPY']) / data['Open']['SPY']

        # Step 6: Create Target Variable (Binary: Up/Down)
        data['Target'] = (data['Adj Close']['SPY'].shift(-1) > data['Adj Close']['SPY']).astype(int)

        # Step 7: Drop rows with NaN values
        data = data.dropna()

        # Step 8: Save the processed DataFrame to a CSV file
        data.to_csv(output_file)
        print(f"Data saved to {output_file}.")

    return data


# Main block
if __name__ == "__main__":
    output_file = "data/spy_daily_data.csv"

    # Prepare the stock data
    processed_data = prepare_stock_data(output_file)

    # Display relevant columns from the DataFrame
    print("\nFinal DataFrame:")
    print(processed_data[['Adj Close', 'SMA_14', 'EMA_14', 'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR', 'Stoch_K', 'Stoch_D', 'USD_JPY', 'VIX', 'Gold', 'Oil', 'Monthly_Return', 'Target']])