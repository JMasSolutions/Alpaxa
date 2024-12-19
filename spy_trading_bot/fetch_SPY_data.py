import csv
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import os
import time

# Define a consistent feature list
FEATURES = [
    "Adj Close", "SMA_14", "EMA_14", "RSI", "BB_upper", "BB_middle", "BB_lower",
    "USD_JPY", "VIX", "Gold", "Oil", "Monthly_Return", "MACD", "ATR", "OBV",
    "Adj_Close_Lag_1", "Adj_Close_Lag_2", "Adj_Close_Lag_3", "Adj_Close_Lag_5",
    "Momentum", "Volatility", "Day_of_Week", "Month"
]

# Function to calculate additional technical indicators
def calculate_additional_indicators(df):
    print("Calculating additional technical indicators...")

    # MACD
    macd = ta.macd(df['Adj Close']['SPY'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACDh_12_26_9'] if macd is not None else 0

    # ATR
    df['ATR'] = ta.atr(df['High']['SPY'], df['Low']['SPY'], df['Adj Close']['SPY'], length=14)

    # OBV
    df['OBV'] = ta.obv(df['Adj Close']['SPY'], df['Volume']['SPY'])

    # Volatility (Standard Deviation of Returns)
    df['Volatility'] = df['Adj Close']['SPY'].pct_change().rolling(window=14).std()

    return df

# Function to add lag features
def add_lag_features(df, lags=[1, 2, 3, 5]):
    print("Adding lag features...")
    for lag in lags:
        df[f'Adj_Close_Lag_{lag}'] = df['Adj Close']['SPY'].shift(lag)
    return df

# Function to calculate momentum
def calculate_momentum(df, window=10):
    print("Calculating momentum indicators...")
    df['Momentum'] = ta.mom(df['Adj Close']['SPY'], length=window)
    return df

# Function to add date-based features
def add_date_features(df):
    print("Adding date-based features...")
    df['Day_of_Week'] = df.index.dayofweek  # 0=Monday, 6=Sunday
    df['Month'] = df.index.month
    return df

# Function to prepare the dataset
def prepare_stock_data(output_file):
    if os.path.exists(output_file):
        print(f"Loading data from {output_file}...")
        data = pd.read_csv(output_file, index_col=0, parse_dates=True)
    else:
        print("Fetching data...")
        tickers = ["SPY", "JPY=X", "^VIX", "GC=F", "CL=F"]
        data = yf.download(tickers, interval="1d", period="5y")

        # Technical Indicators
        print("Calculating technical indicators...")
        data['SMA_14'] = data['Adj Close']['SPY'].rolling(window=14).mean()
        data['EMA_14'] = ta.ema(data['Adj Close']['SPY'], length=14)
        data['RSI'] = ta.rsi(data['Adj Close']['SPY'], length=14)
        bb = ta.bbands(data['Adj Close']['SPY'], length=20)
        data['BB_upper'] = bb['BBU_20_2.0']
        data['BB_middle'] = bb['BBM_20_2.0']
        data['BB_lower'] = bb['BBL_20_2.0']

        # Additional Indicators
        data = calculate_additional_indicators(data)

        # Lag Features
        data = add_lag_features(data)

        # Momentum
        data = calculate_momentum(data)

        # Date Features
        data = add_date_features(data)

        # Market Indicators
        print("Adding market indicators...")
        data['USD_JPY'] = data['Adj Close']['JPY=X']
        data['VIX'] = data['Adj Close']['^VIX']
        data['Gold'] = data['Adj Close']['GC=F']
        data['Oil'] = data['Adj Close']['CL=F']

        # Returns and Target
        print("Calculating returns and target...")
        data['Monthly_Return'] = data['Adj Close']['SPY'].pct_change()
        data['Target'] = (data['Adj Close']['SPY'].shift(-1) > data['Adj Close']['SPY']).astype(int)

        # Clean up NaN values
        data = data.dropna()

        # Save
        data.to_csv(output_file)
        print(f"Data saved to {output_file}.")

    return data

# Main block
if __name__ == "__main__":
    output_file = "data/spy_daily_data.csv"
    processed_data = prepare_stock_data(output_file)
    print(processed_data[FEATURES + ['Target']].head())