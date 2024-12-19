import yfinance as yf
import pandas as pd
import pandas_ta as ta
import os


# Function to flatten and rename columns
def flatten_columns(data):
    """
    Flattens a MultiIndex DataFrame into a single index and renames columns.
    """
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns]
    return data


# Function to add technical indicators
def add_technical_indicators(data, ticker):
    """
    Adds technical indicators for the specified ticker.
    """
    print(f"Adding technical indicators for {ticker}...")
    data[f"{ticker}_SMA_14"] = data[f"{ticker}_Adj Close"].rolling(window=14).mean()
    data[f"{ticker}_EMA_14"] = ta.ema(data[f"{ticker}_Adj Close"], length=14)
    data[f"{ticker}_RSI_14"] = ta.rsi(data[f"{ticker}_Adj Close"], length=14)
    bb = ta.bbands(data[f"{ticker}_Adj Close"], length=20)
    if bb is not None:
        data[f"{ticker}_BB_upper"] = bb["BBU_20_2.0"]
        data[f"{ticker}_BB_middle"] = bb["BBM_20_2.0"]
        data[f"{ticker}_BB_lower"] = bb["BBL_20_2.0"]
    else:
        print(f"Failed to calculate Bollinger Bands for {ticker}")
    macd = ta.macd(data[f"{ticker}_Adj Close"], fast=12, slow=26, signal=9)
    if macd is not None:
        data[f"{ticker}_MACD"] = macd["MACD_12_26_9"]
    return data


# Function to prepare the dataset
def prepare_stock_data(output_file):
    """
    Downloads and processes stock market data using yfinance.
    """
    if os.path.exists(output_file):
        print(f"Loading data from {output_file}...")
        data = pd.read_csv(output_file, index_col=0, parse_dates=True)
    else:
        print("Fetching data...")

        # Define tickers
        tickers = [
            "SPY", "AAPL", "MSFT", "AMZN", "NVDA", "TSLA", "META", "GOOGL",  # SPY and Big Seven
            "XLF", "XLK", "XLV", "XLE", "XLY", "XLU", "XLI",  # Sector ETFs
            "^VIX", "JPY=X", "GC=F", "CL=F"  # Macroeconomic indicators
        ]

        # Download data
        data = yf.download(tickers, interval="1d", period="max", group_by="ticker")

        # Flatten MultiIndex DataFrame
        data = flatten_columns(data)
        print(f"Flattened columns: {data.columns.tolist()}")

        # Add technical indicators for SPY
        data = add_technical_indicators(data, "SPY")

        # Add technical indicators for Big Seven
        big_seven = ["AAPL", "MSFT", "AMZN", "NVDA", "TSLA", "META", "GOOGL"]
        for ticker in big_seven:
            data = add_technical_indicators(data, ticker)

        # Add sector ETF adjusted close prices
        sector_etfs = ["XLF", "XLK", "XLV", "XLE", "XLY", "XLU", "XLI"]
        for etf in sector_etfs:
            data[f"{etf}_Adj_Close"] = data[f"{etf}_Adj Close"]

        # Add macroeconomic indicators
        print("Adding macroeconomic indicators...")
        data["USD_JPY"] = data["JPY=X_Adj Close"]
        data["VIX"] = data["^VIX_Adj Close"]
        data["Gold"] = data["GC=F_Adj Close"]
        data["Oil"] = data["CL=F_Adj Close"]

        # Add returns and target variable
        print("Calculating returns and target...")
        data["SPY_Monthly_Return"] = data["SPY_Adj Close"].pct_change()
        data["Target"] = (data["SPY_Adj Close"].shift(-1) > data["SPY_Adj Close"]).astype(int)

        # Clean up NaN values
        data = data.dropna()

        # Save to CSV
        os.makedirs("data", exist_ok=True)
        data.to_csv(output_file)
        print(f"Data saved to {output_file}.")

    return data


# Main block
if __name__ == "__main__":
    output_file = "data/spy_extended_features.csv"
    try:
        processed_data = prepare_stock_data(output_file)
        print(processed_data.head())
    except Exception as e:
        print(f"An error occurred: {e}")