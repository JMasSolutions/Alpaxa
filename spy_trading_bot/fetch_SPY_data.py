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

# Function to add lag features
def add_lag_features(data, column, lags):
    """
    Adds lagged features for a specified column.
    """
    for lag in lags:
        data[f"{column}_Lag_{lag}"] = data[column].shift(lag)
    return data

# Function to prepare the dataset
def prepare_stock_data(output_file):
    if os.path.exists(output_file):
        print(f"Loading data from {output_file}...")
        data = pd.read_csv(output_file, index_col=0, parse_dates=True)
    else:
        print("Fetching data...")
        tickers = ["SPY", "JPY=X", "^VIX", "GC=F", "CL=F", "XLF", "XLK", "XLV", "XLE", "XLY"]
        data = yf.download(tickers, interval="1d", period="max", group_by="ticker")

        # Flatten MultiIndex DataFrame
        data = flatten_columns(data)
        print(f"Flattened columns: {data.columns.tolist()}")  # Debug: Inspect columns

        # Technical Indicators
        print("Calculating technical indicators...")
        data['SMA_14'] = data['SPY_Adj Close'].rolling(window=14).mean()
        data['EMA_14'] = ta.ema(data['SPY_Adj Close'], length=14)
        data['RSI'] = ta.rsi(data['SPY_Adj Close'], length=14)
        bb = ta.bbands(data['SPY_Adj Close'], length=20)
        if bb is not None:
            data['BB_upper'] = bb['BBU_20_2.0']
            data['BB_middle'] = bb['BBM_20_2.0']
            data['BB_lower'] = bb['BBL_20_2.0']
        else:
            print("Bollinger Bands calculation failed.")
            data['BB_upper'], data['BB_middle'], data['BB_lower'] = 0, 0, 0

        # Additional Indicators
        print("Adding additional indicators...")
        macd = ta.macd(data['SPY_Adj Close'], fast=12, slow=26, signal=9)
        if macd is not None:
            data['MACD'] = macd['MACDh_12_26_9']
        else:
            print("MACD calculation failed.")
            data['MACD'] = 0
        data['ATR'] = ta.atr(data['SPY_High'], data['SPY_Low'], data['SPY_Adj Close'], length=14)
        data['OBV'] = ta.obv(data['SPY_Adj Close'], data['SPY_Volume'])

        # Add Lag Features
        print("Adding lagged features...")
        data = add_lag_features(data, "SPY_Adj Close", [1, 2, 3, 5])

        # Market Indicators
        print("Adding market indicators...")
        data['USD_JPY'] = data['JPY=X_Adj Close']
        data['VIX'] = data['^VIX_Adj Close']
        data['Gold'] = data['GC=F_Adj Close']
        data['Oil'] = data['CL=F_Adj Close']

        # Sector ETF Performance
        print("Adding sector ETF performance...")
        sector_etfs = ["XLF", "XLK", "XLV", "XLE", "XLY"]
        for etf in sector_etfs:
            column_name = f"{etf}_Adj_Close"
            data[column_name] = data[f"{etf}_Adj Close"]

        # Returns and Target
        print("Calculating returns and target...")
        data['Monthly_Return'] = data['SPY_Adj Close'].pct_change()
        data['Target'] = (data['SPY_Adj Close'].shift(-1) > data['SPY_Adj Close']).astype(int)

        # Date-based Features
        print("Adding date-based features...")
        data['Day_of_Week'] = data.index.dayofweek  # 0 = Monday, 6 = Sunday
        data['Month'] = data.index.month

        # Clean up NaN values
        data = data.dropna()

        # Save
        data.to_csv(output_file)
        print(f"Data saved to {output_file}.")

    return data

# Main block
if __name__ == "__main__":
    output_file = "data/spy_daily_data.csv"
    try:
        processed_data = prepare_stock_data(output_file)
        print(processed_data.head())
    except Exception as e:
        print(f"An error occurred: {e}")