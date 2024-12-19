import yfinance as yf

def yfinance_download():
    tickers = ["SPY", "JPY=X", "^VIX", "GC=F", "CL=F"]
    try:
        # Use a valid period like '2mo'
        data = yf.download(tickers, interval="1d", period="1mo")
        if data.empty:
            print("Fetched data is empty.")
        else:
            print("Data fetched successfully.")
            print(data.tail())
    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    yfinance_download()