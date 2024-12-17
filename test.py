import yfinance as yf
import pandas as pd

# Fetch data for TSLA, USD/JPY, and VIX with 1-month interval
tickers = ["TSLA", "JPY=X", "^VIX"]
data = yf.download(tickers, interval="1mo", period="max")['Adj Close']

# Rename columns for clarity
data.columns = ["TSLA", "USD_JPY", "VIX"]

# Drop rows with missing values
data = data.dropna()

# Reset index to include 'Date' as a column
data_reset = data.reset_index()

# Grouping by Year-Month (if needed)
data_reset['YearMonth'] = data_reset['Date'].dt.to_period('M')
grouped_df = data_reset.groupby('YearMonth').mean()

# Print the grouped data
print("Grouped DataFrame by Month (Average TSLA, USD/JPY, and VIX):")
print(grouped_df)




# Ticker data for Tesla
tesla = yf.Ticker("TSLA")

# Fetch Tesla's history with a valid period
history = tesla.history(period="max", interval="1m")
print("History:")
print(history)

# Alternative to `info` (limited data)
print("\nFast Info:")
print(tesla.fast_info)

# Earnings
try:
    earnings = tesla.incomestmt
    print(earnings)
except Exception as e:
    print(e)



# Recommendations
try:
    recommendations = tesla.recommendations
    print("\nRecommendations:")
    print(recommendations)
except Exception as e:
    print("\nError fetching recommendations:", e)