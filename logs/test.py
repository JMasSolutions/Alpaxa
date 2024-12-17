import yfinance as yf

# Ticker data for Tesla
tesla = yf.Ticker("TSLA")

# Fetch Tesla's history with a valid period
history = tesla.history(period="max")
print("History:")
print(history)

# Alternative to `info` (limited data)
print("\nFast Info:")
print(tesla.fast_info)

# Recommendations
try:
    recommendations = tesla.recommendations
    print("\nRecommendations:")
    print(recommendations)
except Exception as e:
    print("\nError fetching recommendations:", e)