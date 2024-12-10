import csv
from alpaca.trading.client import TradingClient

# Path to your CSV file
file_path = "/Users/joanmascastella/Documents/ALPAXA/API_KEYS.csv"

# Read the CSV file
with open(file_path, "r") as file:
    csv_reader = csv.reader(file)
    # Skip the header row
    next(csv_reader)
    # Read the first row (assuming only one key-value pair exists)
    row = next(csv_reader)
    endpoint, api_key, secret_key = [value.strip() for value in row]  # Strip whitespace

# Initialize Alpaca TradingClient
trading_client = TradingClient(api_key, secret_key, paper=True)

# Example: Print account details to verify it works
try:
    account = trading_client.get_account()
    print(account)
except Exception as e:
    print(f"Error: {e}")