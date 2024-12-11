from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime
import csv



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



# Defining ALPACA Credentials
ALPACA_CREDS = {
    "API_KEY":api_key,
    "API_SECRET":secret_key,
    "PAPER":True
}

# Strategy
class MLTRADER(Strategy):
    # Initializes trading algorithim
    def initialize(self):
        pass

    # Runs after every new tick, every new piece of data
    def on_trading_iteration(self):
        pass






# Create broker
broker = Alpaca(ALPACA_CREDS)
