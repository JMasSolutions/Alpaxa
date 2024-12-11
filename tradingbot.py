from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime
from alpaca_trade_api_fixed import REST
from timedelta import Timedelta
import csv
import requests


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

# Define Start & End Date
start_date = datetime(2023, 12,15)
end_date = datetime(2023, 12,31)


# Defining ALPACA Credentials
ALPACA_CREDS = {
    "API_KEY":api_key,
    "API_SECRET":secret_key,
    "PAPER":True
}

# Strategy
class MLTRADER(Strategy):
    # Initializes trading algorithim
    def initialize(self, symbol:str="SPY", cash_at_risk:float = .5):
        self.symbol = symbol
        self.sleeptime = "24h"
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=endpoint,
                        key_id=api_key,
                        secret_key=secret_key,
                        )

    # Gets dates for get_news function
    def get_dates(self):
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return three_days_prior.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

    # Gets sentiment from news headlines
    def get_sentiment(self):
        start, end = self.get_dates()
        url = "https://data.alpaca.markets/v1beta1/news"
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }
        params = {
            "start": start,
            "end": end,
            "symbols": self.symbol,
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"Error fetching news: {response.status_code}, {response.text}")
            return []

        response_data = response.json()
        news_items = response_data.get("news", [])
        news_headlines = [item.get("headline", "No headline") for item in news_items]

        # Log fetched news headlines
        print(f"Fetched news for {self.symbol}: {news_headlines}")
        return news_headlines

    # Dynamically calculate potential position price
    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price) # This formula guides how much of our cash balance
                                                                # we use per trade. cash_at_risk of 0.5 means that for
                                                                # each trade we're using 50% of our remaining cash
                                                                # balance
        return cash, last_price, quantity

    # Runs after every new tick, every new piece of data
    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()

        # Check if cash balance is greater than last price
        if cash > last_price:
            if self.last_trade is None:
                news = self.get_news()
                print(news)
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type="bracket",
                    take_profit_price=last_price*1.20, # Take profit at 20%
                    stop_loss_price=last_price*.95 # Stop loss at 5%
                )
                self.submit_order(order)
                self.last_trade = "buy"



# Create broker
broker = Alpaca(ALPACA_CREDS)

# Instance of strategy
startegy = MLTRADER(name="mlstart", broker=broker,
                    parameters={"symbol":"SPY",
                                "cash_at_risk":0.5})

# Set up backtesting
startegy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={"symbol":"SPY",
                "cash_at_risk":0.5},
)
