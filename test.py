# Corrected Strategy and Trader Setup
from lumibot.brokers import Alpaca
from lumibot.traders import Trader
from lumibot.strategies.strategy import Strategy
from datetime import datetime
from alpaca_trade_api_fixed import REST
from timedelta import Timedelta
import csv
import requests
from finbert_utils import estimate_sentiment

# Read API Credentials
file_path = "/Users/joanmascastella/Documents/ALPAXA/API_KEYS.csv"
with open(file_path, "r") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip header
    row = next(csv_reader)
    endpoint, api_key, secret_key = [value.strip() for value in row]

ALPACA_CREDS = {
    "API_KEY": api_key,
    "API_SECRET": secret_key,
    "PAPER": True,
}

class MLTRADER(Strategy):
    def initialize(self, symbol="NVDA", cash_at_risk=0.5):  # Default symbol updated to NVDA
        self.symbol = symbol
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=endpoint, key_id=api_key, secret_key=secret_key)
        self.last_trade = None
        print("Initialized strategy with the following settings:")
        print(f"Symbol: {self.symbol}, Cash at Risk: {self.cash_at_risk}")

    def get_dates(self):
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return three_days_prior.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

    def get_sentiment(self):
        start, end = self.get_dates()
        url = "https://data.alpaca.markets/v1beta1/news"
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }
        params = {"start": start, "end": end, "symbols": self.symbol}

        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Error fetching news: {response.status_code}, {response.text}")
            return 0.0, "neutral"

        news_items = response.json().get("news", [])
        headlines = [item.get("headline", "No headline") for item in news_items]

        if not headlines:
            print(f"No news for {self.symbol}. Defaulting to neutral sentiment.")
            return 0.0, "neutral"

        # Log the headlines for debugging
        print(f"News Headlines: {headlines}")
        probability, sentiment = estimate_sentiment(headlines)
        print(f"Sentiment: {sentiment}, Probability: {probability}")
        return probability, sentiment

    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price)
        print(f"Position Sizing: Cash: {cash}, Last Price: {last_price}, Quantity: {quantity}")
        return cash, last_price, quantity

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()

        if cash > last_price:
            if sentiment == "positive" and probability > 0.7:  # Adjusted threshold for testing
                if self.last_trade == "sell":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type="bracket",
                    take_profit_price=last_price * 1.2,
                    stop_loss_price=last_price * 0.95,
                )
                self.submit_order(order)
                self.last_trade = "buy"
                print("Placed BUY order.")

            elif sentiment == "negative" and probability > 0.7:
                if self.last_trade == "buy":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "sell",
                    type="bracket",
                    take_profit_price=last_price * 0.8,
                    stop_loss_price=last_price * 1.05,
                )
                self.submit_order(order)
                self.last_trade = "sell"
                print("Placed SELL order.")

# Initialize Broker and Trader
broker = Alpaca(ALPACA_CREDS)
strategy = MLTRADER(name="MLTrader", broker=broker, parameters={"symbol": "NVDA", "cash_at_risk": 0.5})
trader = Trader()
trader.add_strategy(strategy)

# Run the Trader (Live Trading)
trader.run_all()