import torch
import torch.nn as nn
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from datetime import datetime
import pandas as pd
import csv
import requests
import yfinance as yf
from helpful_functions.finbert_utils import estimate_sentiment

# =====================
# HELPER FUNCTIONS
# =====================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, short=12, long=26, signal=9):
    ema_short = series.ewm(span=short, adjust=False).mean()
    ema_long = series.ewm(span=long, adjust=False).mean()
    macd_line = ema_short - ema_long
    return macd_line

def fetch_usd_jpy():
    data = yf.download("JPY=X", period="1d", interval="1d")
    if data.empty:
        print("No data fetched for USD/JPY, returning default 145.0")
        return 145.0
    return data['Close'].iloc[-1].item()

def fetch_vix():
    data = yf.download("^VIX", period="1d", interval="1d")
    if data.empty:
        print("No data fetched for VIX, returning default 20.0")
        return 20.0
    return data['Close'].iloc[-1].item()

def calculate_monthly_return(series):
    if len(series) < 20:
        return pd.Series([0]*len(series), index=series.index)
    monthly_return = (series - series.shift(20)) / (series.shift(20) + 1e-8)
    return monthly_return.fillna(0)

def calculate_price_change(series):
    return series.pct_change().fillna(0)


# =====================
# LSTM MODEL DEFINITION
# =====================
class LSTMD(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the last time step
        return out


# =====================
# LOAD ALPACA CREDENTIALS
# =====================
file_path = "/Users/joanmascastella/Documents/ALPAXA/API_KEYS.csv"

with open(file_path, "r") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    row = next(csv_reader)
    endpoint, api_key, secret_key = [value.strip() for value in row]


# =====================
# TRADING STRATEGY
# =====================
class MLTRADER(Strategy):
    def initialize(self, symbol: str = "TSLA", cash_at_risk: float = 0.3):
        self.symbol = symbol
        self.sleeptime = "24h"
        self.last_trade = None
        self.cash_at_risk = cash_at_risk

        input_size = 10
        hidden_size = 128
        num_layers = 3
        dropout = 0.3
        self.lstm_model = LSTMD(input_size, hidden_size, num_layers, dropout).to("cpu")
        self.lstm_model.load_state_dict(torch.load("models/76mod.pth", map_location="cpu"))
        self.lstm_model.eval()
        print("LSTM model loaded successfully!")

    def get_sentiment(self):
        today = self.get_datetime()
        three_days_prior = today - pd.Timedelta(days=3)
        start = three_days_prior.strftime("%Y-%m-%d")
        end = today.strftime("%Y-%m-%d")

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

        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code != 200:
                print(f"Error fetching news: {response.status_code}, {response.text}")
                return 0.0, "neutral"

            response_data = response.json()
            news_items = response_data.get("news", [])
            news_headlines = [item.get("headline", "No headline") for item in news_items]

            # Print the news headlines for debugging
            print("\nFetched News Headlines:")
            for idx, headline in enumerate(news_headlines, 1):
                print(f"{idx}. {headline}")

            if not news_headlines:
                print(f"No news fetched for {self.symbol}. Defaulting to neutral sentiment.")
                return 0.0, "neutral"

            probability, sentiment = estimate_sentiment(news_headlines)
            return probability, sentiment

        except Exception as e:
            print(f"Exception fetching sentiment: {e}")
            return 0.0, "neutral"  # Default to neutral sentiment on error

    def preprocess_realtime_data(self, data):
        scaled_features = (data - data.mean()) / (data.std() + 1e-8)
        input_tensor = torch.tensor(scaled_features.values, dtype=torch.float32).unsqueeze(0)
        return input_tensor

    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price)
        return cash, last_price, quantity

    def on_trading_iteration(self):
        # Fetch 30 days of daily data from Yahoo (backtester)
        bars = self.get_historical_prices(self.symbol, length=30, timestep="day")
        bars = bars.df  # Convert Bars to DataFrame

        # Ensure we have data
        if len(bars) < 1:
            print("No data fetched, cannot index last row.")
            return

        # Get last close price from historical data
        last_close = bars["close"].iloc[-1]

        # Fetch sentiment
        probability, sentiment = self.get_sentiment()
        print(f"Sentiment Raw Output: Probability={probability}, Sentiment={sentiment}")

        # Run LSTM model inference
        bars["Adj Close"] = bars["close"]
        bars["SMA_14"] = bars["close"].rolling(window=14).mean()
        bars["EMA_14"] = bars["close"].ewm(span=14, adjust=False).mean()
        bars["RSI"] = calculate_rsi(bars["close"], period=14)
        bars["MACD"] = calculate_macd(bars["close"])
        bars["USD_JPY"] = fetch_usd_jpy()
        bars["VIX"] = fetch_vix()
        bars["Monthly_Return"] = calculate_monthly_return(bars["close"])
        bars["Price_Change"] = calculate_price_change(bars["close"])
        bars["Sentiment_Score"] = probability  # Use probability from sentiment

        # Select the last 10 rows and 10 features
        bars = bars.tail(10).fillna(0)
        bars = bars[["Adj Close", "SMA_14", "EMA_14", "RSI", "MACD", "USD_JPY", "VIX", "Monthly_Return", "Price_Change",
                     "Sentiment_Score"]]

        input_tensor = self.preprocess_realtime_data(bars)
        print(f"Input Tensor Shape: {input_tensor.shape}")  # Debugging
        if input_tensor.shape != (1, 10, 10):
            print(f"Invalid input shape {input_tensor.shape}")
            return

        with torch.no_grad():
            output = self.lstm_model(input_tensor)
            prediction = torch.sigmoid(output).item()

        print(f"Model Prediction: {prediction:.4f}")

        # Position sizing
        cash, last_price, quantity = self.position_sizing()

        # Define thresholds
        upper_buy_threshold = 0.521
        lower_sell_threshold = 0.461


        if cash > last_price:
            # If model predicts BUY
            if prediction > upper_buy_threshold:
                print("ML Model Suggests BUY.")
                if sentiment == "positive" and probability > 0.9:
                    print("Strong BUY signal: Positive sentiment confirms model prediction.")
                    if self.last_trade == "sell":
                        self.sell_all()
                    order = self.create_order(
                        self.symbol,
                        quantity,
                        "buy",
                        type="market",
                        take_profit_price=last_price * 1.30,
                        stop_loss_price=last_price * 0.85
                    )
                    self.submit_order(order)
                    self.last_trade = "buy"
                else:
                    print("Sentiment does not confirm BUY signal - NO ACTION.")

            # If model predicts SELL
            elif prediction < lower_sell_threshold:
                print("ML Model Suggests SELL.")
                if sentiment == "negative" and probability > 0.9:
                    print("Strong SELL signal: Negative sentiment confirms model prediction.")
                    if self.last_trade == "buy":
                        self.sell_all()
                    order = self.create_order(
                        self.symbol,
                        quantity,
                        "sell",
                        type="market",
                        take_profit_price=last_price * 0.85,
                        stop_loss_price=last_price * 1.05
                    )
                    self.submit_order(order)
                    self.last_trade = "sell"
                else:
                    print("Sentiment does not confirm SELL signal - NO ACTION.")

            # If model prediction is neutral
            else:
                print("ML Model prediction is neutral - NO ACTION.")


# =====================
# BACKTESTING CONFIGURATION
# =====================
with open(file_path, "r") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    row = next(csv_reader)
    endpoint, api_key, secret_key = [value.strip() for value in row]

ALPACA_CREDS = {
    "API_KEY": api_key,
    "API_SECRET": secret_key,
    "PAPER": True,
}

start_date = datetime(2016, 1, 1)
end_date = datetime(2024, 10, 15)

broker = Alpaca(ALPACA_CREDS)

# Instantiate strategy
strategy = MLTRADER(name="ml_strategy", broker=broker, parameters={"symbol": "TSLA", "cash_at_risk": 0.3})

# Backtest
strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={"symbol": "TSLA", "cash_at_risk": 0.3},
)