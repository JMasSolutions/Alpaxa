import torch
import torch.nn as nn
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from datetime import datetime
from alpaca_trade_api_fixed import REST
import csv


# =====================
# LSTM MODEL DEFINITION
# =====================
class LSTMD(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
        """
        LSTM Model for binary classification.
        """
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
    next(csv_reader)  # Skip header row
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
        self.api = REST(base_url=endpoint, key_id=api_key, secret_key=secret_key)

        # LSTM Model Configuration
        input_size = 10  # Feature size
        hidden_size = 128
        num_layers = 3
        dropout = 0.3

        # Instantiate and load the LSTM model
        self.lstm_model = LSTMD(input_size, hidden_size, num_layers, dropout).to("cpu")
        self.lstm_model.load_state_dict(torch.load("models/76mod.pth", map_location="cpu"))
        self.lstm_model.eval()
        print("LSTM model loaded successfully!")

    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price)
        return cash, last_price, quantity

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()

        # Dummy input for testing, replace this with actual input features
        input_tensor = torch.randn(1, 10, 10)

        # Run LSTM model inference
        with torch.no_grad():
            output = self.lstm_model(input_tensor)
            prediction = torch.sigmoid(output).item()

        print(f"Model Prediction: {prediction:.4f}")

        # Trade decision
        if cash > last_price:
            if prediction > 0.6:  # Threshold for buy
                print("Positive prediction - BUY order.")
                order = self.create_order(self.symbol, quantity, "buy", type="market")
                self.submit_order(order)
                self.last_trade = "buy"
            elif prediction < 0.4:  # Threshold for sell
                print("Negative prediction - SELL order.")
                order = self.create_order(self.symbol, quantity, "sell", type="market")
                self.submit_order(order)
                self.last_trade = "sell"


# =====================
# BACKTESTING CONFIGURATION
# =====================
ALPACA_CREDS = {
    "API_KEY": api_key,
    "API_SECRET": secret_key,
    "PAPER": True,
}

start_date = datetime(2016, 1, 1)
end_date = datetime(2024, 10, 15)

broker = Alpaca(ALPACA_CREDS)

# Instantiate strategy
strategy = MLTRADER(name="ml_strategy", broker=broker, parameters={"symbol": "SPY", "cash_at_risk": 0.3})

# Backtest
strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={"symbol": "TSLA", "cash_at_risk": 0.3},
)