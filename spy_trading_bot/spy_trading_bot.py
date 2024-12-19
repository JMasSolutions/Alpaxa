import os
import csv
import time
import joblib
import pandas as pd
import yfinance as yf
from datetime import datetime
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
import logging
import pandas_ta as ta
import traceback

# =====================
# LOGGING CONFIGURATION
# =====================
def setup_logging():
    """
    Configures the logging settings to log to both a file and the console.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Changed from INFO to DEBUG

    # Formatter for file logs
    file_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    # File Handler
    file_handler = logging.FileHandler('trading_bot.log')
    file_handler.setLevel(logging.DEBUG)  # Capture DEBUG logs in file
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Formatter for console logs
    console_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Keep console at INFO to reduce verbosity
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Prevent duplicate logs if the logger already has handlers
    if logger.hasHandlers():
        logger.handlers = []
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

setup_logging()
logger = logging.getLogger(__name__)

# =====================
# HELPER FUNCTIONS
# =====================

def calculate_rsi(series, period=14):
    """
    Calculates the Relative Strength Index (RSI) for a given series.
    """
    rsi = ta.rsi(series, length=period)
    if rsi is not None and not rsi.empty:
        return rsi
    else:
        logger.error("RSI calculation returned None or empty Series.")
        return pd.Series([0] * len(series), index=series.index)

def calculate_bbands(series, length=20):
    """
    Calculates Bollinger Bands for a given series.
    """
    bb = ta.bbands(series, length=length)
    if bb is not None and not bb.empty:
        return bb['BBL_20_2.0'], bb['BBM_20_2.0'], bb['BBU_20_2.0']
    else:
        logger.error("Bollinger Bands calculation returned None or empty DataFrame.")
        # Return Series of zeros
        return pd.Series([0] * len(series), index=series.index), \
               pd.Series([0] * len(series), index=series.index), \
               pd.Series([0] * len(series), index=series.index)

def calculate_stochastic(high, low, close):
    """
    Calculates the Stochastic Oscillator (%K and %D) for given high, low, and close series.
    """
    stoch = ta.stoch(high, low, close)
    if stoch is not None and not stoch.empty:
        return stoch['STOCHk_14_3_3'], stoch['STOCHd_14_3_3']
    else:
        logger.error("Stochastic Oscillator calculation returned None or empty DataFrame.")
        # Return Series of zeros
        return pd.Series([0] * len(close), index=close.index), pd.Series([0] * len(close), index=close.index)

def calculate_sma(series, period=14):
    """
    Calculates the Simple Moving Average (SMA) for a given series.
    """
    sma = ta.sma(series, length=period)
    if sma is not None and not sma.empty:
        return sma
    else:
        logger.error("SMA calculation returned None or empty Series.")
        return pd.Series([0] * len(series), index=series.index)

def calculate_ema(series, period=14):
    """
    Calculates the Exponential Moving Average (EMA) for a given series.
    """
    ema = ta.ema(series, length=period)
    if ema is not None and not ema.empty:
        return ema
    else:
        logger.error("EMA calculation returned None or empty Series.")
        return pd.Series([0] * len(series), index=series.index)

def load_model(model_path='model/lightgbm_model.pkl'):
    """
    Loads the pre-trained LightGBM model from the specified path.
    """
    if not os.path.exists(model_path):
        logger.critical(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = joblib.load(model_path)
    logger.info("LightGBM model loaded successfully!")
    return model

def load_scaler(scaler_path='model/scaler.pkl'):
    """
    Loads the scaler used during training.
    """
    if not os.path.exists(scaler_path):
        logger.critical(f"Scaler file not found at {scaler_path}")
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    scaler = joblib.load(scaler_path)
    if hasattr(scaler, 'n_features_in_'):
        logger.info(f"Scaler was trained on {scaler.n_features_in_} features.")
    else:
        logger.warning("Scaler does not have 'n_features_in_' attribute.")
    logger.info("Scaler loaded successfully!")
    return scaler

def load_alpaca_credentials(file_path):
    """
    Loads Alpaca API credentials from a CSV file.
    The CSV should have the following columns: endpoint, api_key, secret_key
    """
    if not os.path.exists(file_path):
        logger.critical(f"Alpaca credentials file not found at {file_path}")
        raise FileNotFoundError(f"Alpaca credentials file not found at {file_path}")

    with open(file_path, "r") as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader, None)  # Skip header if present
        row = next(csv_reader, None)
        if not row or len(row) < 3:
            logger.critical(
                "Alpaca credentials file must contain at least three columns: endpoint, api_key, secret_key")
            raise ValueError(
                "Alpaca credentials file must contain at least three columns: endpoint, api_key, secret_key")
        endpoint, api_key, secret_key = [value.strip() for value in row[:3]]

    logger.info("Alpaca credentials loaded successfully.")
    return {
        "API_KEY": api_key,
        "API_SECRET": secret_key,
        "PAPER": True,  # Set to False for live trading
        "ENDPOINT": endpoint
    }

def prepare_latest_data(symbol='SPY', max_retries=5, delay=5):
    """
    Fetches the latest stock data with technical indicators for the specified symbol.
    Implements retries in case of transient failures.
    Returns a DataFrame with only the specified 12 features and 'Target'.
    """
    logger.info(f"Fetching latest data for {symbol}...")
    tickers = [symbol, "JPY=X", "^VIX", "GC=F", "CL=F"]
    data = pd.DataFrame()

    for ticker in tickers:
        for attempt in range(1, max_retries + 1):
            try:
                ticker_data = yf.download(ticker, interval="1d", period="1y")
                if ticker_data.empty or ticker_data.isnull().all().all():
                    raise ValueError(f"Fetched data for {ticker} is empty or contains only NaN values.")
                data[ticker] = ticker_data['Adj Close']
                logger.info(f"Data fetched successfully for {ticker}.")
                break
            except Exception as e:
                logger.error(f"Attempt {attempt} failed for {ticker}: {e}")
                if attempt < max_retries:
                    logger.info(f"Retrying {ticker} in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.critical(f"Max retries reached for {ticker}. Skipping this ticker.")
                    raise e

    # Calculate technical indicators for SPY
    logger.info("Calculating technical indicators...")
    try:
        df_features = pd.DataFrame()
        df_features['Adj Close'] = data[symbol]
        df_features['SMA_14'] = calculate_sma(data[symbol], period=14)
        df_features['EMA_14'] = calculate_ema(data[symbol], period=14)
        df_features['RSI'] = calculate_rsi(data[symbol], period=14)
        bb_lower, bb_middle, bb_upper = calculate_bbands(data[symbol], length=20)
        df_features['BB_lower'] = bb_lower
        df_features['BB_middle'] = bb_middle
        df_features['BB_upper'] = bb_upper
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        logger.error(traceback.format_exc())
        raise e

    # Add market indicators
    logger.info("Adding market indicators...")
    try:
        df_features['USD_JPY'] = data['JPY=X']
        df_features['VIX'] = data['^VIX']
        df_features['Gold'] = data['GC=F']
        df_features['Oil'] = data['CL=F']
    except Exception as e:
        logger.error(f"Error adding market indicators: {e}")
        logger.error(traceback.format_exc())
        raise e

    # Calculate returns
    logger.info("Calculating returns...")
    try:
        df_features['Monthly_Return'] = df_features['Adj Close'].pct_change()
    except Exception as e:
        logger.error(f"Error calculating returns: {e}")
        logger.error(traceback.format_exc())
        raise e

    # Create target variable
    logger.info("Creating target variable...")
    try:
        df_features['Target'] = (df_features['Adj Close'].shift(-1) > df_features['Adj Close']).astype(int)
    except Exception as e:
        logger.error(f"Error creating target variable: {e}")
        logger.error(traceback.format_exc())
        raise e

    # Drop rows with NaN values
    df_features = df_features.dropna()

    # Select relevant features (12 features as specified)
    features = [
        "Adj Close", "SMA_14", "EMA_14", "RSI",
        "BB_upper", "BB_middle", "BB_lower",
        "USD_JPY", "VIX",
        "Gold", "Oil", "Monthly_Return"
    ]

    try:
        df_features = df_features[features + ['Target']]  # 12 features + 'Target' =13 columns
        logger.info(f"Selected features: {df_features.columns.tolist()}")
    except Exception as e:
        logger.error(f"Error selecting features: {e}")
        logger.error(traceback.format_exc())
        raise e

    logger.info(f"Data shape after feature selection: {df_features.shape}")
    if df_features.shape[1] != 13:  # 12 features + 'Target'
        logger.error(f"Feature count mismatch: Expected 13, got {df_features.shape[1]}")
        raise ValueError(f"Feature count mismatch: Expected 13, got {df_features.shape[1]}")

    logger.debug(f"Prepared Data:\n{df_features.tail()}")  # Add this line

    logger.info("Data preparation successful.")
    return df_features

# =====================
# TRADING STRATEGY CLASS
# =====================

class LightGBMTrader(Strategy):
    def initialize(self, symbol: str = "SPY", cash_at_risk: float = 0.3):
        """
        Initializes the trading strategy.
        """
        self.symbol = symbol
        self.sleeptime = "24h"  # Trading interval
        self.last_trade = None
        self.cash_at_risk = cash_at_risk

        # Load the pre-trained LightGBM model
        self.model = load_model('model/lightgbm_model.pkl')
        print(f"Model Classes: {self.model.classes_}")  # Add this line
        # Load scaler used during training
        self.scaler = load_scaler('model/scaler.pkl')

        logger.info("LightGBMTrader initialized successfully!")

    def preprocess_realtime_data(self, data: pd.DataFrame):
        """
        Preprocess the data to match the training features.
        Apply scaling using the loaded scaler.
        """
        try:
            logger.info(f"Features before scaling: {data.columns.tolist()}")
            expected_features = 12  # Number of features used during training
            if data.shape[1] != expected_features:
                logger.error(f"Expected {expected_features} features, but got {data.shape[1]}")
                raise ValueError(f"Expected {expected_features} features, but got {data.shape[1]}")
            scaled_features = self.scaler.transform(data)
            logger.debug(f"Scaled Features:\n{scaled_features}")
            return scaled_features
        except Exception as e:
            logger.error(f"Error during data scaling: {e}")
            logger.error(traceback.format_exc())
            raise e

    def position_sizing(self):
        """
        Determines the position size based on available cash and risk parameters.
        """
        try:
            cash = self.get_cash()
            last_price = self.get_last_price(self.symbol)
            quantity = round(cash * self.cash_at_risk / last_price)
            logger.debug(f"Position Sizing - Cash: {cash}, Last Price: {last_price}, Quantity: {quantity}")
            return cash, last_price, quantity
        except Exception as e:
            logger.error(f"Error during position sizing: {e}")
            logger.error(traceback.format_exc())
            raise e

    def on_trading_iteration(self):
        """
        Main trading logic executed at each trading interval.
        """
        logger.debug("on_trading_iteration called.")
        logger.info(f"Starting trading iteration for {self.symbol}...")

        try:
            # Step 1: Fetch and prepare the latest data
            data = prepare_latest_data(self.symbol)
            logger.debug(f"Data fetched for trading iteration:\n{data.tail()}")

            # Step 2: Get the most recent row for prediction
            latest_data = data.tail(1)
            logger.debug(f"Latest data for prediction:\n{latest_data}")
            if latest_data.empty:
                logger.warning("No data available for prediction.")
                return

            # Exclude the latest row if it contains NaN values
            if latest_data.isnull().values.any():
                logger.info("Excluding the latest incomplete row with NaN values.")
                latest_data = latest_data.iloc[:-1]
                if latest_data.empty:
                    logger.warning("No valid data available after excluding incomplete rows.")
                    return

            # Step 3: Preprocess data (scale features)
            features = [
                "Adj Close", "SMA_14", "EMA_14", "RSI",
                "BB_upper", "BB_middle", "BB_lower",
                "USD_JPY", "VIX",
                "Gold", "Oil", "Monthly_Return"
            ]
            latest_features = latest_data[features]

            # Handle any missing values if necessary
            latest_features = latest_features.fillna(0)
            logger.debug(f"Features after handling missing values:\n{latest_features}")

            # Log the features being used for prediction
            logger.debug(f"Latest Features:\n{latest_features}")

            # Scale the features
            X = self.preprocess_realtime_data(latest_features)
            logger.debug(f"Scaled Features:\n{X}")

            # Step 4: Make prediction using the LightGBM model
            prediction = self.model.predict(X)[0]
            prediction_proba_buy = self.model.predict_proba(X)[0][1]  # Probability of class 1 (Buy)
            prediction_proba_sell = self.model.predict_proba(X)[0][0]  # Probability of class 0 (Sell)

            # Log the prediction and probabilities
            logger.info(
                f"Model Prediction: {'Buy' if prediction == 1 else 'Sell'} with Buy probability {prediction_proba_buy:.4f} and Sell probability {prediction_proba_sell:.4f}")

            # Step 5: Determine position sizing
            cash, last_price, quantity = self.position_sizing()
            logger.info(
                f"Available Cash: ${cash:.2f}, Last Price: ${last_price:.2f}, Quantity to Trade: {quantity}"
            )

            # Step 6: Define thresholds
            buy_threshold = 0.6  # Probability above which to buy
            sell_threshold = 0.6  # Probability above which to sell

            # Log the thresholds and check against prediction probabilities
            logger.debug(f"Buy Threshold: {buy_threshold}, Sell Threshold: {sell_threshold}")
            logger.debug(
                f"Prediction Buy Probability: {prediction_proba_buy:.4f}, Prediction Sell Probability: {prediction_proba_sell:.4f}")

            # Step 7: Trading logic
            if prediction == 1 and prediction_proba_buy > buy_threshold:
                print("Model suggests BUY." + str(prediction_proba_buy))
                if self.last_trade == "sell":
                    self.sell_all()
                    logger.info("Sold all previous short positions.")
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type="market",
                    take_profit_price=last_price * 1.05,  # 5% take profit
                    stop_loss_price=last_price * 0.95  # 5% stop loss
                )
                self.submit_order(order)
                logger.info(
                    f"BUY order submitted for {quantity} shares of {self.symbol} at ${last_price:.2f}"
                )
                self.last_trade = "buy"

            elif prediction == 0 and prediction_proba_sell > sell_threshold:
                print("Model suggests SELL." + str(prediction_proba_sell))
                if self.last_trade == "buy":
                    self.sell_all()
                    logger.info("Sold all previous long positions.")
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "sell",
                    type="market",
                    take_profit_price=last_price * 0.95,  # 5% take profit
                    stop_loss_price=last_price * 1.05  # 5% stop loss
                )
                self.submit_order(order)
                logger.info(
                    f"SELL order submitted for {quantity} shares of {self.symbol} at ${last_price:.2f}"
                )
                self.last_trade = "sell"

            else:
                logger.info("Model suggests HOLD. No action taken.")

        except Exception as e:
            logger.error(f"Exception during trading iteration: {e}")
            logger.error(traceback.format_exc())

# =====================
# MAIN EXECUTION BLOCK
# =====================

def main():
    # Path to Alpaca credentials CSV
    alpaca_credentials_path = "/Users/joanmascastella/Documents/ALPAXA/API_KEYS.csv"

    # Load Alpaca credentials
    try:
        alpaca_creds = load_alpaca_credentials(alpaca_credentials_path)
    except Exception as e:
        logger.critical(f"Failed to load Alpaca credentials: {e}")
        return

    # Initialize Alpaca broker
    try:
        broker = Alpaca(alpaca_creds)
    except Exception as e:
        logger.critical(f"Failed to initialize Alpaca broker: {e}")
        return

    # Instantiate the trading strategy
    try:
        strategy = LightGBMTrader(
            name="lightgbm_strategy",
            broker=broker,
            parameters={"symbol": "SPY", "cash_at_risk": 0.3}
        )
    except Exception as e:
        logger.critical(f"Failed to instantiate trading strategy: {e}")
        return

    # Define backtesting parameters
    start_date = datetime(2023, 1, 1)  # Temporarily narrowed for testing
    end_date = datetime(2023, 12, 31)  # Temporarily narrowed for testing

    # Run backtest
    logger.info("Starting backtest...")
    try:
        strategy.backtest(
            YahooDataBacktesting,
            start_date,
            end_date,
            parameters={"symbol": "SPY", "cash_at_risk": 0.3},
        )
        logger.info("Backtest completed successfully.")
    except Exception as e:
        logger.error(f"Error during backtesting: {e}")
        logger.error(traceback.format_exc())

    # For live trading, uncomment the following lines
    # logger.info("Starting live trading...")
    # try:
    #     strategy.run()
    #     logger.info("Live trading started successfully.")
    # except Exception as e:
    #     logger.error(f"Error during live trading: {e}")
    #     logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()