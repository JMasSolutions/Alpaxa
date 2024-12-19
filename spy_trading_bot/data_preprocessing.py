import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from imblearn.over_sampling import SMOTE
import joblib
import os

# Define a consistent feature list
FEATURES = [
    "SPY_Adj_Close", "SPY_Open", "SPY_High", "SPY_Low", "SPY_Close", "SPY_Volume",
    "SMA_14", "EMA_14", "RSI", "BB_upper", "BB_middle", "BB_lower",
    "USD_JPY", "VIX", "Gold", "Oil", "Monthly_Return", "MACD", "ATR", "OBV",
    "Momentum", "Volatility", "Day_of_Week", "Month"
]

# Generate lag features
def add_lag_features(data, column, lags):
    for lag in lags:
        data[f"{column}_Lag_{lag}"] = data[column].shift(lag)
    return data

# Generate momentum
def add_momentum(data, column, window=10):
    data["Momentum"] = data[column].diff(window)
    return data

# Generate volatility
def add_volatility(data, column, window=14):
    data["Volatility"] = data[column].pct_change().rolling(window=window).std()
    return data

# Formatting the DataFrame to our model's liking
def clean_DF(path):
    data = pd.read_csv(path, index_col=0, parse_dates=True)
    data.columns = data.columns.str.replace(' ', '_')  # Standardize column names

    # Ensure the required columns exist
    required_columns = ["SPY_Adj_Close"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Generate lag features
    data = add_lag_features(data, "SPY_Adj_Close", [1, 2, 3, 5])

    # Generate momentum
    data = add_momentum(data, "SPY_Adj_Close")

    # Generate volatility
    data = add_volatility(data, "SPY_Adj_Close")

    # Add date features
    data["Day_of_Week"] = data.index.dayofweek
    data["Month"] = data.index.month

    # Validate columns
    missing_features = [col for col in FEATURES + ["Target"] if col not in data.columns]
    if missing_features:
        raise ValueError(f"Missing required columns: {missing_features}")

    # Filter relevant columns
    cleaned_data = data[FEATURES + ["Target"]].dropna()
    return cleaned_data

# Scale & preprocess data
def scale_processD(df, scaler, balance_data=False):
    X = df[FEATURES]
    y = df["Target"]

    # Train-test split based on time
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Fit scaler on training data
    scaler.fit(X_train)

    # Scale data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Optionally balance the training data using SMOTE
    if balance_data:
        smote = SMOTE(random_state=42)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

    # Convert scaled data back to DataFrame for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=FEATURES)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=FEATURES, index=X_test.index)

    return X_train_scaled, X_test_scaled, y_train, y_test

# PyTorch Dataset class for our data
class StockDataSet(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Prepare and return datasets for training
def prepare_stock_data(file_path, balance_data=False):
    df = clean_DF(file_path)
    print("Cleaned DataFrame:")
    print(df.head())

    scaler = MinMaxScaler(feature_range=(-1, 1))

    X_train_scaled, X_test_scaled, y_train, y_test = scale_processD(df, scaler, balance_data=balance_data)

    # Save processed data
    os.makedirs('data', exist_ok=True)
    X_train_scaled.to_csv("data/scaled_features_train.csv", index=False)
    X_test_scaled.to_csv("data/scaled_features_test.csv", index=False)
    y_train.to_csv("data/target_train.csv", index=False)
    y_test.to_csv("data/target_test.csv", index=False)

    # Save the scaler
    os.makedirs('model', exist_ok=True)
    joblib.dump(scaler, 'model/scaler.pkl')
    print("Scaler saved as 'model/scaler.pkl'")

    # Create PyTorch Datasets
    train_dataset = StockDataSet(X_train_scaled, y_train)
    test_dataset = StockDataSet(X_test_scaled, y_test)

    print(f"\nTraining Dataset Length: {len(train_dataset)}")
    print(f"Testing Dataset Length: {len(test_dataset)}")

    return train_dataset, test_dataset

# Main block
if __name__ == "__main__":
    file_path = "data/spy_daily_data.csv"
    train_dataset, test_dataset = prepare_stock_data(file_path, balance_data=True)
    print("\nData preparation complete. Datasets are ready for training.")