import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from imblearn.over_sampling import SMOTE  # For balancing data
import joblib  # For saving and loading scaler
import os  # To handle directory creation

# Define a consistent feature list
FEATURES = [
    "Adj Close", "SMA_14", "EMA_14", "RSI", "BB_upper", "BB_middle", "BB_lower",
    "USD_JPY", "VIX", "Gold", "Oil", "Monthly_Return", "MACD", "ATR", "OBV",
    "Adj_Close_Lag_1", "Adj_Close_Lag_2", "Adj_Close_Lag_3", "Adj_Close_Lag_5",
    "Momentum", "Volatility", "Day_of_Week", "Month"
]


# Formatting the DataFrame to our model's liking
def clean_DF(path):
    """
    Cleans the raw data and keeps only relevant columns.
    Removes rows with NaN values and validates required columns.
    """
    data = pd.read_csv(path, index_col=0, parse_dates=True)

    # Validate the presence of required columns
    missing_features = [col for col in FEATURES + ["Target"] if col not in data.columns]
    if missing_features:
        raise ValueError(f"Missing required columns: {missing_features}")

    # Filter the columns to keep only what matters
    cleaned_data = data[FEATURES + ["Target"]].dropna()
    return cleaned_data


# Scale & preprocess data
def scale_processD(df, scaler, balance_data=False):
    """
    Fits the scaler on training data and transforms both training and testing data.
    Optionally balances the dataset using SMOTE.
    """
    X = df[FEATURES]
    y = df["Target"]

    # Train-test split based on time (no shuffling)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Fit scaler on training data
    scaler.fit(X_train)

    # Transform both training and testing data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Optionally balance the training data
    if balance_data:
        smote = SMOTE(random_state=42)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

    # Convert training data into a new DataFrame (no index for SMOTE)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=FEATURES)

    # Convert testing data into a DataFrame (keep the original index for testing)
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
    """
    Cleans, scales, and creates train/test datasets.
    Optionally balances the training dataset using SMOTE.
    """
    # Step 1: Clean the data
    df = clean_DF(file_path)
    print("Cleaned DataFrame:")
    print(df.head())

    # Step 2: Initialize scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # Step 3: Scale the features and split
    X_train_scaled, X_test_scaled, y_train, y_test = scale_processD(df, scaler, balance_data=balance_data)

    print(f"\nScaled Training Features Shape: {X_train_scaled.shape}")
    print(f"Scaled Testing Features Shape: {X_test_scaled.shape}")
    print(f"Training Target Shape: {y_train.shape}")
    print(f"Testing Target Shape: {y_test.shape}")
    print("\nFirst 5 Scaled Training Features:")
    print(X_train_scaled.head())

    # Save scaled features and targets
    os.makedirs('data', exist_ok=True)
    X_train_scaled.to_csv("data/scaled_features_train.csv", index=False)
    X_test_scaled.to_csv("data/scaled_features_test.csv", index=False)
    y_train.to_csv("data/target_train.csv", index=False)
    y_test.to_csv("data/target_test.csv", index=False)

    # Step 4: Save the scaler for future use
    os.makedirs('model', exist_ok=True)  # Ensure the 'model' directory exists
    joblib.dump(scaler, 'model/scaler.pkl')
    print("Scaler saved as 'model/scaler.pkl'")

    # Step 5: Create PyTorch Datasets
    train_dataset = StockDataSet(X_train_scaled, y_train)
    test_dataset = StockDataSet(X_test_scaled, y_test)

    print(f"\nTraining Dataset Length: {len(train_dataset)}")
    print(f"Testing Dataset Length: {len(test_dataset)}")

    return train_dataset, test_dataset


# Main block
if __name__ == "__main__":
    file_path = "data/spy_daily_data.csv"

    # Prepare datasets with optional balancing
    train_dataset, test_dataset = prepare_stock_data(file_path, balance_data=True)

    print("\nData preparation complete. Datasets are ready for training.")