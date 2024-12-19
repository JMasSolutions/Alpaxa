import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from imblearn.over_sampling import SMOTE
import joblib
import os

# Dynamically derive features from the dataset after loading
def get_features(df):
    return [col for col in df.columns if col not in ["Target"]]

# Formatting the DataFrame
def clean_DF(path):
    """
    Cleans the raw data and keeps only relevant columns.
    Removes rows with NaN values and validates required columns.
    """
    data = pd.read_csv(path, index_col=0, parse_dates=True)
    data.columns = data.columns.str.replace(' ', '_')  # Standardize column names

    # Drop rows with NaN values
    data = data.dropna()
    return data

# Scale and preprocess data
def scale_processD(df, scaler, balance_data=False):
    """
    Scales the data, splits into train/test sets, and balances training data if specified.
    """
    features = get_features(df)  # Dynamically fetch feature columns
    X = df[features]
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

    # Dynamically derive feature names post-scaling
    scaled_features = [f"Feature_{i}" for i in range(X_train_scaled.shape[1])]

    # Convert scaled data back to DataFrame for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=scaled_features)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=scaled_features, index=X_test.index)

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
    Cleans, scales, and prepares train/test datasets.
    """
    # Step 1: Clean the data
    df = clean_DF(file_path)
    print("Cleaned DataFrame:")
    print(df.head())

    scaler = MinMaxScaler(feature_range=(-1, 1))

    # Step 2: Scale features and split
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
    file_path = "data/spy_extended_features.csv"
    train_dataset, test_dataset = prepare_stock_data(file_path, balance_data=True)
    print("\nData preparation complete. Datasets are ready for training.")