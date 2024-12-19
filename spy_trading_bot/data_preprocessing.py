import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import joblib  # Import joblib to save the scaler
import os  # To handle directory creation

# Formatting the DataFrame to our model's liking
def clean_DF(path):
    """
    Cleans the raw data and keeps only relevant columns.
    Removes rows with NaN values.
    """
    data = pd.read_csv(path, index_col=0, parse_dates=True)

    columns_to_keep = [
        'Adj Close', 'SMA_14', 'EMA_14', 'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower',
        'ATR', 'Stoch_K', 'Stoch_D', 'USD_JPY', 'VIX', 'Gold', 'Oil', 'Monthly_Return', 'Price_Change', 'Target'
    ]

    # Filter the columns to keep only what matters
    cleaned_data = data[columns_to_keep].dropna()
    return cleaned_data

# Scale & preprocess data
def scale_processD(df, scaler):
    """
    Scales the features using the specified scaler.
    Target column remains unscaled.
    """
    scaled_data = scaler.fit_transform(df.drop(columns=["Target"]))
    return pd.DataFrame(scaled_data, columns=df.columns[:-1], index=df.index), df["Target"]

# PyTorch Dataset class for our data
class StockDataSet(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Prepare and return datasets for training
def prepare_stock_data(file_path):
    """
    Cleans, scales, and creates train/test datasets.
    """
    # Step 1: Clean the data
    df = clean_DF(file_path)
    print("Cleaned DataFrame:")
    print(df.head())

    # Step 2: Scale the features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_features, target = scale_processD(df, scaler)

    print(f"\nScaled Features Shape: {scaled_features.shape}")
    print(f"Target Shape: {target.shape}")
    print("\nFirst 5 Scaled Features:")
    print(scaled_features.head())
    scaled_features.to_csv("data/scaled_features.csv", index=False)
    target.to_csv("data/target.csv", index=False)

    # Step 3: Save the scaler for future use
    os.makedirs('model', exist_ok=True)  # Ensure the 'model' directory exists
    joblib.dump(scaler, 'model/scaler.pkl')
    print("Scaler saved as 'model/scaler.pkl'")

    # Step 4: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, shuffle=False)

    # Step 5: Create PyTorch Datasets
    train_dataset = StockDataSet(X_train.values, y_train.values)
    test_dataset = StockDataSet(X_test.values, y_test.values)

    print(f"\nTraining Dataset Length: {len(train_dataset)}")
    print(f"Testing Dataset Length: {len(test_dataset)}")

    return train_dataset, test_dataset

# Main block
if __name__ == "__main__":
    file_path = "data/spy_daily_data.csv"

    # Prepare datasets
    train_dataset, test_dataset = prepare_stock_data(file_path)

    print("\nData preparation complete. Datasets are ready for training.")