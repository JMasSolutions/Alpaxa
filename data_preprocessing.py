import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Formatting the DF to our models liking
def clean_DF(path):
    data = pd.read_csv(path, index_col=0, parse_dates=True)

    columns_to_keep = [
        'Adj Close', 'SMA_14', 'EMA_14', 'RSI', 'MACD', 'USD_JPY',
        'VIX', 'Monthly_Return', 'Price_Change', 'Sentiment_Score', 'Target'
    ]

    # Filter the columns to keep only what matters
    cleaned_data = data[columns_to_keep].dropna()
    return cleaned_data

# Scale & preprocess data
def scale_processD(df, scaler):
    scaled_data = scaler.fit_transform(df.drop(columns=["Target"]))
    return pd.DataFrame(scaled_data, columns=df.columns[:-1], index=df.index), df["Target"]

# Create sequences for LSTM
def create_sequences(features, target, sequence_length):
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
        y.append(target[i + sequence_length])
    return np.array(X), np.array(y)


# Load and clean the data
file_path = "data/tsla_monthly_sentiment_data.csv"
df = clean_DF(file_path)
print("Cleaned DataFrame:")
print(df.head())

# Scaling
scaler = StandardScaler()
scaled_features, target = scale_processD(df, scaler)

# Create sequences
sequence_length = 10
X, y = create_sequences(scaled_features.values, target.values, sequence_length)

print(f"\nFeature Shape: {X.shape} \nTarget Shape: {y.shape}")