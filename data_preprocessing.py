import pandas as pd

def clean_DF(path):

    data = pd.read_csv(path, index_col=0, parse_dates=True)

    columns_to_keep = [
        'Adj Close', 'SMA_14', 'EMA_14', 'RSI', 'MACD', 'USD_JPY',
        'VIX', 'Monthly_Return', 'Price_Change', 'Sentiment_Score', 'Target'
    ]

    # Filter the columns to keep only what matters
    cleaned_data = data[columns_to_keep].dropna()

    return cleaned_data


file_path = "data/tsla_monthly_sentiment_data.csv"
df = clean_DF(file_path)
print(df.head())

