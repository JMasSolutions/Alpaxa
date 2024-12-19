import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib  # For saving and loading the model

# Load data
scaled_features = pd.read_csv('data/scaled_features.csv')
target = pd.read_csv('data/target.csv')

# Prepare data
X = scaled_features.values
y = target['Target'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Select all features for LightGBM
important_features = scaled_features.columns  # Keeping all features for LightGBM
X_important = scaled_features[important_features].values

# Handle class imbalance using SMOTE
print(f"Original class distribution: {Counter(y)}")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_important, y)
print(f"Class distribution after SMOTE: {Counter(y_resampled)}")

# Transform X_test to include all selected features
X_test_important = pd.DataFrame(X_test, columns=scaled_features.columns)[important_features].values

# Train LightGBM
lgbm = LGBMClassifier(random_state=42)
lgbm.fit(X_resampled, y_resampled)

# Evaluate LightGBM
lgbm_accuracy = accuracy_score(y_test, lgbm.predict(X_test_important))
print(f'LightGBM Accuracy: {lgbm_accuracy}')

# Save the trained LightGBM model
joblib.dump(lgbm, 'lightgbm_model.pkl')
print("Model saved as lightgbm_model.pkl")

# Load the model for inference
loaded_model = joblib.load('lightgbm_model.pkl')
print("Model loaded successfully!")

# Example inference
# Replace `new_data` with the actual data for which you want predictions
# Ensure `new_data` has the same features and preprocessing applied as the training data
new_data = X_test_important[:5]  # Example: using the first 5 rows of test data
predictions = loaded_model.predict(new_data)

print("Predictions:", predictions)