import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Load data from CSV files
scaled_features = pd.read_csv('data/scaled_features.csv')
target = pd.read_csv('data/target.csv')

# Convert to numpy arrays
X = scaled_features.values
y = target.values.flatten()

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# SVM with RandomizedSearchCV
def svm_classifier():
    svm = SVC()
    params = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.001, 0.01]
    }
    tuned_svm = RandomizedSearchCV(svm, param_distributions=params, n_iter=10, scoring='accuracy', cv=6, random_state=42, n_jobs=-1)
    return tuned_svm

# Random Forest with RandomizedSearchCV
def rf_classifier():
    rf = RandomForestClassifier(random_state=42)
    params = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5],
    }
    tuned_rf = RandomizedSearchCV(rf, param_distributions=params, n_iter=5, scoring='accuracy', cv=4, random_state=42, n_jobs=-1)
    return tuned_rf

# XGBoost with RandomizedSearchCV
def gradient_boosting_classifier():
    xgb_classifier = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
    params = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.01],
        'subsample': [0.8, 1.0],
    }
    tuned_xgb = RandomizedSearchCV(xgb_classifier, param_distributions=params, n_iter=5, scoring='accuracy', cv=4, random_state=42, n_jobs=-1)
    return tuned_xgb

# k-NN with RandomizedSearchCV
def knn_classifier():
    knn = KNeighborsClassifier()
    params = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
    }
    tuned_knn = RandomizedSearchCV(knn, param_distributions=params, n_iter=5, scoring='accuracy', cv=4, random_state=42, n_jobs=-1)
    return tuned_knn

# MLP with RandomizedSearchCV
def mlp_classifier():
    mlp = MLPClassifier(max_iter=500, random_state=42)
    params = {
        'hidden_layer_sizes': [(64,), (64, 32)],
        'alpha': [0.0001, 0.001],
        'learning_rate_init': [0.001, 0.01],
    }
    tuned_mlp = RandomizedSearchCV(mlp, param_distributions=params, n_iter=5, scoring='accuracy', cv=4, random_state=42, n_jobs=-1)
    return tuned_mlp

# List of classifiers
classifiers = {
    'SVM': svm_classifier(),
    'Random Forest': rf_classifier(),
    'XGBoost': gradient_boosting_classifier(),
    'k-NN': knn_classifier(),
    'MLP': mlp_classifier()
}

best_val_accuracy = 0
best_model = None
best_model_name = None

# Evaluate each classifier
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    val_predictions = clf.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model = clf
        best_model_name = name

# Evaluate the best model on the test set 100 times
test_accuracies = []

for _ in range(100):
    val_predictions = best_model.predict(X_val)
    test_accuracy = accuracy_score(y_val, val_predictions)
    test_accuracies.append(test_accuracy)

print(f'Best Model: {best_model_name}')
print(f'Best Params: {best_model.best_params_}')
print(f'Average Test Accuracy: {np.mean(test_accuracies):.4f}')