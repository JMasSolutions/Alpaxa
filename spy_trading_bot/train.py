import pandas as pd
import numpy as np
import joblib
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, StratifiedKFold
from lightgbm import LGBMClassifier
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report, confusion_matrix
import os

# Load data
def load_data():
    """
    Loads preprocessed and scaled training and testing data.
    """
    X_train = pd.read_csv('data/scaled_features_train.csv')
    X_test = pd.read_csv('data/scaled_features_test.csv')
    y_train = pd.read_csv('data/target_train.csv')['Target']
    y_test = pd.read_csv('data/target_test.csv')['Target']

    # Ensure alignment
    if len(X_train) != len(y_train):
        print(f"Aligning X_train and y_train: {len(X_train)} features, {len(y_train)} targets.")
        X_train = X_train.iloc[:len(y_train)]

    print("Data loaded successfully.")
    return X_train, X_test, y_train, y_test

# Compute class weights
def compute_class_weights_func(y):
    """
    Computes class weights to handle class imbalance.
    """
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weights = dict(zip(classes, weights))
    print(f"Class weights: {class_weights}")
    return class_weights

# Feature selection
def select_features(X_train, y_train, k=30):
    """
    Selects top k features using univariate statistical tests.
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    selected_features = [X_train.columns[i] for i in selector.get_support(indices=True)]
    print(f"Selected features: {selected_features}")
    return pd.DataFrame(X_train_selected, columns=selected_features), selected_features

# Train LightGBM model with Bayesian optimization
def train_lgbm(X_train, y_train, selected_features, class_weights):
    """
    Trains a LightGBM model using Bayesian optimization for hyperparameter tuning.
    """
    param_grid = {
        'n_estimators': (50, 300),
        'learning_rate': (0.01, 0.2, 'log-uniform'),
        'num_leaves': (20, 150),
        'max_depth': (-1, 20),
        'subsample': (0.6, 1.0, 'uniform'),
        'colsample_bytree': (0.6, 1.0, 'uniform')
    }

    lgbm = LGBMClassifier(
        random_state=42,
        class_weight=class_weights
    )

    bayes_search = BayesSearchCV(
        estimator=lgbm,
        search_spaces=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_iter=30,
        random_state=42,
        verbose=1
    )

    print("Starting Bayesian optimization...")
    bayes_search.fit(X_train, y_train)

    print(f"Best parameters: {bayes_search.best_params_}")
    return bayes_search.best_estimator_

# Evaluate model performance
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model using various metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}, ROC-AUC: {roc_auc:.4f}, F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return acc, roc_auc, f1

# Save the model
def save_model(model, model_path='model/lightgbm_model.pkl'):
    """
    Saves the trained model to disk.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved as '{model_path}'")

# Main training pipeline
def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Dynamically fetch features
    features = list(X_train.columns)

    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    # Compute class weights
    class_weights = compute_class_weights_func(y_train)

    # Feature selection
    X_train_selected, selected_features = select_features(X_train, y_train, k=30)

    # Train LightGBM with Bayesian optimization
    model = train_lgbm(X_train_selected, y_train, selected_features, class_weights)

    # Evaluate the model
    evaluate_model(model, X_test[selected_features], y_test)

    # Save the trained model
    save_model(model, 'model/lightgbm_model.pkl')

if __name__ == "__main__":
    main()