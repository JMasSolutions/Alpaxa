from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import joblib
import os

def load_data():
    X_train = pd.read_csv('data/scaled_features_train.csv')
    X_test = pd.read_csv('data/scaled_features_test.csv')
    y_train = pd.read_csv('data/target_train.csv')['Target']
    y_test = pd.read_csv('data/target_test.csv')['Target']
    return X_train, X_test, y_train, y_test

def hyperparameter_tuning(X_train, y_train, class_weights):
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05],
        'num_leaves': [31, 50],
        'max_depth': [-1, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    lgbm = LGBMClassifier(random_state=42, class_weight=class_weights)
    grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=3, scoring='roc_auc', verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    print(f'Accuracy: {acc:.4f}, ROC-AUC: {roc_auc:.4f}, F1 Score: {f1:.4f}')
    print("Confusion Matrix:", confusion_matrix(y_test, y_pred))

def main():
    X_train, X_test, y_train, y_test = load_data()
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(zip(np.unique(y_train), class_weights))
    model = hyperparameter_tuning(X_train, y_train, class_weights)
    evaluate_model(model, X_test, y_test)
    joblib.dump(model, "model/lightgbm_model.pkl")
    print("Model saved as 'model/lightgbm_model.pkl'")

if __name__ == "__main__":
    main()