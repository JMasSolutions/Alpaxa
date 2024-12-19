import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_class_weight
import joblib  # For saving and loading the model
import numpy as np
import os


# Load data
def load_data():
    """
    Loads preprocessed and scaled training and testing data.
    """
    try:
        X_train = pd.read_csv('data/scaled_features_train.csv')
        X_test = pd.read_csv('data/scaled_features_test.csv')
        y_train = pd.read_csv('data/target_train.csv')['Target']
        y_test = pd.read_csv('data/target_test.csv')['Target']
        print("Data loaded successfully.")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)


# Handle class imbalance using class weights
def compute_class_weights_func(y):
    """
    Computes class weights to handle class imbalance.
    """
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weights = {cls: weight for cls, weight in zip(classes, weights)}
    print(f"Computed Class Weights: {class_weights}")
    return class_weights


# Train LightGBM model with class weights
def train_lightgbm(X_train, y_train, class_weights):
    """
    Trains a LightGBM classifier with class weights.
    """
    lgbm = LGBMClassifier(
        random_state=42,
        class_weight=class_weights,
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8
    )
    lgbm.fit(X_train, y_train)
    print("LightGBM model trained successfully.")
    return lgbm


# Evaluate model performance
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model using various classification metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of class '1' (Buy)

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=['Sell', 'Buy'])
    cm = confusion_matrix(y_test, y_pred)

    print(f'\nLightGBM Accuracy: {acc:.4f}')
    print(f'LightGBM ROC-AUC: {roc_auc:.4f}')
    print('\nClassification Report:')
    print(report)
    print('Confusion Matrix:')
    print(cm)

    # Save evaluation metrics
    os.makedirs('model', exist_ok=True)
    with open('model/evaluation_metrics.txt', 'w') as f:
        f.write(f'Accuracy: {acc:.4f}\n')
        f.write(f'ROC-AUC: {roc_auc:.4f}\n')
        f.write('Classification Report:\n')
        f.write(report)
        f.write('Confusion Matrix:\n')
        f.write(np.array2string(cm))

    return acc, roc_auc, report, cm


# Save the trained model
def save_model(model, model_path='model/lightgbm_model.pkl'):
    """
    Saves the trained model to the specified path.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved as {model_path}")


# Plot feature importance (Optional)
def plot_feature_importance(model, feature_names):
    """
    Plots the feature importance of the trained model.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('model/feature_importance.png')
    plt.show()
    print("Feature importance plot saved as 'model/feature_importance.png'.")


# Main training function
def main():
    # Step 1: Load data
    X_train, X_test, y_train, y_test = load_data()

    # Step 2: Clean feature names by replacing spaces with underscores
    X_train.columns = X_train.columns.str.replace(' ', '_')
    X_test.columns = X_test.columns.str.replace(' ', '_')

    # Step 3: Retrieve actual feature names used in training
    actual_features = X_train.columns.tolist()
    print(f"Actual Features Used: {actual_features}")

    # Step 4: Compute class weights
    class_weights = compute_class_weights_func(y_train)

    # Step 5: Train LightGBM model with class weights
    model = train_lightgbm(X_train, y_train, class_weights)

    # Step 6: Evaluate model performance
    acc, roc_auc, report, cm = evaluate_model(model, X_test, y_test)

    # Step 7: Save the trained model
    save_model(model, 'model/lightgbm_model.pkl')

    # Step 8: (Optional) Plot Feature Importance
    plot_feature_importance(model, actual_features)

    # Step 9: Example inference
    # Replace `new_data` with the actual data for which you want predictions
    # Ensure `new_data` has the same features and preprocessing applied as the training data
    new_data = X_test.iloc[:5].values  # Example: using the first 5 rows of test data
    predictions = model.predict(new_data)

    print("\nPredictions on new data:", predictions)


if __name__ == "__main__":
    main()