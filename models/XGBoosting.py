import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def xgboost_with_tuning(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Train and evaluate an XGBoost classifier with hyperparameter tuning.

    Parameters:
    - X_train, X_val, X_test: Feature matrices for train, validation, and test
    - y_train, y_val, y_test: Label vectors for train, validation, and test

    Returns:
    - best_model: Best trained XGBoost model after tuning
    """

    # Initialize base XGBoost Classifier (default parameters)
    base_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

    # Fit the base model (before tuning)
    base_model.fit(X_train, y_train)

    # Predictions before tuning
    y_train_pred_base = base_model.predict(X_train)
    y_val_pred_base = base_model.predict(X_val)
    y_test_pred_base = base_model.predict(X_test)

    # Print pre-tuning evaluation metrics
    print("\n[INFO] Evaluation BEFORE Tuning")
    print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred_base):.4f}")
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred_base):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred_base):.4f}")
    print("\nClassification Report (Test Set - Before Tuning):")
    print(classification_report(y_test, y_test_pred_base))

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }

    # Perform Grid Search with Cross-Validation on training + validation set
    grid_search = GridSearchCV(
        base_model, param_grid,
        scoring='accuracy', cv=3,
        n_jobs=-1, verbose=1
    )
    grid_search.fit(np.vstack((X_train, X_val)), np.hstack((y_train, y_val)))

    # Get best model after tuning
    best_model = grid_search.best_estimator_

    # Predictions after tuning
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)

    # Print post-tuning evaluation metrics
    print("\n[INFO] Best Parameters Found:")
    print(grid_search.best_params_)

    print("\n[INFO] Evaluation AFTER Tuning")
    print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    
    print("\nClassification Report (Test Set - After Tuning):")
    print(classification_report(y_test, y_test_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    return best_model
