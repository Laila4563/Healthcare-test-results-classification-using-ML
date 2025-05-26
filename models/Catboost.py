import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def catboost_with_tuning(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Train and evaluate a CatBoost classifier with and without hyperparameter tuning.

    Parameters:
    - X_train, X_val, X_test: Feature matrices for train, validation, and test
    - y_train, y_val, y_test: Label vectors for train, validation, and test

    Returns:
    - best_model: Best trained CatBoost model after tuning
    """

    # ----------- BEFORE TUNING -----------
    print("\n[INFO] Evaluation BEFORE Tuning")

    default_model = CatBoostClassifier(verbose=0, random_state=42)
    default_model.fit(X_train, y_train)

    y_train_pred_before = default_model.predict(X_train)
    y_val_pred_before = default_model.predict(X_val)
    y_test_pred_before = default_model.predict(X_test)

    print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred_before):.4f}")
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred_before):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred_before):.4f}")

    print("\nClassification Report (Test Set - Before Tuning):")
    print(classification_report(y_test, y_test_pred_before))

    # ----------- HYPERPARAMETER TUNING -----------
    model = CatBoostClassifier(verbose=0, random_state=42)

    param_grid = {
        'iterations': [100, 200],
        'learning_rate': [0.01, 0.1],
        'depth': [4, 6, 8],
        'l2_leaf_reg': [1, 3, 5]
    }

    grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(np.vstack((X_train, X_val)), np.hstack((y_train, y_val)))

    best_model = grid_search.best_estimator_

    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)

    print("\n[INFO] Best Parameters Found:")
    print(grid_search.best_params_)

    print("\n[INFO] CatBoost Evaluation AFTER Tuning")
    print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

    print("\nClassification Report (Test Set - After Tuning):")
    print(classification_report(y_test, y_test_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    return best_model
