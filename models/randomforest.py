import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

def random_forest_with_tuning(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Train and evaluate a Random Forest classifier with hyperparameter tuning (no OvR).

    Parameters:
    - X_train, X_val, X_test: Feature matrices for train, validation, and test
    - y_train, y_val, y_test: Label vectors for train, validation, and test

    Returns:
    - best_model: Best trained Random Forest model after tuning
    """

    # Initialize base Random Forest Classifier
    model = RandomForestClassifier(random_state=42)

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    """
    - n_estimators: Number of trees in the forest (default=100 (most common)) (More trees → better accuracy but slower training)
    - max_depth: Maximum depth of each tree (default=None, grows until pure or min samples) (Controls complexity and overfitting)
    - random_state: Seed for reproducibility (A seed value to control randomness) (Same results on every run)
    """


    # Perform Grid Search with Cross-Validation on training + validation set
    grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(np.vstack((X_train, X_val)), np.hstack((y_train, y_val)))

    # Get best model after tuning
    best_model = grid_search.best_estimator_

    # Predictions
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)

    # Print evaluation metrics
    print("\n[INFO] Best Parameters Found:")
    print(grid_search.best_params_)

    print("\n[INFO] Random Forest Evaluation")
    print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.2%}")
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.2%}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.2%}")

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))

    return best_model