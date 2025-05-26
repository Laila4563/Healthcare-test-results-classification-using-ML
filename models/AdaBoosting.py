import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def adaboost_with_tuning(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Train and evaluate an AdaBoost classifier with hyperparameter tuning.

    Parameters:
    - X_train, X_val, X_test: Feature matrices for train, validation, and test
    - y_train, y_val, y_test: Label vectors for train, validation, and test

    Returns:
    - best_model: Best trained AdaBoost model after tuning
    """

    # Initialize base AdaBoost classifier with DecisionTree as base estimator
    base_estimator = DecisionTreeClassifier(random_state=42)
    model = AdaBoostClassifier(estimator=base_estimator, random_state=42)

    # ----------- BEFORE TUNING -----------
    model.fit(X_train, y_train)

    y_train_pred_before = model.predict(X_train)
    y_val_pred_before = model.predict(X_val)
    y_test_pred_before = model.predict(X_test)

    print("\n[INFO] Evaluation BEFORE Tuning")
    print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred_before):.4f}")
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred_before):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred_before):.4f}")

    print("\nClassification Report (Test Set - Before Tuning):")
    print(classification_report(y_test, y_test_pred_before))

    # ----------- HYPERPARAMETER TUNING -----------
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 1],
        'estimator__max_depth': [1, 3, 5]
    }

    grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(np.vstack((X_train, X_val)), np.hstack((y_train, y_val)))

    best_model = grid_search.best_estimator_

    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)

    print("\n[INFO] Best Parameters Found:")
    print(grid_search.best_params_)

    print("\n[INFO] AdaBoost Evaluation")
    print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    return best_model
