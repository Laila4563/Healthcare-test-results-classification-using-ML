import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def random_forest_with_grid(X_train, y_train, X_val, y_val, X_test, y_test):
    
    model = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(np.vstack((X_train, X_val)), np.hstack((y_train, y_val)))

    best_model = grid_search.best_estimator_

    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)

    print("\n[INFO] Best Parameters Found:")
    print(grid_search.best_params_)

    print("\n[INFO] Random Forest Evaluation")
    print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.2%}")
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.2%}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.2%}")

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))

    return {
        'model': best_model,
        'best_params': grid_search.best_params_,
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'val_accuracy': accuracy_score(y_val, y_val_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'confusion_matrices': {
            'train': confusion_matrix(y_train, y_train_pred),
            'val': confusion_matrix(y_val, y_val_pred),
            'test': confusion_matrix(y_test, y_test_pred),
        }
    }