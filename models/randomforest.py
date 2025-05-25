import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

def random_forest_with_grid(X_train, y_train, X_val, y_val, X_test, y_test):
    # -------- Initial Model (Before Tuning) --------
    base_model = RandomForestClassifier(random_state=42)
    base_model.fit(X_train, y_train)

    base_train_pred = base_model.predict(X_train)
    base_val_pred = base_model.predict(X_val)
    base_test_pred = base_model.predict(X_test)

    print("\n[INFO] Baseline Random Forest (Before Hyperparameter Tuning)")
    print(f"Train Accuracy: {accuracy_score(y_train, base_train_pred):.4f}")
    print(f"Validation Accuracy: {accuracy_score(y_val, base_val_pred):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, base_test_pred):.4f}")

    # -------- Grid Search Hyperparameter Tuning --------
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    # Combine training and validation for more robust training during tuning
    grid_search.fit(np.vstack((X_train, X_val)), np.hstack((y_train, y_val)))
    best_model = grid_search.best_estimator_

    tuned_train_pred = best_model.predict(X_train)
    tuned_val_pred = best_model.predict(X_val)
    tuned_test_pred = best_model.predict(X_test)

    print("\n[INFO] Best Parameters Found:")
    print(grid_search.best_params_)

    print("\n[INFO] Random Forest After Hyperparameter Tuning")
    print(f"Train Accuracy: {accuracy_score(y_train, tuned_train_pred):.4f}")
    print(f"Validation Accuracy: {accuracy_score(y_val, tuned_val_pred):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, tuned_test_pred):.4f}")

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, tuned_test_pred))
    
    return best_model
