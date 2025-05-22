from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def run_random_forest(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=None, random_state=42):
    """
    Train and evaluate a Random Forest classifier for multiclass classification.

    Parameters:
    - X_train, X_test: Feature matrices
    - y_train, y_test: Label vectors
    - n_estimators: Number of trees in the forest (default=100 (most common)) (More trees → better accuracy but slower training)
    - max_depth: Maximum depth of each tree (default=None, grows until pure or min samples) (Controls complexity and overfitting)
    - random_state: Seed for reproducibility (A seed value to control randomness) (Same results on every run)

    Returns:
    - model: Trained Random Forest model
    """
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Print evaluation metrics
    print("\n[INFO] Random Forest Evaluation")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return model
