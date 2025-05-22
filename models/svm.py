from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def svm_classifier(X_train, X_test, y_train, y_test, kernel='linear'):
    """
    Train and evaluate an SVM model on pre-split data.

    Parameters:
    - X_train, X_test: Feature matrices
    - y_train, y_test: Label vectors
    - kernel: SVM kernel ('linear' or 'rbf')

    Returns:
    - model: Trained SVM model
    """
    
    # Initialize the SVM model
    model = SVC(kernel=kernel, decision_function_shape='ovr')  # Multiclass: One-vs-Rest strategy
    # used OvR rather than OvO due to simplicity and speed

    # Train the model
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Print evaluation metrics
    print(f"\n[INFO] SVM Evaluation (kernel='{kernel}')")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return model
