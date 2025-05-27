import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def naive_bayes_classifier(X_train, X_val, X_test, y_train, y_val, y_test, label=""):
    """
    Trains a default Gaussian Naive Bayes model using training data and evaluates on validation and test sets.
    """
    model = GaussianNB()
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    print(f"Naive Bayes {label} - Validation Performance:")
    print(classification_report(y_val, y_val_pred))

    y_test_pred = model.predict(X_test)
    print(f"Naive Bayes {label} - Test Performance:")
    print(classification_report(y_test, y_test_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    return model


def naive_bayes_with_grid_search(
    X_train, X_val, X_test, y_train, y_val, y_test, label=""
):
    """
    Trains a Gaussian Naive Bayes model with hyperparameter tuning using validation set.
    The best model is selected based on validation performance, not cross-validation.
    """
    param_grid = {"var_smoothing": np.logspace(-12, -1, 12)}

    best_model = None
    best_score = -np.inf
    best_params = {}

    for var in param_grid["var_smoothing"]:
        model = GaussianNB(var_smoothing=var)
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)

        if score > best_score:
            best_score = score
            best_model = model
            best_params = {"var_smoothing": var}

    print(f"\n[INFO] Best parameters for Naive Bayes {label}: {best_params}")

    y_val_pred = best_model.predict(X_val)
    print(f"\nNaive Bayes {label} (Tuned) - Validation Performance:")
    print(classification_report(y_val, y_val_pred))

    y_test_pred = best_model.predict(X_test)
    print(f"Naive Bayes {label} (Tuned) - Test Performance:")
    print(classification_report(y_test, y_test_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    
    # Save the best model
    joblib.dump(best_model, f"saved_models/nb_best_model.pkl")

    return best_model
