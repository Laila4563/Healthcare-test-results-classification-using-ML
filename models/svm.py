from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from pyswarm import pso
import numpy as np



# --- LINEAR SVM ---
def linear_svm_with_grid(X_train, y_train, X_val, y_val, X_test, y_test):
    model = LinearSVC(multi_class='ovr', max_iter=10000)
    param_grid = {'C': [0.1, 1, 10]}
    grid = GridSearchCV(model, {'C': param_grid['C']}, cv=3)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)

    print("\n[RESULTS] Linear SVM with Grid Search")
    print("Best Parameters:", grid.best_params_)
    print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}\n")

    print("[Confusion Matrices]")
    print("Train:\n", confusion_matrix(y_train, y_train_pred))
    print("Validation:\n", confusion_matrix(y_val, y_val_pred))
    print("Test:\n", confusion_matrix(y_test, y_test_pred))

    return best_model


def rbf_svm_with_pso(X_train, y_train, X_val, y_val, X_test, y_test):
    def objective(params):
        C, gamma = params
        model = SVC(C=C, gamma=gamma, kernel='rbf', decision_function_shape='ovr')
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return 1 - accuracy_score(y_val, preds)

    # Define PSO bounds
    lb = [0.01, 0.0001]  # Lower bounds for C and gamma
    ub = [100, 1]        # Upper bounds

    best_params, _ = pso(objective, lb, ub, swarmsize=20, maxiter=10)
    best_C, best_gamma = best_params

    final_model = SVC(C=best_C, gamma=best_gamma, kernel='rbf', decision_function_shape='ovr')
    final_model.fit(X_train, y_train)

    y_train_pred = final_model.predict(X_train)
    y_val_pred = final_model.predict(X_val)
    y_test_pred = final_model.predict(X_test)

    return {
        'model': final_model,
        'best_params': {'C': best_C, 'gamma': best_gamma},
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'val_accuracy': accuracy_score(y_val, y_val_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'confusion_matrices': {
            'train': confusion_matrix(y_train, y_train_pred),
            'val': confusion_matrix(y_val, y_val_pred),
            'test': confusion_matrix(y_test, y_test_pred),
        }
    }

def rbf_svm_with_pso(X_train, X_val, X_test, y_train, y_val, y_test):
    # Define the PSO objective function with fixed data passed via args
    def objective(params, X_train, y_train, X_val, y_val):
        C, gamma = params
        model = SVC(C=C, gamma=gamma, kernel='rbf', decision_function_shape='ovr')
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return 1 - accuracy_score(y_val, preds)

    # Define PSO bounds
    lb = [0.01, 0.0001]  # Lower bounds for C and gamma
    ub = [100, 1]        # Upper bounds

    # Call PSO with arguments
    best_params, _ = pso(
        objective, lb, ub,
        args=(X_train, y_train, X_val, y_val),
        swarmsize=20, maxiter=10
    )
    best_C, best_gamma = best_params

    # Final model training
    final_model = SVC(C=best_C, gamma=best_gamma, kernel='rbf', decision_function_shape='ovr')
    final_model.fit(X_train, y_train)

    # Predictions
    y_train_pred = final_model.predict(X_train)
    y_val_pred = final_model.predict(X_val)
    y_test_pred = final_model.predict(X_test)

    # Print results
    print("\n[RESULTS] RBF SVM with PSO Optimization")
    print("Best Parameters: C =", best_C, ", gamma =", best_gamma)
    print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}\n")

    print("[Confusion Matrices]")
    print("Train:\n", confusion_matrix(y_train, y_train_pred))
    print("Validation:\n", confusion_matrix(y_val, y_val_pred))
    print("Test:\n", confusion_matrix(y_test, y_test_pred))

    return {
        'model': final_model,
        'best_params': {'C': best_C, 'gamma': best_gamma},
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'val_accuracy': accuracy_score(y_val, y_val_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'confusion_matrices': {
            'train': confusion_matrix(y_train, y_train_pred),
            'val': confusion_matrix(y_val, y_val_pred),
            'test': confusion_matrix(y_test, y_test_pred),
        }
    }

