from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from pyswarm import pso



# --- LINEAR SVM ---
def linear_svm_with_grid(X_train, y_train, X_val, y_val, X_test, y_test):
    # --- Before Tuning: Baseline Model ---
    base_model = LinearSVC(multi_class='ovr', max_iter=10000)
    base_model.fit(X_train, y_train)

    base_train_pred = base_model.predict(X_train)
    base_val_pred = base_model.predict(X_val)
    base_test_pred = base_model.predict(X_test)

    print("\n[INFO] Baseline Linear SVM (Before Grid Search)")
    print(f"Train Accuracy: {accuracy_score(y_train, base_train_pred):.4f}")
    print(f"Validation Accuracy: {accuracy_score(y_val, base_val_pred):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, base_test_pred):.4f}")

    # --- After Tuning with GridSearchCV ---
    param_grid = {'C': [0.1, 1, 10]}
    grid = GridSearchCV(LinearSVC(multi_class='ovr', max_iter=10000), param_grid, cv=3)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    tuned_train_pred = best_model.predict(X_train)
    tuned_val_pred = best_model.predict(X_val)
    tuned_test_pred = best_model.predict(X_test)

    print("\n[INFO] Linear SVM After Grid Search")
    print("Best Parameters:", grid.best_params_)
    print(f"Train Accuracy: {accuracy_score(y_train, tuned_train_pred):.4f}")
    print(f"Validation Accuracy: {accuracy_score(y_val, tuned_val_pred):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, tuned_test_pred):.4f}")

    print("\n[Classification Report - Test Set]")
    print(classification_report(y_test, tuned_test_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, tuned_test_pred))

    return best_model




# --- SVM WITH GAUSSIAN KERNAL  ---
def rbf_svm_with_pso(X_train, X_val, X_test, y_train, y_val, y_test):
    # --- Before Tuning: Baseline RBF SVM ---
    base_model = SVC(kernel='rbf', decision_function_shape='ovr')
    base_model.fit(X_train, y_train)

    base_train_pred = base_model.predict(X_train)
    base_val_pred = base_model.predict(X_val)
    base_test_pred = base_model.predict(X_test)

    print("\n[INFO] Baseline RBF SVM (Before PSO Tuning)")
    print(f"Train Accuracy: {accuracy_score(y_train, base_train_pred):.4f}")
    print(f"Validation Accuracy: {accuracy_score(y_val, base_val_pred):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, base_test_pred):.4f}")

    # --- Define PSO Objective Function ---
    def objective(params, X_train, y_train, X_val, y_val):
        C, gamma = params
        model = SVC(C=C, gamma=gamma, kernel='rbf', decision_function_shape='ovr')
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return 1 - accuracy_score(y_val, preds)

    lb = [0.01, 0.0001]
    ub = [100, 1]

    best_params, _ = pso(
        objective, lb, ub,
        args=(X_train, y_train, X_val, y_val),
        swarmsize=20, maxiter=10
    )

    best_C, best_gamma = best_params

    final_model = SVC(C=best_C, gamma=best_gamma, kernel='rbf', decision_function_shape='ovr')
    final_model.fit(X_train, y_train)

    tuned_train_pred = final_model.predict(X_train)
    tuned_val_pred = final_model.predict(X_val)
    tuned_test_pred = final_model.predict(X_test)

    print("\n[INFO] RBF SVM After PSO Tuning")
    print("Best Parameters: C =", best_C, ", gamma =", best_gamma)
    print(f"Train Accuracy: {accuracy_score(y_train, tuned_train_pred):.4f}")
    print(f"Validation Accuracy: {accuracy_score(y_val, tuned_val_pred):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, tuned_test_pred):.4f}")

    print("\n[Classification Report - Test Set]")
    print(classification_report(y_test, tuned_test_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, tuned_test_pred))

    return final_model

