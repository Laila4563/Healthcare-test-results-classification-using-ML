import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV


class MultiClassLogisticRegression:
    def __init__(self, hyperparameters=None):
        default_hyperparameters = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'max_iter': [500, 1000, 1500],
            'class_weight': [None, 'balanced']
        }

        self.hyperparameters = hyperparameters if hyperparameters is not None else default_hyperparameters
        self.model = None
        self.best_params_ = None
        self.is_tuned = hyperparameters is not None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if self.is_tuned:
            self._tune_hyperparameters(X_train, y_train, X_val, y_val)
        else:
            self.model = LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                max_iter=1000,
                random_state=42
            )
            self.model.fit(X_train, y_train)

    def _tune_hyperparameters(self, X_train, y_train, X_val, y_val):
        model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            random_state=42
        )

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=self.hyperparameters,
            cv=5,
            n_jobs=-1,
            verbose=1,
            scoring='accuracy'
        )

        X_combined = np.vstack((X_train, X_val))
        y_combined = np.concatenate((y_train, y_val))

        grid_search.fit(X_combined, y_combined)

        self.model = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)
        return accuracy, report

    def get_params(self):
        if self.is_tuned:
            return self.best_params_
        return self.model.get_params()


def print_evaluation(model, X_train, y_train, X_val, y_val, X_test, y_test, label=""):
    print(f"\n=== {label} Model Evaluation ===")

    train_acc, _ = model.evaluate(X_train, y_train)
    val_acc, _ = model.evaluate(X_val, y_val)
    test_acc, test_report = model.evaluate(X_test, y_test)

    print(f"Train Accuracy:      {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy:       {test_acc:.4f}")
    print("\nClassification Report (Test Set):")
    print(test_report)
    