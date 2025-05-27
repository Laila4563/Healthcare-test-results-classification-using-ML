# MLP.py
import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras_tuner as kt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import tempfile
import os
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
import joblib

# === Build MLP Model ===
def build_model(hp, input_shape):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_shape,)))

    for i in range(hp.Int('num_layers', 1, 2)):
        model.add(layers.Dense(
            units=hp.Int(f'units_{i}', min_value=64, max_value=128, step=32),
            activation='relu'
        ))

    model.add(layers.Dense(3, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3])
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# === Default model (no tuning) ===
def build_default_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# === Compute class weights ===
def get_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    return dict(zip(classes, weights))

# === Hyperparameter Tuning ===
def run_hyperparameter_tuning(X_train, y_train, X_val, y_val, input_shape):
    def model_builder(hp):
        return build_model(hp, input_shape)

    temp_dir = tempfile.mkdtemp()

    tuner = kt.RandomSearch(
        model_builder,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=1,
        directory=temp_dir,
        project_name='mlp_healthcare',
        overwrite=True
    )

    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    tuner.search(
        X_train, y_train,
        epochs=10,
        validation_data=(X_val, y_val),  # Use explicit validation set here
        batch_size=64,
        callbacks=[stop_early],
        verbose=1,
        class_weight=get_class_weights(y_train)
    )

    return tuner

# === Final Model Training ===
def train_final_model(tuner, X_train, y_train, X_val, y_val):
    best_hps = tuner.get_best_hyperparameters(1)[0]
    model = tuner.hypermodel.build(best_hps)

    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=64,
        validation_data=(X_val, y_val),  # Use explicit validation set here
        callbacks=[stop_early],
        verbose=1,
        class_weight=get_class_weights(y_train)
    )
    return model, history

# === Default Model Training ===
def train_default_model(X_train, y_train, X_val, y_val, input_shape):
    model = build_default_model(input_shape)

    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=64,
        validation_data=(X_val, y_val),  # Use explicit validation set here
        verbose=1,
        class_weight=get_class_weights(y_train)
    )

    return model, history

    model = build_default_model(input_shape)

    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=64,
        validation_split=0.2,
        verbose=1,
        class_weight=get_class_weights(y_train)
    )

    return model, history

# === Evaluation ===
def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    joblib.dump(model, "saved_models/mlp_model.pkl")

    print(f"Test Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_classes))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_classes))




def plot_mlp_model(history, model_name="MLP"):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title(f'{model_name} Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
