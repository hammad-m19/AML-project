"""Deep learning model runners using TensorFlow/Keras — ANN, LSTM, CNN."""

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping


# ── ANN (Feed-Forward) ──────────────────────────────────────────────────

def _build_ann_classifier(input_dim, n_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _build_ann_regressor(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# ── LSTM ─────────────────────────────────────────────────────────────────

def _build_lstm_classifier(input_dim, n_classes):
    """LSTM for tabular data: treat each feature as one timestep."""
    model = models.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _build_lstm_regressor(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# ── CNN (1D Convolution) ────────────────────────────────────────────────

def _build_cnn_classifier(input_dim, n_classes):
    """1D-CNN for tabular data: treats features as a 1D signal."""
    model = models.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
        layers.MaxPooling1D(pool_size=2, padding="same"),
        layers.Conv1D(32, kernel_size=3, padding="same", activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _build_cnn_regressor(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
        layers.MaxPooling1D(pool_size=2, padding="same"),
        layers.Conv1D(32, kernel_size=3, padding="same", activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# ── Pipeline ─────────────────────────────────────────────────────────────

_ARCH_NAMES = {
    "ann": "ANN (Artificial Neural Network)",
    "lstm": "LSTM (Long Short-Term Memory)",
    "cnn": "CNN (Convolutional Neural Network)",
}


def run_dl_pipeline(preprocessed, architecture="ann", epochs=30, batch_size=32):
    """Train a deep learning model on preprocessed tabular data.

    Args:
        architecture: One of 'ann', 'lstm', 'cnn'.
    """
    tf.keras.backend.clear_session()

    info = preprocessed["info"]
    task_type = info["task_type"]
    X_train = preprocessed["X_train"]
    X_test = preprocessed["X_test"]
    y_train = preprocessed["y_train"]
    y_test = preprocessed["y_test"]
    label_encoder = preprocessed["label_encoder"]

    input_dim = X_train.shape[1]
    needs_reshape = architecture in ("lstm", "cnn")

    # Reshape for LSTM / CNN: (samples, features) → (samples, features, 1)
    if needs_reshape:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    if task_type == "classification":
        n_classes = len(np.unique(y_train))
        builders = {
            "ann": _build_ann_classifier,
            "lstm": _build_lstm_classifier,
            "cnn": _build_cnn_classifier,
        }
        if architecture not in builders:
            raise ValueError(f"Unknown architecture: {architecture}")
        model = builders[architecture](input_dim, n_classes)

        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0,
        )
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        acc = accuracy_score(y_test, y_pred)
        results = {
            "algorithm": architecture,
            "task_type": "classification",
            "model_name": _ARCH_NAMES.get(architecture, architecture),
            "metrics": {
                "accuracy": round(float(acc), 4),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "classification_report": classification_report(
                    y_test, y_pred, output_dict=True, zero_division=0
                ),
                "labels": label_encoder.classes_.tolist() if label_encoder else None,
            },
        }
    else:
        builders = {
            "ann": _build_ann_regressor,
            "lstm": _build_lstm_regressor,
            "cnn": _build_cnn_regressor,
        }
        if architecture not in builders:
            raise ValueError(f"Unknown architecture: {architecture}")
        model = builders[architecture](input_dim)

        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0,
        )
        y_pred = model.predict(X_test, verbose=0).flatten()
        results = {
            "algorithm": architecture,
            "task_type": "regression",
            "model_name": _ARCH_NAMES.get(architecture, architecture),
            "metrics": {
                "r2_score": round(float(r2_score(y_test, y_pred)), 4),
                "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
            },
            "predictions": {
                "actual": [round(float(v), 4) for v in y_test[:50]],
                "predicted": [round(float(v), 4) for v in y_pred[:50]],
            },
        }

    # Training history
    results["training_history"] = {
        "loss": [round(float(v), 4) for v in history.history["loss"]],
        "val_loss": [round(float(v), 4) for v in history.history["val_loss"]],
    }
    if "accuracy" in history.history:
        results["training_history"]["accuracy"] = [
            round(float(v), 4) for v in history.history["accuracy"]
        ]
        results["training_history"]["val_accuracy"] = [
            round(float(v), 4) for v in history.history["val_accuracy"]
        ]

    return results
