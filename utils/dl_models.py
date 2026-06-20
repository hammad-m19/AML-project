"""Deep learning model runners using TensorFlow/Keras."""

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


def _build_classifier(input_dim, n_classes):
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _build_regressor(input_dim):
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def run_dl_pipeline(preprocessed, algorithm="neural_network", epochs=30, batch_size=32):
    """Train a feed-forward neural network on preprocessed tabular data."""
    tf.keras.backend.clear_session()

    info = preprocessed["info"]
    task_type = info["task_type"]
    X_train = preprocessed["X_train"]
    X_test = preprocessed["X_test"]
    y_train = preprocessed["y_train"]
    y_test = preprocessed["y_test"]
    label_encoder = preprocessed["label_encoder"]

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    if task_type == "classification":
        n_classes = len(np.unique(y_train))
        model = _build_classifier(X_train.shape[1], n_classes)
        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0,
        )
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        acc = accuracy_score(y_test, y_pred)
        results = {
            "algorithm": algorithm,
            "task_type": "classification",
            "model_name": "Deep Neural Network (Classifier)",
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
        model = _build_regressor(X_train.shape[1])
        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0,
        )
        y_pred = model.predict(X_test, verbose=0).flatten()
        results = {
            "algorithm": algorithm,
            "task_type": "regression",
            "model_name": "Deep Neural Network (Regressor)",
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
