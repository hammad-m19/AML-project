"""Classical machine learning model runners."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.svm import SVC


def _classification_metrics(y_test, y_pred, label_encoder):
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    labels = label_encoder.classes_.tolist() if label_encoder else None
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": round(float(acc), 4),
        "confusion_matrix": cm,
        "classification_report": report,
        "labels": labels,
    }


def _regression_metrics(y_test, y_pred):
    return {
        "r2_score": round(float(r2_score(y_test, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
    }


def run_ml_pipeline(preprocessed, algorithm):
    """Run selected ML algorithm and return metrics + chart data."""
    info = preprocessed["info"]
    task_type = info["task_type"]
    X_train = preprocessed["X_train"]
    X_test = preprocessed["X_test"]
    y_train = preprocessed["y_train"]
    y_test = preprocessed["y_test"]
    label_encoder = preprocessed["label_encoder"]

    results = {"algorithm": algorithm, "task_type": task_type}

    if algorithm == "kmeans":
        from utils.data_preprocessing import preprocess_for_clustering

        raise ValueError("Use run_clustering for K-Means.")

    if task_type == "classification":
        models = {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "svm": SVC(kernel="rbf", random_state=42),
        }
        if algorithm not in models:
            raise ValueError(f"Unknown classification algorithm: {algorithm}")
        model = models[algorithm]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results["metrics"] = _classification_metrics(y_test, y_pred, label_encoder)
        if hasattr(model, "feature_importances_"):
            results["feature_importance"] = model.feature_importances_.tolist()
        elif hasattr(model, "coef_"):
            results["feature_importance"] = np.abs(model.coef_[0]).tolist()
    else:
        models = {
            "linear_regression": LinearRegression(),
            "random_forest_regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        }
        if algorithm not in models:
            raise ValueError(f"Unknown regression algorithm: {algorithm}")
        model = models[algorithm]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results["metrics"] = _regression_metrics(y_test, y_pred)
        results["predictions"] = {
            "actual": [round(float(v), 4) for v in y_test[:50]],
            "predicted": [round(float(v), 4) for v in y_pred[:50]],
        }
        if hasattr(model, "feature_importances_"):
            results["feature_importance"] = model.feature_importances_.tolist()

    results["model_name"] = algorithm.replace("_", " ").title()
    return results


def run_clustering(X, n_clusters=3):
    """Run K-Means clustering."""
    n_clusters = min(n_clusters, max(2, len(X) // 2))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0.0

    return {
        "algorithm": "kmeans",
        "task_type": "clustering",
        "metrics": {
            "n_clusters": int(n_clusters),
            "silhouette_score": round(float(sil), 4),
            "inertia": round(float(model.inertia_), 4),
        },
        "cluster_labels": labels.tolist(),
        "cluster_centers": model.cluster_centers_.tolist(),
    }
