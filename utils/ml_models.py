"""Classical machine learning model runners."""

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)


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

    results = {"algorithm": algorithm, "task_type": task_type}

    if algorithm == "kmeans":
        raise ValueError("Use run_clustering for K-Means.")

    if algorithm == "dbscan":
        raise ValueError("Use run_dbscan for DBSCAN.")

    if algorithm != "linear_regression":
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: linear_regression")

    # Linear Regression (works for both regression and classification-as-regression)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results["task_type"] = "regression"
    results["metrics"] = _regression_metrics(y_test, y_pred)
    results["predictions"] = {
        "actual": [round(float(v), 4) for v in y_test[:50]],
        "predicted": [round(float(v), 4) for v in y_pred[:50]],
    }
    if hasattr(model, "coef_"):
        results["feature_importance"] = np.abs(model.coef_).tolist()

    results["model_name"] = "Linear Regression"
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


def run_dbscan(X, eps=0.5, min_samples=5):
    """Run DBSCAN clustering."""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))

    # Silhouette score only valid when we have 2+ clusters and not all noise
    if n_clusters >= 2 and n_noise < len(labels):
        non_noise_mask = labels != -1
        if len(np.unique(labels[non_noise_mask])) >= 2:
            sil = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
        else:
            sil = 0.0
    else:
        sil = 0.0

    return {
        "algorithm": "dbscan",
        "task_type": "clustering",
        "metrics": {
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "silhouette_score": round(float(sil), 4),
            "total_points": int(len(labels)),
        },
        "cluster_labels": labels.tolist(),
    }
