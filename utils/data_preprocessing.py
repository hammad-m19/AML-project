"""Data preprocessing utilities for CSV uploads."""

import io
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def load_csv(file_storage):
    """Load CSV from Flask file storage."""
    content = file_storage.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(content))


def analyze_dataset(df):
    """Return summary statistics and metadata for visualization."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": df.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(missing[col]) for col in df.columns},
        "missing_percent": {col: float(missing_pct[col]) for col in df.columns},
        "describe_numeric": df[numeric_cols].describe().round(4).to_dict()
        if numeric_cols
        else {},
        "head": df.head(10).fillna("").astype(str).to_dict(orient="records"),
        "correlation": df[numeric_cols].corr().round(4).fillna(0).to_dict()
        if len(numeric_cols) >= 2
        else {},
    }
    return summary


def preprocess_for_supervised(df, target_column, test_size=0.2, random_state=42):
    """Preprocess dataset for classification/regression tasks."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    work_df = df.copy()
    feature_cols = [c for c in work_df.columns if c != target_column]
    X = work_df[feature_cols]
    y = work_df[target_column]

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    from sklearn.pipeline import Pipeline

    transformers = []
    if numeric_features:
        numeric_pipeline = SimpleImputer(strategy="median")
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_pipe, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    X_processed = preprocessor.fit_transform(X)

    target_is_numeric = pd.api.types.is_numeric_dtype(y)
    label_encoder = None
    if not target_is_numeric or y.nunique() <= 20:
        label_encoder = LabelEncoder()
        y_processed = label_encoder.fit_transform(y.astype(str))
        task_type = "classification"
    else:
        y_processed = y.astype(float).values
        task_type = "regression"

    split_kwargs = {
        "test_size": test_size,
        "random_state": random_state,
    }
    if task_type == "classification" and len(np.unique(y_processed)) > 1:
        split_kwargs["stratify"] = y_processed
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, **split_kwargs)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    info = {
        "task_type": task_type,
        "feature_count": int(X_processed.shape[1]),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "target_classes": label_encoder.classes_.tolist() if label_encoder else None,
    }

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "info": info,
    }


def preprocess_for_clustering(df, n_features=None):
    """Preprocess dataset for unsupervised clustering."""
    work_df = df.copy()
    numeric_cols = work_df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("Clustering requires at least one numeric column.")

    if n_features:
        numeric_cols = numeric_cols[:n_features]

    X = work_df[numeric_cols].copy()
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return {
        "X": X_scaled,
        "columns": numeric_cols,
        "imputer": imputer,
        "scaler": scaler,
        "info": {"samples": int(len(X_scaled)), "features": len(numeric_cols)},
    }


def safe_json(obj):
    """Convert numpy types to JSON-serializable Python types."""
    return json.loads(
        json.dumps(obj, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    )
