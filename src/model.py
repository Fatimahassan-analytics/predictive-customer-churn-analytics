"""Customer Churn Prediction — model definitions and training utilities."""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42

# Categorical vs numerical columns (matches generate_sample_data schema)
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
TARGET_COL = "Churn"


def build_preprocessor() -> ColumnTransformer:
    """Build the column transformer: OHE for categoricals, scale numerics."""
    cat_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    num_scaler = StandardScaler()
    return ColumnTransformer(
        transformers=[
            ("cat", cat_enc, CATEGORICAL_COLS),
            ("num", num_scaler, NUMERIC_COLS),
        ],
        remainder="drop",
    )


def build_models() -> dict:
    """Return the three candidate model pipelines keyed by name."""
    pre = build_preprocessor()
    return {
        "logistic_regression": Pipeline([
            ("pre", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        ]),
        "random_forest": Pipeline([
            ("pre", build_preprocessor()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=None,
                class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)),
        ]),
        "gradient_boosting": Pipeline([
            ("pre", build_preprocessor()),
            ("clf", GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.05,
                max_depth=3, random_state=RANDOM_SEED)),
        ]),
    }


def split_data(df: pd.DataFrame, test_size: float = 0.20, seed: int = RANDOM_SEED):
    """Stratified train/test split. Drops customerID and parses TotalCharges."""
    X = df.drop(columns=[TARGET_COL, "customerID"], errors="ignore").copy()
    y = (df[TARGET_COL] == "Yes").astype(int)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def cross_validate_model(pipeline, X, y, cv_folds: int = 5) -> dict:
    """Stratified k-fold CV returning F1, ROC-AUC, precision, recall (mean ± std)."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    scores = {
        "f1": cross_val_score(pipeline, X, y, cv=skf, scoring="f1", n_jobs=-1),
        "roc_auc": cross_val_score(pipeline, X, y, cv=skf, scoring="roc_auc", n_jobs=-1),
        "precision": cross_val_score(pipeline, X, y, cv=skf, scoring="precision", n_jobs=-1),
        "recall": cross_val_score(pipeline, X, y, cv=skf, scoring="recall", n_jobs=-1),
    }
    return {k: {"mean": float(v.mean()), "std": float(v.std())} for k, v in scores.items()}


def evaluate_model(pipeline, X_test, y_test) -> dict:
    """Evaluate a fitted pipeline on held-out test data."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred
    return {
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }


def save_model(pipeline, name: str, models_dir: Path) -> Path:
    """Persist a fitted pipeline to disk."""
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.joblib"
    joblib.dump(pipeline, path)
    return path


def load_model(name: str, models_dir: Path) -> object:
    """Load a previously saved pipeline."""
    return joblib.load(models_dir / f"{name}.joblib")
