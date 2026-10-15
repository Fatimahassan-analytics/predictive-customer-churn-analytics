"""Customer Churn Prediction — data loading and sample generation.

This module provides:
- load_data(): load the IBM Telco dataset (or the bundled sample if the full
  dataset is not present in data/raw/).
- generate_sample_data(): create a small synthetic churn dataset that mirrors
  the schema of the IBM Telco dataset, so the notebooks run end-to-end
  without requiring an external download.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SAMPLE_PATH = RAW_DIR / "telco_churn_sample.csv"
FULL_PATH = RAW_DIR / "telco_churn.csv"

RANDOM_SEED = 42

CHURN_SCHEMA = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]


def generate_sample_data(n_rows: int = 1000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a synthetic churn dataset mirroring the IBM Telco schema.

    The relationships are deliberately stylised so a model can learn them:
    - month-to-month contracts churn more often than one/two-year contracts
    - higher MonthlyCharges increase churn probability
    - longer tenure decreases churn probability
    - fibre InternetService churns more than DSL or none
    """
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], size=n_rows)
    senior = rng.choice([0, 1], size=n_rows, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n_rows, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n_rows, p=[0.30, 0.70])
    tenure = rng.integers(0, 73, size=n_rows)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_rows, p=[0.55, 0.21, 0.24],
    )
    internet = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n_rows, p=[0.34, 0.44, 0.22]
    )
    paperless = rng.choice(["Yes", "No"], size=n_rows, p=[0.59, 0.41])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)",
         "Credit card (automatic)"],
        size=n_rows, p=[0.34, 0.23, 0.22, 0.21],
    )
    phone = rng.choice(["Yes", "No"], size=n_rows, p=[0.90, 0.10])
    monthly = rng.normal(64.8, 30.0, size=n_rows).clip(min=18.25, max=118.75)

    # linear predictor for churn
    eta = (
        -2.5
        + 0.9 * (contract == "Month-to-month")
        - 0.6 * (contract == "Two year")
        + 0.02 * monthly
        - 0.03 * tenure
        + 0.5 * (internet == "Fiber optic")
        - 0.2 * (internet == "DSL")
        + 0.3 * (payment == "Electronic check")
        + 0.2 * senior
    )
    p = 1 / (1 + np.exp(-eta))
    churn = rng.binomial(1, p)
    churn_label = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame({
        "customerID": [f"CUST-{i:05d}" for i in range(n_rows)],
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], size=n_rows),
        "InternetService": internet,
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], size=n_rows),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], size=n_rows),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], size=n_rows),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], size=n_rows),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], size=n_rows),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], size=n_rows),
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly.round(2),
        "TotalCharges": (tenure * monthly).round(2),
        "Churn": churn_label,
    })
    return df


def load_data() -> pd.DataFrame:
    """Load churn data: prefer full Kaggle CSV, fall back to bundled sample."""
    if FULL_PATH.exists():
        df = pd.read_csv(FULL_PATH)
    elif SAMPLE_PATH.exists():
        df = pd.read_csv(SAMPLE_PATH)
    else:
        df = generate_sample_data()
        df.to_csv(SAMPLE_PATH, index=False)
    return df


def save_processed(df: pd.DataFrame, name: str = "processed.csv") -> Path:
    """Save a processed dataframe to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / name
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    # Generate the bundled sample when run as a script
    df = generate_sample_data()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAMPLE_PATH, index=False)
    print(f"Wrote sample dataset: {SAMPLE_PATH} ({len(df)} rows)")
    print(f"Churn rate: {(df['Churn']=='Yes').mean():.2%}")
    print(df.head())
