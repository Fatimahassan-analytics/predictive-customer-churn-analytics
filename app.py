"""Streamlit app — Customer Churn Prediction.

Run locally:
    streamlit run app.py

Deploy:
    Push this repo to GitHub, then go to https://share.streamlit.io
    and select app.py as the entry point.
"""
from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# Ensure src/ is on the path when running from anywhere
sys.path.insert(0, str(Path(__file__).parent / "src"))
from data import load_data  # noqa: E402
from model import build_models, split_data, save_model, RANDOM_SEED  # noqa: E402

MODELS_DIR = Path(__file__).parent / "artefacts"
CACHED_MODEL_PATH = MODELS_DIR / "best_model.joblib"


@st.cache_resource(show_spinner="Training model on first run ...")
def get_or_train_model():
    """Load cached model if present, else train and cache."""
    if CACHED_MODEL_PATH.exists():
        import joblib
        return joblib.load(CACHED_MODEL_PATH)
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    models = build_models()
    best, best_f1 = None, -1
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        from sklearn.metrics import f1_score
        f1 = f1_score(y_test, pipe.predict(X_test))
        if f1 > best_f1:
            best, best_f1 = pipe, f1
    save_model(best, "best_model", MODELS_DIR)
    return best


def main():
    st.set_page_config(page_title="Customer Churn Prediction", page_icon=":bar_chart:",
                       layout="wide")
    st.title("Customer Churn Prediction")
    st.caption("Predict which customers are most likely to churn in the next 30–60 days.")

    model = get_or_train_model()

    df = load_data()
    st.sidebar.header("Enter customer features")
    input_data = collect_input(df)
    input_df = pd.DataFrame([input_data])

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Customer profile")
        st.dataframe(input_df.T, use_container_width=True)
    with col2:
        st.subheader("Churn prediction")
        proba = float(model.predict_proba(input_df)[0, 1])
        pred = "Churn" if proba >= 0.5 else "No churn"
        delta_color = "inverse" if proba >= 0.5 else "off"
        st.metric("Churn probability", f"{proba:.1%}", delta=pred, delta_color=delta_color)
        st.progress(min(int(proba * 100), 100))

    st.divider()
    st.subheader("Batch scoring")
    st.caption("Upload a CSV with the same columns as the training data to score many customers at once.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            keep_cols = [c for c in model.feature_names_in_] if hasattr(model, "feature_names_in_") else batch.columns
            # Align columns: only use those present, fill missing with defaults
            batch_aligned = batch.reindex(columns=keep_cols, fill_value=np.nan)
            probs = model.predict_proba(batch_aligned)[:, 1]
            batch["churn_probability"] = probs
            batch["predicted_churn"] = np.where(probs >= 0.5, "Yes", "No")
            st.dataframe(batch.head(20))
            st.download_button("Download scored CSV",
                               batch.to_csv(index=False).encode("utf-8"),
                               file_name="scored_customers.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Could not score uploaded file: {e}")


def collect_input(df: pd.DataFrame) -> dict:
    s = st.sidebar
    return {
        "gender": s.selectbox("Gender", ["Male", "Female"]),
        "SeniorCitizen": s.selectbox("Senior citizen", [0, 1], format_func=lambda v: "Yes" if v else "No"),
        "Partner": s.selectbox("Partner", ["Yes", "No"]),
        "Dependents": s.selectbox("Dependents", ["Yes", "No"]),
        "tenure": s.slider("Tenure (months)", 0, 72, 12),
        "PhoneService": s.selectbox("Phone service", ["Yes", "No"]),
        "MultipleLines": s.selectbox("Multiple lines", ["Yes", "No", "No phone service"]),
        "InternetService": s.selectbox("Internet service", ["DSL", "Fiber optic", "No"]),
        "OnlineSecurity": s.selectbox("Online security", ["Yes", "No", "No internet service"]),
        "OnlineBackup": s.selectbox("Online backup", ["Yes", "No", "No internet service"]),
        "DeviceProtection": s.selectbox("Device protection", ["Yes", "No", "No internet service"]),
        "TechSupport": s.selectbox("Tech support", ["Yes", "No", "No internet service"]),
        "StreamingTV": s.selectbox("Streaming TV", ["Yes", "No", "No internet service"]),
        "StreamingMovies": s.selectbox("Streaming movies", ["Yes", "No", "No internet service"]),
        "Contract": s.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
        "PaperlessBilling": s.selectbox("Paperless billing", ["Yes", "No"]),
        "PaymentMethod": s.selectbox("Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]),
        "MonthlyCharges": s.slider("Monthly charges ($)", 18.0, 120.0, 65.0, 0.5),
        "TotalCharges": s.slider("Total charges ($)", 0.0, 9000.0, 1000.0, 50.0),
    }


if __name__ == "__main__":
    main()
