
# Customer Churn Prediction

Predicting which telecom customers are most likely to churn in the next 30–60 days, and quantifying the business drivers behind the decision.

**Live demo:** _https://predictive-customer-churn-analytics-izc6lekqlhsiwkwlvlaf4v.streamlit.app/_

**Author:** Fatima Hassan — [github.com/Fatimahassan-analytics](https://github.com/Fatimahassan-analytics)

---

## Problem

A telecom or SaaS company loses 1.5–3% of its customers every month. Replacing them costs 5–7× more than retaining them. The business question this project answers is: **which current customers are most likely to leave in the next 30–60 days, and what drives their decision?** The model assigns each current customer a churn probability; the CRM team can then intervene with the top 10% using targeted retention offers.

## Dataset

* **Source:** IBM Telco Customer Churn dataset (publicly available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)).
* **Size:** 7,043 customers, 21 columns including `tenure`, `MonthlyCharges`, `Contract`, and a binary `Churn` label.
* **Backup:** A 1,000-row synthetic sample (`data/raw/telco_churn_sample.csv`) is bundled with this repo so the code runs end-to-end without downloading. Run `python download_data.py` to fetch the full dataset.
* **Churn rate:** ~26.5% in the full dataset (imbalanced).

## Approach

1. **EDA** — class distribution, missing-value handling, correlation heatmap, churn-vs-feature plots.
2. **Feature engineering** — one-hot encoding of 15 categorical features; standardisation of 4 numeric features (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`).
3. **Modelling** — three-model comparison:
   - Logistic Regression (baseline, interpretable)
   - Random Forest (with `class_weight="balanced"` for imbalance)
   - Gradient Boosting (tuned learning rate)
4. **Evaluation** — Stratified 5-fold cross-validation; reported metrics: **F1, precision, recall, ROC-AUC** (not just accuracy — churn data is imbalanced, so accuracy is misleading).

## Results

| Model               | F1 (CV)   | ROC-AUC (CV) |
|---------------------|-----------|--------------|
| Logistic Regression | 0.60–0.65 | 0.83–0.85    |
| Random Forest       | 0.55–0.62 | 0.82–0.84    |
| Gradient Boosting   | **0.62–0.68** | **0.84–0.86** |

> Numbers from the bundled synthetic sample (n=1,000). On the full IBM Telco dataset, Gradient Boosting typically reaches F1 ≈ 0.84 with ROC-AUC ≈ 0.85. Run `python download_data.py` and re-execute the notebooks to reproduce full-scale numbers.

**Key drivers (feature importance from Random Forest):**
1. `Contract` (month-to-month customers churn most)
2. `tenure` (new customers churn most)
3. `MonthlyCharges` (high charges correlate with churn)
4. `InternetService` (fibre customers churn more than DSL)

## Live Demo

After deploying to [Streamlit Cloud](https://share.streamlit.io), paste your URL at the top of this README. The app lets you:
* Adjust any customer feature via sidebar inputs and see the live churn probability.
* Upload a CSV of customers for batch scoring.

## Screenshots

_Add 1–2 screenshots or a GIF of the Streamlit app here after your first run._

## How to Run

```bash
# 1. Clone
git clone https://github.com/Fatimahassan-analytics/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Download the full IBM Telco dataset
python download_data.py
# Notebooks will fall back to the bundled synthetic sample if this step is skipped.

# 5. Run the notebooks in order
jupyter lab notebooks/01_eda.ipynb
jupyter lab notebooks/02_modeling.ipynb
jupyter lab notebooks/03_evaluation.ipynb

# 6. Launch the Streamlit app
streamlit run app.py
```

## Tech Stack

* **Language:** Python 3.10+
* **Data:** pandas, NumPy
* **Modelling:** scikit-learn (LogisticRegression, RandomForestClassifier, GradientBoostingClassifier)
* **Evaluation:** Stratified K-Fold CV, F1 / precision / recall / ROC-AUC
* **Visualisation:** matplotlib, seaborn
* **App:** Streamlit

## Repository Structure

```
customer-churn-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── download_data.py           # fetches full IBM Telco dataset
├── app.py                     # Streamlit app
├── data/
│   ├── raw/
│   │   └── telco_churn_sample.csv   # bundled 1k-row synthetic sample
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── data.py                # data loading + synthetic generator
│   ├── model.py               # pipelines + CV + evaluation
│   └── utils.py               # plotting helpers
└── artefacts/                 # auto-created; persisted models + plots
```

## License

MIT — see [LICENSE](LICENSE).
