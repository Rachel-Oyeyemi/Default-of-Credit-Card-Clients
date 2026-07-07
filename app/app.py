"""
Streamlit application for the Default of Credit Card Clients project.

This app provides an interactive interface to explore the project, visualise the data,
view model performance and make predictions using the trained Random Forest model.
Run with `streamlit run app/app.py` from the repository root.
"""
import os
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
VISUALS_DIR = BASE_DIR / "visuals"

@st.cache_resource
def load_model(model_path: Path):
    """Load a trained model from disk.  Caches the result for performance."""
    return joblib.load(model_path)

@st.cache_resource
def load_data(sample: bool = True):
    """Load a processed dataset.  If `sample` is True, loads a 500‑row sample for display.

    Returns:
        DataFrame: processed (and engineered) data
    """
    full_path = DATA_DIR / "credit_default_engineered.csv"
    df = pd.read_csv(full_path)
    if sample:
        return df.sample(n=min(500, len(df)), random_state=42)
    return df

@st.cache_resource
def load_metrics():
    """Load model evaluation metrics from CSV."""
    metrics_path = VISUALS_DIR / "model_metrics.csv"
    return pd.read_csv(metrics_path)

def navigation_menu():
    pages = {
        "Home": home_page,
        "Project Overview": overview_page,
        "Make Prediction": prediction_page,
        "Model Performance": performance_page,
        "Visualisations": visualisations_page,
        "Business Insights": insights_page,
        "About": about_page,
    }
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    pages[selection]()

def home_page():
    st.title("Default of Credit Card Clients")
    st.markdown(
        """
        **Predicting credit‑card default risk.**

        This application accompanies a data science project that analyses the UCI
        Default of Credit Card Clients dataset【382343060160666†L35-L64】.  It provides a
        high‑level overview of the business problem, exploratory data analysis,
        machine‑learning models and actionable insights.  You can also enter your own
        customer information to obtain a default probability estimate.
        """
    )
    st.image(str(VISUALS_DIR / "target_distribution.png"), caption="Target Variable Distribution")

def overview_page():
    st.header("Project Overview")
    st.markdown(
        """
        **Business problem:**  Predict whether a credit‑card customer will default on their
        payment in the next month.  Accurate predictions allow banks to manage
        risk, adjust credit limits and engage in proactive customer outreach.

        **Dataset:**  30 000 historical customer records with 23 explanatory variables and
        one binary target【382343060160666†L35-L64】.  Variables include credit limit, demographic
        attributes, repayment status in previous months, bill amounts and past
        payments【382343060160666†L95-L112】.  There are no missing values and the default rate is
        approximately 22 %.

        **Modelling approach:**  A baseline logistic regression model establishes a
        reference point.  Advanced models (Random Forest and XGBoost) capture
        non‑linear relationships and class imbalance.  The Random Forest model
        achieved the best overall performance (accuracy 94 %, ROC–AUC 0.96).
        """
    )
    st.subheader("Sample of processed data")
    df_sample = load_data(sample=True)
    st.dataframe(df_sample.head(50))

def prediction_page():
    st.header("Make a Prediction")
    st.markdown(
        """
        Input customer attributes below to estimate the probability of default using
        the trained Random Forest model.  All numeric inputs should correspond to the
        latest known values for the customer.
        """
    )
    # Feature inputs
    credit_limit = st.number_input("Credit limit (LIMIT_BAL)", value=50000, step=1000)
    age = st.number_input("Age", value=35, step=1)
    education = st.selectbox("Education level", options=[1, 2, 3, 4], format_func=lambda x: {1: "Graduate school", 2: "University", 3: "High school", 4: "Others"}.get(x, str(x)))
    marriage = st.selectbox("Marital status", options=[1, 2, 3], format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}.get(x, str(x)))
    pay_status = {}
    for idx, month in enumerate(["Sep", "Aug", "Jul", "Jun", "May", "Apr"], start=0):
        pay_status[f"PAY_{idx if idx == 0 else idx*2}"] = st.selectbox(
            f"Delay in payment in {month} (PAY_{idx if idx == 0 else idx*2})", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
        )
    bill_amts = {}
    pay_amts = {}
    for i in range(1, 7):
        bill_amts[f"BILL_AMT{i}"] = st.number_input(f"Bill amount month {i}", value=0.0, step=1000.0)
        pay_amts[f"PAY_AMT{i}"] = st.number_input(f"Payment amount month {i}", value=0.0, step=1000.0)
    if st.button("Predict"):
        # Assemble feature vector matching training columns
        input_dict = {
            "LIMIT_BAL": credit_limit,
            "SEX": 1,  # default placeholder (not asked explicitly)
            "EDUCATION": education,
            "MARRIAGE": marriage,
            "AGE": age,
        }
        input_dict.update(pay_status)
        input_dict.update(bill_amts)
        input_dict.update(pay_amts)
        # Compute engineered features on the fly
        import pandas as pd
        df_input = pd.DataFrame([input_dict])
        # Feature engineering
        bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
        pay_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
        status_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
        df_input["avg_bill_amt"] = df_input[bill_cols].mean(axis=1)
        df_input["avg_pay_amt"] = df_input[pay_cols].mean(axis=1)
        df_input["total_bill_amt"] = df_input[bill_cols].sum(axis=1)
        df_input["total_pay_amt"] = df_input[pay_cols].sum(axis=1)
        df_input["payment_ratio"] = df_input["total_pay_amt"] / (df_input["total_bill_amt"] + 1e-6)
        df_input["delayed_pay_count"] = (df_input[status_cols] > 0).sum(axis=1)
        # Align columns with model training
        train_columns = [
            'LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
            'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
            'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6',
            'avg_bill_amt','avg_pay_amt','total_bill_amt','total_pay_amt','payment_ratio','delayed_pay_count'
        ]
        df_input = df_input.reindex(columns=train_columns, fill_value=0)
        model = load_model(MODELS_DIR / "random_forest.pkl")
        prob = model.predict_proba(df_input)[:, 1][0]
        pred = int(prob >= 0.5)
        st.success(f"Predicted probability of default: {prob:.2%}")
        st.write(f"Predicted class: {'Default' if pred == 1 else 'Non‑default'}")

def performance_page():
    st.header("Model Performance")
    st.markdown(
        """
        This page summarises the evaluation metrics for each model.  Accuracy alone can be
        misleading when classes are imbalanced; therefore precision, recall, F1 and
        ROC–AUC are reported.  Confusion matrices and ROC curves are also provided.
        """
    )
    metrics_df = load_metrics()
    st.dataframe(metrics_df)
    # Display images
    st.subheader("Confusion Matrices")
    cols = st.columns(3)
    for col, model in zip(cols, ["logistic_regression", "random_forest", "xgboost"]):
        col.image(str(VISUALS_DIR / f"confusion_matrix_{model}.png"), caption=model.replace("_", " ").title())
    st.subheader("ROC Curves")
    cols = st.columns(3)
    for col, model in zip(cols, ["logistic_regression", "random_forest", "xgboost"]):
        col.image(str(VISUALS_DIR / f"roc_curve_{model}.png"), caption=model.replace("_", " ").title())

def visualisations_page():
    st.header("Visualisations")
    st.markdown(
        """
        Exploratory data analysis revealed important patterns in the dataset.  The
        correlation heatmap shows relationships among variables, while the limit
        balance distribution illustrates the skewness of credit limits.
        """
    )
    st.image(str(VISUALS_DIR / "correlation_heatmap.png"), caption="Correlation Heatmap", use_column_width=True)
    st.image(str(VISUALS_DIR / "limit_bal_distribution.png"), caption="Distribution of Credit Limits", use_column_width=True)

def insights_page():
    st.header("Business Insights")
    # Display the text of the business recommendations
    recommendations_file = BASE_DIR / "BUSINESS_RECOMMENDATIONS.md"
    with open(recommendations_file, "r", encoding="utf-8") as f:
        st.markdown(f.read())

def about_page():
    st.header("About")
    st.markdown(
        """
        Developed by a data scientist as part of a portfolio project.  The goal is to
        demonstrate end‑to‑end proficiency in data analysis, machine learning and
        full‑stack deployment.  For more information or questions, please refer
        to the project repository on GitHub.
        """
    )

def main():
    st.set_page_config(page_title="Credit Default Risk", layout="wide")
    navigation_menu()


if __name__ == "__main__":
    main()
