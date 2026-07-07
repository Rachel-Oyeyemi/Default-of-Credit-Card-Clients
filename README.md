# Default of Credit Card Clients

## Project Overview

This project analyses the **Default of Credit Card Clients** dataset from the UCI Machine Learning Repository.  The dataset contains information about Taiwanese credit‑card customers in 2005.  Each record includes demographic data, past payment history and financial indicators; the target variable indicates whether the client defaulted on their credit‑card payment in the following month.

The goal is to build a machine‑learning pipeline that predicts the probability of default.  In addition to modelling, the project generates exploratory analyses, business insights and a Streamlit application to demonstrate how the model can be used in practice.

## Business Problem

Credit‑card issuers need to anticipate the risk of default so they can make informed decisions about credit limits, interest rates and debt‑collection strategies.  By predicting whether a customer is likely to default next month, the bank can proactively adjust credit terms or target early interventions, reducing financial losses.  The project frames the problem as a **binary classification**: predict `1` if the client will default, or `0` otherwise.

## Dataset Source

The dataset originates from the **UCI Machine Learning Repository**.  It contains **30 000** observations and **23** predictive features (plus an `ID` column and the binary target `default payment next month`).  According to the dataset description, there are no missing values【382343060160666†L35-L64】 and the variables cover credit limit, demographic attributes (age, gender, education, marriage), repayment history (`PAY_0` – `PAY_6`), bill statements (`BILL_AMT1` – `BILL_AMT6`) and past payments (`PAY_AMT1` – `PAY_AMT6`)【382343060160666†L95-L112】.

## Methodology

1. **Data Acquisition** – Download the Excel dataset from the UCI repository or Kaggle.  A script (`src/download_data.py`) is provided to automate this step.
2. **Preprocessing** – Load the data into pandas, rename columns, check for missing values and duplicates, and prepare the features and target (`src/preprocess.py`).  Because the dataset is purely numeric, the primary cleaning steps include type conversions and optional scaling.
3. **Feature Engineering** – Generate additional features such as average bill amount, ratio of payments to bills, and delayed payment indicators (`src/feature_engineering.py`).
4. **Modelling** – Train baseline and advanced classification models.  The baseline model uses **Logistic Regression**, while advanced models include **Random Forest** and **Gradient Boosting** (e.g., **XGBoost**).  Modelling code is contained in `src/train_model.py`.
5. **Evaluation** – Evaluate models using accuracy, precision, recall, F1‑score and ROC–AUC.  Confusion matrices and ROC curves help interpret model performance (`src/evaluate_model.py`).
6. **Streamlit App** – An interactive application allows users to input customer attributes and obtain a default‑probability prediction.  It also displays model performance metrics, key visualisations and business recommendations.

## EDA

The exploratory data analysis examines distributions, correlations and potential outliers for each variable.  Because the target variable (default) is imbalanced, we pay particular attention to class distribution and evaluate whether resampling techniques (e.g., SMOTE) are necessary.  EDA results and plots are summarised in `notebooks/01_data_exploration.ipynb` and `visuals/`.

## Feature Engineering

Key engineered features include:

- **Average Bill Amount** – mean of `BILL_AMT1`–`BILL_AMT6`.
- **Average Payment Amount** – mean of `PAY_AMT1`–`PAY_AMT6`.
- **Payment to Bill Ratio** – ratio of total payments to total bills.
- **Delayed Payment Count** – count of months with positive values in `PAY_0`–`PAY_6`, indicating payment delay.

These features help capture customers’ financial behaviour beyond raw statement amounts.

## Modelling

- **Baseline:**  Logistic Regression provides a simple, interpretable benchmark.  It is trained on a stratified train/test split (e.g., 70/30 split).  Typical performance on this dataset yields **~78 % accuracy** and **ROC–AUC around 0.75**, though results may vary depending on preprocessing and random seed.
- **Advanced Models:**  Random Forests and Gradient Boosting (XGBoost) often achieve higher recall and AUC scores.  Hyper‑parameter tuning is performed with cross‑validation to optimise model parameters.

## Results

Model evaluation emphasises both recall (identifying true defaulters) and precision (avoiding false alarms).  Three models were trained and compared:

| Model | Accuracy | Precision | Recall | F1‑Score | ROC–AUC |
|------:|---------:|----------:|------:|---------:|--------:|
| **Logistic Regression** | 0.779 | 0.000 | 0.000 | 0.000 | 0.653 |
| **Random Forest** | 0.944 | 0.927 | 0.809 | 0.864 | 0.961 |
| **XGBoost** | 0.878 | 0.844 | 0.550 | 0.666 | 0.911 |

The logistic regression baseline struggled to detect defaulters at the default 0.5 threshold, highlighting the challenge posed by class imbalance.  Tree‑based models performed substantially better: the Random Forest achieved the highest accuracy (94 %) and recall (>80 %), while XGBoost offered competitive performance.  See `MODEL_COMPARISON.md` and `MODEL_EVALUATION.md` for detailed analysis and confusion matrices.

## Business Impact

Accurately predicting default risk allows credit‑card issuers to:

- **Reduce Credit Losses:**  By identifying high‑risk customers early, banks can lower credit limits or adjust interest rates accordingly.
- **Improve Customer Management:**  Targeted interventions (e.g., payment reminders or restructuring offers) can be directed at at‑risk clients.
- **Support Regulatory Compliance:**  Transparent models (e.g., Logistic Regression) support explainability requirements.

## How to Run

1. **Clone the repository** and install requirements:

   ```bash
   git clone https://github.com/Rachel-Oyeyemi/Default-of-Credit-Card-Clients.git
   cd Default-of-Credit-Card-Clients
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Download the dataset** (e.g., from Kaggle) and save it to `data/raw/default_of_credit_card_clients.xls`.  The original file is an old‑format Excel spreadsheet (`.xls`); convert it to CSV using LibreOffice or another tool (e.g.,
   `libreoffice --headless --convert-to csv 'default of credit card clients.xls'`).  Alternatively, use the provided script `src/download_data.py` to fetch and convert the dataset automatically.

3. **Run preprocessing and feature engineering**:

   ```bash
   # Preprocess raw data (CSV or XLS)
   python src/preprocess.py --input data/raw/default_of_credit_card_clients.csv --output data/processed/credit_default_processed.csv
   # Create engineered features
   python src/feature_engineering.py --input data/processed/credit_default_processed.csv --output data/processed/credit_default_engineered.csv
   ```

4. **Train and evaluate models**:

   ```bash
   python src/train_model.py --input data/processed/credit_default_engineered.csv --model_output models/
   python src/evaluate_model.py --model_dir models/ --data data/processed/credit_default_engineered.csv --report MODEL_EVALUATION.md
   ```

5. **Launch the Streamlit app**:

   ```bash
   streamlit run app/app.py
   ```

## Future Improvements

- **Hyper‑parameter tuning** using grid search or Bayesian optimisation.
- **Handling class imbalance** via resampling or cost‑sensitive learning.
- **Feature selection** and dimensionality reduction to simplify models.
- **Model explainability** using SHAP values to interpret predictions.
- **Deployment** via containerisation (Docker) or cloud services (e.g., AWS, Azure).
