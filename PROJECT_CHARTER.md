# Project Charter – Default of Credit Card Clients

## Business Problem

Credit‑card default can lead to significant financial losses for issuing banks and negatively impact customers’ credit scores.  The bank needs a reliable way to estimate the likelihood that a client will default on their credit‑card payment in the following month.  Effective prediction enables proactive measures such as adjusting credit limits, flagging accounts for closer monitoring or offering tailored financial counselling.

## Project Objectives

1. **Build a predictive model** that estimates the probability of default for each client using historical payment and demographic data.
2. **Benchmark baseline and advanced algorithms** and select the most effective model based on accuracy, recall, precision, F1‑score and ROC–AUC.
3. **Generate business insights** by analysing feature importances and relationships between variables and default risk.
4. **Deliver an interactive Streamlit application** for stakeholders to explore the model, perform predictions on new data and visualise key findings.

## Stakeholders

- **Risk Management Team:**  Responsible for monitoring credit risk and setting policies.
- **Data Science & Analytics Team:**  Designs, trains and maintains the predictive models.
- **IT / Engineering:**  Supports data pipelines and deploys the Streamlit application.
- **Executive Management:**  Uses insights to inform strategic decisions on credit products.

## Success Metrics

- **Model Performance:**  Achieve ROC–AUC ≥ 0.80 and maintain balanced precision and recall (e.g., F1‑score ≥ 0.65).
- **Business Adoption:**  Stakeholders integrate the model into the credit‑risk workflow.
- **Interpretability:**  Provide clear explanations for predictions and maintain compliance with regulatory requirements.

## Expected Business Impact

- **Reduced default losses:**  Early identification of high‑risk clients allows for timely interventions.
- **Improved customer segmentation:**  Better understanding of customer behaviour informs marketing and retention strategies.
- **Data‑driven policy adjustments:**  Evidence‑based insights guide credit limit and interest rate decisions.

## Technical Architecture

1. **Data Storage:**  Raw dataset stored in `data/raw`; processed and engineered datasets stored in `data/processed`.
2. **Processing Pipeline:**  Python scripts (`download_data.py`, `preprocess.py`, `feature_engineering.py`) handle data ingestion, cleaning and feature engineering.  Logging ensures traceability and error handling.
3. **Model Training:**  Models are trained via `train_model.py`, saving artefacts to `models/` and evaluation metrics to `MODEL_EVALUATION.md`.
4. **Model Evaluation:**  `evaluate_model.py` computes metrics and plots; results are stored in `visuals/` and summarised in `MODEL_COMPARISON.md`.
5. **Application:**  A Streamlit app in `app/app.py` loads the trained model and displays prediction forms, performance charts and business insights.
6. **Version Control & Documentation:**  The entire project is hosted on GitHub with notebooks, markdown reports and a presentation.

## End‑to‑End Workflow

1. **Acquire** the dataset and save it to the `data/raw` directory.
2. **Preprocess** the data: load, clean, rename columns, handle missing values and duplicates.
3. **Engineer Features:**  Compute additional behavioural metrics (e.g., average bill amount, payment ratios).
4. **Split** the data into training and test sets using stratified sampling to preserve class proportions.
5. **Train Baseline Model** (Logistic Regression) and evaluate.
6. **Train Advanced Models** (Random Forest, XGBoost) and compare performance.
7. **Select** the best model and interpret feature importances.
8. **Generate Reports** including EDA, modelling results and business recommendations.
9. **Deploy** the Streamlit app for interactive exploration and prediction.
