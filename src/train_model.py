"""
Train machine‑learning models on the credit default dataset.

This script trains a baseline logistic regression model and advanced models
(Random Forest and XGBoost).  It saves the trained models into the `models/`
directory.  The script prints basic accuracy for the baseline model and uses
default hyper‑parameters for the advanced models.  Further tuning can be
conducted using cross‑validation.
"""

import logging
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
from loguru import logger

# Configure logging
logger.remove()
logger.add(lambda msg: logging.getLogger("train_model").info(msg), format="{message}")


def train_models(df: pd.DataFrame, model_dir: str) -> None:
    """Train baseline and advanced models and persist them.

    Args:
        df: DataFrame containing features and the target column `DEFAULT`.
        model_dir: Directory to save model files.
    """
    # Separate features and target; drop ID since it is an identifier
    X = df.drop(columns=["DEFAULT", "ID"])
    y = df["DEFAULT"]

    # Stratified split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    # Baseline model: Logistic Regression
    logger.info("Training Logistic Regression baseline model")
    lr = LogisticRegression(max_iter=1000, solver="liblinear")
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Baseline accuracy: {acc:.3f}")
    joblib.dump(lr, model_dir_path / "logistic_regression.pkl")

    # Random Forest model
    logger.info("Training Random Forest model")
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf, model_dir_path / "random_forest.pkl")

    # XGBoost model
    logger.info("Training XGBoost model")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )
    xgb_clf.fit(X_train, y_train)
    joblib.dump(xgb_clf, model_dir_path / "xgboost.pkl")

    logger.info("Models saved to %s", model_dir)


def main() -> None:
    """Command-line entry point for model training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train credit default models")
    parser.add_argument("--input", "-i", required=True, help="CSV file with engineered features and target")
    parser.add_argument("--model_output", "-o", required=True, help="Directory to save trained models")
    args = parser.parse_args()

    logger.info("Loading features from %s", args.input)
    df = pd.read_csv(args.input)
    logger.info("Data shape %s", df.shape)

    train_models(df, args.model_output)


if __name__ == "__main__":
    main()