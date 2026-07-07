"""
Feature engineering for the Default of Credit Card Clients dataset.

This script computes additional features that capture behavioural patterns in
credit use and payment histories.  The engineered dataset is saved as CSV.
"""

import logging
from pathlib import Path
import pandas as pd
from loguru import logger

# Configure logging
logger.remove()
logger.add(lambda msg: logging.getLogger("feature_engineering").info(msg), format="{message}")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from the existing columns.

    Features include:
    - `avg_bill_amt`: Average of BILL_AMT1–BILL_AMT6.
    - `avg_pay_amt`: Average of PAY_AMT1–PAY_AMT6.
    - `total_bill_amt`: Sum of BILL_AMT1–BILL_AMT6.
    - `total_pay_amt`: Sum of PAY_AMT1–PAY_AMT6.
    - `payment_ratio`: total_pay_amt / (total_bill_amt + 1e-6).
    - `delayed_pay_count`: Count of months where PAY_X > 0.

    Args:
        df: Preprocessed DataFrame.

    Returns:
        DataFrame with new features appended.
    """
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    pay_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    status_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

    # Copy to avoid modifying original DataFrame
    df = df.copy()

    df["avg_bill_amt"] = df[bill_cols].mean(axis=1)
    df["avg_pay_amt"] = df[pay_cols].mean(axis=1)
    df["total_bill_amt"] = df[bill_cols].sum(axis=1)
    df["total_pay_amt"] = df[pay_cols].sum(axis=1)
    df["payment_ratio"] = df["total_pay_amt"] / (df["total_bill_amt"] + 1e-6)

    # Count months with delayed payment (positive values in PAY_* indicate delay)
    df["delayed_pay_count"] = (df[status_cols] > 0).sum(axis=1)

    return df


def main():
    """Command-line entry point for feature engineering."""
    import argparse

    parser = argparse.ArgumentParser(description="Perform feature engineering on the credit default dataset")
    parser.add_argument("--input", "-i", required=True, help="Path to the processed CSV file")
    parser.add_argument("--output", "-o", required=True, help="Path to save the engineered CSV file")
    args = parser.parse_args()

    logger.info("Loading processed data from %s", args.input)
    df = pd.read_csv(args.input)
    logger.info("Data loaded with shape %s", df.shape)

    df_engineered = engineer_features(df)
    logger.info("Feature engineering complete; new shape %s", df_engineered.shape)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_engineered.to_csv(args.output, index=False)
    logger.info("Engineered data saved to %s", args.output)


if __name__ == "__main__":
    main()