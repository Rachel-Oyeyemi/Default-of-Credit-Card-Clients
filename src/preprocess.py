"""
Preprocess the Default of Credit Card Clients dataset.

This script reads the raw dataset (Excel or CSV), performs basic cleaning,
renames columns for clarity, handles missing values and duplicates, and writes
out a processed CSV for downstream modelling.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

# Configure logging via loguru to integrate with standard logging
logger.remove()
logger.add(lambda msg: logging.getLogger("preprocess").info(msg), format="{message}")

# Expected original columns and mapping for renaming
ORIGINAL_COLUMNS = [
    "ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "default payment next month"
]

RENAMED_COLUMNS = {
    "default payment next month": "DEFAULT",
    "SEX": "GENDER",
}


def load_dataset(path: str) -> pd.DataFrame:
    """Load the dataset from an Excel or CSV file.

    The original file from UCI is an `.xls` file.  This function attempts to
    read it using pandas.  If the required engine for Excel files is unavailable,
    users should convert the file to CSV manually or using the provided download script.

    Args:
        path: File path to load.

    Returns:
        DataFrame containing the dataset.

    Raises:
        ValueError: If the file extension is unsupported or the engine is missing.
    """
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in {".xls", ".xlsx"}:
        try:
            # Skip the first row as it contains duplicate header information in UCI file
            df = pd.read_excel(path, header=1)
        except ImportError as e:
            raise ValueError(
                "Reading Excel files requires the `xlrd` or `openpyxl` package. "
                "Install the package or convert the file to CSV."
            ) from e
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Perform initial cleaning on the dataset.

    Steps:
    - Rename columns for clarity.
    - Drop duplicate rows.
    - Handle missing values by imputing column means.

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    # Rename columns if present
    df = df.rename(columns=RENAMED_COLUMNS)

    # Drop duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count:
        logger.info(f"Dropping {duplicate_count} duplicate rows")
        df = df.drop_duplicates()

    # Fill missing values with column means
    missing_total = df.isnull().sum().sum()
    if missing_total:
        logger.info(f"Found {missing_total} missing values; filling with column means")
        df = df.fillna(df.mean())

    return df


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    """Save the processed DataFrame to CSV.

    Args:
        df: DataFrame to save.
        output_path: Destination CSV path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")


def main() -> None:
    """Command-line entry point for preprocessing."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess the credit default dataset")
    parser.add_argument("--input", "-i", required=True, help="Path to the raw dataset (xls or csv)")
    parser.add_argument("--output", "-o", required=True, help="Path to save the processed CSV")
    args = parser.parse_args()

    logger.info("Loading dataset from %s", args.input)
    df_raw = load_dataset(args.input)
    logger.info("Dataset loaded with shape %s", df_raw.shape)

    df_clean = preprocess(df_raw)
    logger.info("Dataset processed; final shape %s", df_clean.shape)

    save_dataset(df_clean, args.output)


if __name__ == "__main__":
    main()