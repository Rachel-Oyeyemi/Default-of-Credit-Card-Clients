"""
Make predictions using trained models.

This script loads a persisted model and applies it to new data supplied via a CSV
file.  The output is a CSV containing the predicted probabilities and labels.
"""

import joblib
import pandas as pd
from pathlib import Path
import logging
from loguru import logger

logger.remove()
logger.add(lambda msg: logging.getLogger("predict").info(msg), format="{message}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Predict default probabilities for new data")
    parser.add_argument("--model", required=True, help="Path to the trained model (pkl)")
    parser.add_argument("--input", required=True, help="CSV file with features")
    parser.add_argument("--output", required=True, help="Output CSV file for predictions")
    args = parser.parse_args()

    logger.info("Loading model from %s", args.model)
    model = joblib.load(args.model)

    df = pd.read_csv(args.input)
    probabilities = model.predict_proba(df)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    result = df.copy()
    result["prob_default"] = probabilities
    result["prediction"] = predictions

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    logger.info("Predictions saved to %s", args.output)


if __name__ == "__main__":
    main()