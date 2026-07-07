"""
Evaluate trained machine‑learning models on the credit default dataset.

This script loads trained models from `models/` and computes evaluation metrics
(accuracy, precision, recall, F1‑score, ROC–AUC).  It also plots confusion
matrices and ROC curves, saving figures to the `visuals/` directory.  A
Markdown report is generated summarising results.
"""

import logging
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

logger.remove()
logger.add(lambda msg: logging.getLogger("evaluate_model").info(msg), format="{message}")


def load_models(model_dir: str):
    models = {}
    for model_name in ["logistic_regression.pkl", "random_forest.pkl", "xgboost.pkl"]:
        path = Path(model_dir) / model_name
        if path.exists():
            models[model_name.split(".")[0]] = joblib.load(path)
    return models


def evaluate_models(models: dict, X: pd.DataFrame, y: pd.Series, visuals_dir: str, report_path: str) -> None:
    Path(visuals_dir).mkdir(parents=True, exist_ok=True)
    metrics = []
    for name, model in models.items():
        logger.info(f"Evaluating {name}")
        probas = model.predict_proba(X)[:, 1]
        preds = (probas >= 0.5).astype(int)
        metrics.append({
            "model": name,
            "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds),
            "recall": recall_score(y, preds),
            "f1": f1_score(y, preds),
            "roc_auc": roc_auc_score(y, probas),
        })
        # Confusion matrix
        cm = confusion_matrix(y, preds)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix – {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(Path(visuals_dir) / f"confusion_matrix_{name}.png")
        plt.close()

        # ROC curve
        fpr, tpr, _ = roc_curve(y, probas)
        plt.figure(figsize=(4, 3))
        plt.plot(fpr, tpr, label=f"ROC – {name}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve – {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(visuals_dir) / f"roc_curve_{name}.png")
        plt.close()

    # Save metrics report
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(Path(visuals_dir) / "model_metrics.csv", index=False)

    # Write markdown summary
    lines = ["# Model Evaluation Report", "", "| Model | Accuracy | Precision | Recall | F1‑Score | ROC–AUC |", "|---|---|---|---|---|---|"]
    for row in metrics:
        lines.append(f"| {row['model']} | {row['accuracy']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['roc_auc']:.3f} |")
    Path(report_path).write_text("\n".join(lines))
    logger.info("Evaluation completed and report saved to %s", report_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate credit default models")
    parser.add_argument("--model_dir", "-m", required=True, help="Directory containing trained model files")
    parser.add_argument("--data", "-d", required=True, help="CSV file with engineered features and target")
    parser.add_argument("--report", "-r", required=True, help="Path to save the markdown report")
    parser.add_argument("--visuals", "-v", default="visuals/", help="Directory to save figures")
    args = parser.parse_args()

    logger.info("Loading data from %s", args.data)
    df = pd.read_csv(args.data)
    X = df.drop(columns=["DEFAULT", "ID"])
    y = df["DEFAULT"]

    models = load_models(args.model_dir)
    if not models:
        logger.error("No models found in %s", args.model_dir)
        return
    evaluate_models(models, X, y, args.visuals, args.report)


if __name__ == "__main__":
    main()