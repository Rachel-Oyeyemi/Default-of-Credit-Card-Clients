"""
Download the Default of Credit Card Clients dataset.

This script provides a function to download the dataset from the UCI Machine
Learning Repository.  The downloaded file is saved into the `data/raw` directory.
Users may also manually download the dataset from Kaggle or other sources and
place it into `data/raw`.
"""

import os
import logging
import requests
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger(__name__)

# URL to the UCI dataset (Excel format)
UCI_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"


def download_dataset(output_path: str) -> None:
    """Download the dataset from the UCI repository.

    Args:
        output_path: Path where the dataset will be saved.  This should end with
            `.xls` or `.xlsx`.

    Raises:
        IOError: If the download fails or the response is not OK.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if os.path.exists(output_path):
        LOGGER.info("Dataset already exists at %s; skipping download.", output_path)
        return

    LOGGER.info("Downloading dataset from %s", UCI_DATA_URL)
    try:
        response = requests.get(UCI_DATA_URL, timeout=60)
        response.raise_for_status()
    except Exception as exc:
        LOGGER.error("Failed to download dataset: %s", exc)
        raise IOError(f"Failed to download dataset: {exc}") from exc

    with open(output_path, "wb") as f:
        f.write(response.content)
    LOGGER.info("Dataset downloaded and saved to %s", output_path)


def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Download the credit default dataset")
    parser.add_argument(
        "--output", "-o",
        default=str(Path(__file__).resolve().parent.parent / "data/raw/default_of_credit_card_clients.xls"),
        help="Output file path for the downloaded dataset",
    )
    args = parser.parse_args()
    download_dataset(args.output)


if __name__ == "__main__":
    main()