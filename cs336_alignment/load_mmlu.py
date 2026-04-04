"""Load MMLU dataset from CSV files."""
import csv
from pathlib import Path
from typing import Any


def load_mmlu_split(split: str = "test") -> list[dict[str, Any]]:
    """
    Load MMLU data from CSV files.

    Args:
        split: One of "test", "val", or "dev"

    Returns:
        List of dicts with keys: subject, question, options, answer
    """
    data_dir = Path(__file__).parent.parent / "data" / "mmlu" / split
    examples = []

    for csv_file in data_dir.glob("*.csv"):
        subject = csv_file.stem.replace("_test", "").replace("_val", "").replace("_dev", "")
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue
                question = row[0]
                options = [row[1], row[2], row[3], row[4]]
                answer = row[5].strip()
                examples.append({
                    "subject": subject,
                    "question": question,
                    "options": options,
                    "answer": answer,
                })
    return examples