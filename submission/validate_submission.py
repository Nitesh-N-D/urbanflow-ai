"""Validate Gridlock submission CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def validate_submission(csv_path: str | Path) -> bool:
    """Validate required submission columns.

    Args:
        csv_path: Submission CSV path.

    Returns:
        True when valid.

    Raises:
        FileNotFoundError: If CSV is missing.
        ValueError: If schema is invalid.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        columns = set(reader.fieldnames or [])
    required = {"image_id", "predicted_label", "confidence"}
    if not required.issubset(columns):
        raise ValueError(f"Submission missing columns: {required - columns}")
    for row in rows:
        try:
            confidence = float(row["confidence"])
        except (TypeError, ValueError) as exc:
            raise ValueError("confidence must be numeric") from exc
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
    return True


def main() -> None:
    """Run CSV validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    print("valid" if validate_submission(args.csv) else "invalid")


if __name__ == "__main__":
    main()
