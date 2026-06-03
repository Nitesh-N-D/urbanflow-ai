"""Submission formatting helpers."""

from __future__ import annotations

import csv


def format_submission(rows: list[dict], output: str = "submission.csv") -> str:
    """Write a leaderboard submission CSV.

    Args:
        rows: Rows with image_id, predicted_label, confidence.
        output: Output CSV path.

    Returns:
        Output path.

    Raises:
        ValueError: If required fields are missing.
    """
    required = {"image_id", "predicted_label", "confidence"}
    for row in rows:
        if not required.issubset(row):
            raise ValueError(f"Missing submission fields: {required - set(row)}")
    with open(output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_id", "predicted_label", "confidence"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in ["image_id", "predicted_label", "confidence"]})
    return output
