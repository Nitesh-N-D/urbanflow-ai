"""Submission tests."""

from __future__ import annotations

from pathlib import Path

from submission.format_output import format_submission
from submission.validate_submission import validate_submission


def test_submission_validation(tmp_path: Path) -> None:
    """Valid submission file passes validation."""
    output = tmp_path / "submission.csv"
    format_submission([{"image_id": "a", "predicted_label": "free_flow", "confidence": 0.8}], str(output))
    assert validate_submission(output)
