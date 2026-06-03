"""Violation analytics."""

from __future__ import annotations

from typing import Any


def detect_basic_violations(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract known violation detections.

    Args:
        detections: Detector outputs.

    Returns:
        Violation records.

    Raises:
        KeyError: If required detection keys are missing.
    """
    violations = []
    for det in detections:
        if det["class_name"] == "no_helmet":
            violations.append({"type": "no_helmet", "bbox": det["bbox"], "confidence": det["confidence"]})
    return violations
