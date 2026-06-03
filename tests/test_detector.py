"""Detector tests."""

from __future__ import annotations

from PIL import Image

from src.models.detector import TrafficDetector


def test_detector_predict_schema() -> None:
    """Fallback detector returns schema-compatible detections."""
    detector = TrafficDetector(weights="missing.pt")
    detections = detector.predict(Image.new("RGB", (160, 120), (120, 120, 120)))
    assert isinstance(detections, list)
    if detections:
        assert {"bbox", "confidence", "class_name"}.issubset(detections[0])
