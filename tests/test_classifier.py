"""Classifier tests."""

from __future__ import annotations

from PIL import Image

from src.models.classifier import CongestionClassifier


def test_classifier_probabilities_sum_to_one() -> None:
    """Heuristic classifier probabilities are normalized."""
    classifier = CongestionClassifier()
    probs = classifier.predict_proba(Image.new("RGB", (160, 120), (80, 80, 80)))
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert set(probs) == {"free_flow", "slow_moving", "heavy_traffic", "standstill"}
