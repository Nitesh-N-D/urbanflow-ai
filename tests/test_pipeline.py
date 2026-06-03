"""Pipeline tests."""

from __future__ import annotations

from PIL import Image

from src.inference.pipeline import TrafficInferencePipeline


def test_pipeline_prediction_schema() -> None:
    """Pipeline returns complete prediction schema."""
    pipeline = TrafficInferencePipeline("config/config.yaml")
    pred = pipeline.predict_image(Image.new("RGB", (160, 120), (110, 110, 110)))
    assert {"congestion_level", "confidence", "vehicle_count", "detections", "violations"}.issubset(pred)
    assert 0.0 <= pred["confidence"] <= 1.0
