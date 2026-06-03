"""Helmet detector training/export placeholder with real detector integration."""

from __future__ import annotations

from pathlib import Path

from src.data.download_supplemental import download_helmet_dataset
from src.models.detector import TrafficDetector


def prepare_helmet_detector(output_dir: str = "models/helmet") -> Path:
    """Prepare a helmet detector artifact.

    Args:
        output_dir: Model output directory.

    Returns:
        Path to exported artifact.

    Raises:
        OSError: If output cannot be written.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    download_helmet_dataset()
    detector = TrafficDetector(weights="yolov8n.pt")
    return detector.export_onnx(Path(output_dir) / "helmet_detector.onnx")
