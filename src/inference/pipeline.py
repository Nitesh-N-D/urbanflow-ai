"""High-level inference pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from src.inference.postprocess import filter_detections
from src.models.ensemble import TrafficEnsemble
from src.utils.io_utils import load_yaml


class TrafficInferencePipeline:
    """Config-driven wrapper around the traffic ensemble."""

    def __init__(self, config_path: str | Path = "config/config.yaml") -> None:
        """Initialize pipeline.

        Args:
            config_path: Configuration path.

        Returns:
            None.

        Raises:
            FileNotFoundError: If config is missing.
        """
        self.config = load_yaml(config_path)
        self.ensemble = TrafficEnsemble()
        self.thresholds = self.config.get("inference", {}).get("class_thresholds", {})

    def predict_image(self, image: str | Path | Image.Image) -> dict[str, Any]:
        """Run image inference.

        Args:
            image: Path or PIL image.

        Returns:
            Prediction payload.

        Raises:
            FileNotFoundError: If image path is missing.
        """
        pred = self.ensemble.predict(image)
        pred["detections"] = filter_detections(pred["detections"], self.thresholds)
        return pred
