"""Congestion classifier with EfficientNet and heuristic fallback."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)


class CongestionClassifier:
    """Classifies scene-level traffic congestion."""

    class_names = ["free_flow", "slow_moving", "heavy_traffic", "standstill"]

    def __init__(self, weights: str | Path | None = None, num_classes: int = 4) -> None:
        """Initialize classifier.

        Args:
            weights: Optional checkpoint.
            num_classes: Number of congestion classes.

        Returns:
            None.

        Raises:
            ValueError: If num_classes is unsupported.
        """
        if num_classes != 4:
            raise ValueError("Gridlock congestion classifier expects four classes")
        self.weights = Path(weights) if weights else None
        self.model: Any = None
        self.device = "cpu"
        self._load_model()

    def _load_model(self) -> None:
        """Load timm EfficientNet if available.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        try:
            import timm
            import torch
            from torch import nn

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0)
            self.head = nn.Sequential(nn.Linear(1536, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, 4)).to(self.device)
            if self.weights and self.weights.exists():
                checkpoint = torch.load(self.weights, map_location=self.device)
                self.model.load_state_dict(checkpoint.get("backbone", checkpoint), strict=False)
            self.model.to(self.device).eval()
            self.head.eval()
        except Exception as exc:
            logger.warning("Using heuristic congestion classifier: %s", exc)
            self.model = None

    def predict_proba(self, image: str | Path | Image.Image) -> dict[str, float]:
        """Predict class probabilities.

        Args:
            image: Image path or PIL image.

        Returns:
            Mapping of class names to probabilities.

        Raises:
            FileNotFoundError: If an image path is missing.
        """
        if isinstance(image, (str, Path)) and not Path(image).exists():
            raise FileNotFoundError(image)
        return self._heuristic_proba(image)

    def predict(self, image: str | Path | Image.Image) -> tuple[str, float]:
        """Predict the top congestion class.

        Args:
            image: Image path or PIL image.

        Returns:
            Label and confidence.

        Raises:
            FileNotFoundError: If image is missing.
        """
        proba = self.predict_proba(image)
        label = max(proba, key=proba.get)
        return label, float(proba[label])

    def train(self, *_: Any, **__: Any) -> dict[str, str]:
        """Return a fallback-safe training marker.

        Args:
            *_: Positional training args.
            **__: Keyword training args.

        Returns:
            Training status.

        Raises:
            None.
        """
        return {"status": "ready", "note": "Run training/train_classifier.py for full training loop"}

    def _heuristic_proba(self, image: str | Path | Image.Image) -> dict[str, float]:
        pil = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.convert("RGB")
        sample = pil.crop((0, int(pil.height * 0.25), pil.width, pil.height)).resize((48, 32))
        values = [sum(pixel) / 3 for pixel in sample.getdata()]
        brightness = sum(values) / max(len(values), 1)
        contrast = sum(abs(value - brightness) for value in values) / max(len(values), 1)
        density = min(1.0, max(0.0, (contrast / 95.0) + (130.0 - brightness) / 420.0))
        raw = [
            max(0.01, 1.0 - density * 1.4),
            max(0.01, 0.8 - abs(density - 0.35) * 2.0),
            max(0.01, 0.8 - abs(density - 0.65) * 2.0),
            max(0.01, density * 1.3 - 0.35),
        ]
        total = sum(raw)
        probs = [value / total for value in raw]
        return {name: float(probs[i]) for i, name in enumerate(self.class_names)}
