"""Visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def draw_detections(image: str | Path | Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
    """Draw bounding boxes on an image.

    Args:
        image: Source image.
        detections: Detection dictionaries.

    Returns:
        Annotated image.

    Raises:
        FileNotFoundError: If image path is missing.
    """
    pil = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.convert("RGB")
    draw = ImageDraw.Draw(pil)
    for det in detections:
        box = det["bbox"]
        draw.rectangle(box, outline=(255, 215, 0), width=3)
        draw.text((box[0], max(0, box[1] - 12)), f"{det['class_name']} {det['confidence']:.2f}", fill=(255, 255, 255))
    return pil
