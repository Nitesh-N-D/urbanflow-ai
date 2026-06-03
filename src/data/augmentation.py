"""Traffic-focused image augmentation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

try:
    import albumentations as A
except ImportError:  # pragma: no cover
    A = None


class TrafficAugmentor:
    """Builds robust augmentations for CCTV traffic imagery."""

    def __init__(self, image_size: int = 640) -> None:
        """Initialize the augmentor.

        Args:
            image_size: Target image size.

        Returns:
            None.

        Raises:
            ValueError: If image_size is not positive.
        """
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.image_size = image_size
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Any:
        """Create an Albumentations pipeline with graceful fallback.

        Args:
            None.

        Returns:
            Albumentations compose object or None.

        Raises:
            None.
        """
        if A is None:
            return None
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.RandomShadow(p=0.2),
            A.CLAHE(p=0.3),
        ]
        try:
            transforms.append(A.RandomRain(blur_value=2, p=0.1))
        except AttributeError:
            transforms.append(A.GaussianBlur(blur_limit=5, p=0.1))
        return A.Compose(transforms, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

    def augment_image(self, image: Image.Image, boxes: list[list[float]] | None = None) -> tuple[Image.Image, list[list[float]]]:
        """Augment a PIL image and YOLO boxes.

        Args:
            image: Source image.
            boxes: YOLO boxes as [x_center, y_center, width, height].

        Returns:
            Augmented image and boxes.

        Raises:
            ValueError: If image is empty.
        """
        if image.width <= 0 or image.height <= 0:
            raise ValueError("Cannot augment an empty image")
        boxes = boxes or []
        if self.pipeline is not None:
            labels = [0 for _ in boxes]
            result = self.pipeline(image=np.array(image), bboxes=boxes, class_labels=labels)
            return Image.fromarray(result["image"]), [list(box) for box in result["bboxes"]]
        fallback = ImageEnhance.Contrast(image).enhance(1.15).filter(ImageFilter.SHARPEN)
        return fallback, boxes

    def augment_file(self, image_path: str | Path, output_path: str | Path) -> Path:
        """Augment one image file.

        Args:
            image_path: Input image path.
            output_path: Output image path.

        Returns:
            Written output path.

        Raises:
            FileNotFoundError: If image_path is missing.
        """
        src = Path(image_path)
        if not src.exists():
            raise FileNotFoundError(src)
        image = Image.open(src).convert("RGB")
        aug, _ = self.augment_image(image)
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        aug.save(dest)
        return dest
