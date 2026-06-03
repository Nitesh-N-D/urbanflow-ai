"""Torch dataset helpers with optional torch fallback."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object
    transforms = None


class CongestionImageDataset(Dataset):
    """Image classification dataset for congestion labels."""

    def __init__(self, csv_path: str | Path, split: str = "train", image_size: int = 300) -> None:
        """Load samples from a classification CSV.

        Args:
            csv_path: CSV with split,image_path,label columns.
            split: Split to load.
            image_size: Transform image size.

        Returns:
            None.

        Raises:
            FileNotFoundError: If the CSV is missing.
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)
        self.labels = ["free_flow", "slow_moving", "heavy_traffic", "standstill"]
        self.samples: list[tuple[Path, int]] = []
        with self.csv_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("split") == split:
                    self.samples.append((Path(row["image_path"]), self.labels.index(row.get("label", "free_flow"))))
        if transforms:
            self.transform: Any = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = None

    def __len__(self) -> int:
        """Return sample count."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        """Return one transformed sample.

        Args:
            index: Sample index.

        Returns:
            Image tensor-like object and numeric label.

        Raises:
            IndexError: If index is out of range.
        """
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            return self.transform(image), label
        return image, label
