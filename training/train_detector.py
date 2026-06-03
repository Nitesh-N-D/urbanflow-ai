"""Train or dry-run the YOLO traffic detector."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ingestion import TrafficDatasetIngestor
from src.models.detector import TrafficDetector
from src.utils.io_utils import load_yaml


def auto_batch(default: int = 16) -> int:
    """Choose a safe batch size from available CUDA memory.

    Args:
        default: Default batch size.

    Returns:
        Batch size.

    Raises:
        None.
    """
    try:
        import torch

        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            return 32 if gb > 14 else 16 if gb > 8 else 8
    except Exception:
        pass
    return min(default, 4)


def main() -> None:
    """Run detector training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data-yaml", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", "offline")
    cfg = load_yaml(args.config)
    if args.data_yaml:
        data_yaml = Path(args.data_yaml)
    else:
        ingestor = TrafficDatasetIngestor(args.config)
        items = ingestor.ingest()
        data_yaml = ingestor.copy_splits_to_disk(*ingestor.create_splits(items))
    detector = TrafficDetector(weights=cfg["model"]["detector"]["pretrained_weights"], class_names=cfg["model"]["detector"]["class_names"])
    epochs = 2 if args.dry_run else int(cfg["model"]["detector"]["epochs"])
    batch = auto_batch(int(cfg["model"]["detector"]["batch_size"]))
    result = detector.train(data_yaml, epochs=epochs, batch=batch, imgsz=int(cfg["data"]["image_size"]), optimizer="AdamW", lr0=0.001, mosaic=1.0)
    print(result)
    print(detector.validate(data_yaml))


if __name__ == "__main__":
    main()
