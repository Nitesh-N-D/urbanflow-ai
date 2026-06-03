"""Train or smoke-test the congestion classifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ingestion import TrafficDatasetIngestor
from src.models.classifier import CongestionClassifier


def main() -> None:
    """Run classifier preparation.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    ingestor = TrafficDatasetIngestor(args.config)
    items = ingestor.ingest()
    split_dir = Path(ingestor.copy_splits_to_disk(*ingestor.create_splits(items))).parent
    classifier = CongestionClassifier()
    print(classifier.train(csv_path=split_dir / "classification_labels.csv", dry_run=args.dry_run))


if __name__ == "__main__":
    main()
