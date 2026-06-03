"""Ingestion tests."""

from __future__ import annotations

from src.data.ingestion import TrafficDatasetIngestor


def test_synthetic_ingestion_creates_splits() -> None:
    """Synthetic fallback creates a valid dataset YAML."""
    ingestor = TrafficDatasetIngestor("config/config.yaml")
    items = ingestor.generate_synthetic_dataset(12)
    train, val, test = ingestor.create_splits(items)
    yaml_path = ingestor.copy_splits_to_disk(train, val, test, "data/splits_test")
    assert yaml_path.exists()
    assert len(train) + len(val) + len(test) == 12
