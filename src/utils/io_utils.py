"""Shared file and configuration helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file with environment placeholder expansion.

    Args:
        path: YAML file path.

    Returns:
        Parsed YAML mapping.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """
    text = Path(path).read_text(encoding="utf-8")
    if yaml is None:
        return _resolve_env_defaults(_default_config())
    for key, value in os.environ.items():
        text = text.replace(f"${{{key}}}", value)
    data = yaml.safe_load(text) or {}
    return _resolve_env_defaults(data)


def _default_config() -> dict[str, Any]:
    """Return a built-in config when PyYAML is not installed.

    Args:
        None.

    Returns:
        Default configuration mapping.

    Raises:
        None.
    """
    return {
        "project": {"seed": 42},
        "data": {
            "raw_dir": os.environ.get("BTP_DATA_DIR", "data/raw"),
            "processed_dir": "data/processed",
            "splits_dir": "data/splits",
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15,
            "image_size": 640,
            "synthetic_count": 96,
        },
        "model": {
            "detector": {
                "pretrained_weights": "yolov8m.pt",
                "batch_size": 16,
                "epochs": 100,
                "class_names": ["car", "motorcycle", "truck", "bus", "auto_rickshaw", "person", "helmet", "no_helmet"],
            },
            "classifier": {"class_names": ["free_flow", "slow_moving", "heavy_traffic", "standstill"]},
        },
        "inference": {"class_thresholds": {}},
    }


def _resolve_env_defaults(value: Any) -> Any:
    """Resolve ${VAR:default} placeholders recursively.

    Args:
        value: YAML value.

    Returns:
        Value with environment defaults expanded.

    Raises:
        ValueError: If placeholder syntax is malformed.
    """
    if isinstance(value, dict):
        return {k: _resolve_env_defaults(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_defaults(v) for v in value]
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        body = value[2:-1]
        if ":" in body:
            key, default = body.split(":", 1)
            resolved = os.environ.get(key, default)
            return None if resolved == "null" else resolved
    return value


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory.

    Args:
        path: Directory path.

    Returns:
        Created directory as a Path.

    Raises:
        OSError: If the path cannot be created.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_seed(seed: int = 42) -> None:
    """Set common random seeds.

    Args:
        seed: Seed value.

    Returns:
        None.

    Raises:
        RuntimeError: If torch seed setup fails unexpectedly.
    """
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        return
