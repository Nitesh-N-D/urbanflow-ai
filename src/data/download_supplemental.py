"""Downloads free supplemental traffic datasets with local cache markers."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _cached(output_dir: str | Path) -> Path | None:
    path = Path(output_dir)
    marker = path / ".download_complete"
    return path if marker.exists() else None


def _mark(output_dir: str | Path) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / ".download_complete").write_text("ok\n", encoding="utf-8")


def download_idd(output_dir: str = "data/supplemental/idd") -> str | None:
    """Download an IDD-like dataset via Roboflow when credentials exist.

    Args:
        output_dir: Destination directory.

    Returns:
        Dataset location or None.

    Raises:
        None.
    """
    cached = _cached(output_dir)
    if cached:
        return str(cached)
    try:
        from roboflow import Roboflow

        api_key = os.environ.get("ROBOFLOW_API_KEY", "")
        if not api_key:
            logger.warning("Set ROBOFLOW_API_KEY for Roboflow downloads")
            return None
        rf = Roboflow(api_key=api_key)
        dataset = rf.workspace("indian-driving").project("idd-detection").version(1).download("yolov8", location=output_dir)
        _mark(output_dir)
        return str(dataset.location)
    except Exception as exc:
        logger.warning("IDD download failed: %s", exc)
        return None


def download_helmet_dataset(output_dir: str = "data/supplemental/helmet") -> str | None:
    """Download a public helmet dataset via Roboflow when credentials exist.

    Args:
        output_dir: Destination directory.

    Returns:
        Dataset location or None.

    Raises:
        None.
    """
    cached = _cached(output_dir)
    if cached:
        return str(cached)
    try:
        from roboflow import Roboflow

        api_key = os.environ.get("ROBOFLOW_API_KEY", "")
        if not api_key:
            logger.warning("Set ROBOFLOW_API_KEY for Roboflow downloads")
            return None
        rf = Roboflow(api_key=api_key)
        dataset = rf.workspace("joseph-nelson").project("helmet-detection").version(1).download("yolov8", location=output_dir)
        _mark(output_dir)
        return str(dataset.location)
    except Exception as exc:
        logger.warning("Helmet download failed: %s", exc)
        return None
