"""Deployment export helpers."""

from __future__ import annotations

from pathlib import Path


def write_model_card(output_path: str | Path = "model_card.md") -> Path:
    """Write a concise model card.

    Args:
        output_path: Destination path.

    Returns:
        Written path.

    Raises:
        OSError: If writing fails.
    """
    path = Path(output_path)
    path.write_text(
        "# Gridlock 2.0 Model Card\n\nCapabilities: traffic object detection, congestion classification, violation summaries.\n\nLimitations: synthetic fallback is for pipeline validation only; leaderboard results require BTP data fine-tuning.\n",
        encoding="utf-8",
    )
    return path
