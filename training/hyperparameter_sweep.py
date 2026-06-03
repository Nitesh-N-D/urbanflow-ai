"""Minimal hyperparameter sweep launcher."""

from __future__ import annotations


def candidate_grid() -> list[dict[str, float]]:
    """Return a small detector sweep grid.

    Args:
        None.

    Returns:
        Hyperparameter candidates.

    Raises:
        None.
    """
    return [
        {"lr0": 0.001, "iou": 0.45, "conf": 0.25},
        {"lr0": 0.0007, "iou": 0.55, "conf": 0.20},
        {"lr0": 0.0015, "iou": 0.50, "conf": 0.30},
    ]


if __name__ == "__main__":
    for candidate in candidate_grid():
        print(candidate)
