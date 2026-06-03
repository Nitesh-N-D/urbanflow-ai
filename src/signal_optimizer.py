"""Adaptive traffic signal optimizer."""

from __future__ import annotations


def recommend_signal_timing(approach_scores: dict[str, float], min_green: int = 10, max_cycle: int = 120) -> dict[str, int]:
    """Allocate green time proportional to approach congestion.

    Args:
        approach_scores: Approach name to congestion demand score.
        min_green: Minimum green seconds.
        max_cycle: Maximum total cycle seconds.

    Returns:
        Approach name to green seconds.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not approach_scores:
        raise ValueError("approach_scores cannot be empty")
    total = sum(max(0.0, score) for score in approach_scores.values()) or 1.0
    remaining = max_cycle - min_green * len(approach_scores)
    return {name: int(min_green + remaining * max(0.0, score) / total) for name, score in approach_scores.items()}
