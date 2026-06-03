"""Vehicle counting and speed proxy helpers."""

from __future__ import annotations


def count_line_crossings(track_history: dict[int, list[tuple[float, float]]], line_y: float) -> dict[str, int]:
    """Count tracks crossing a horizontal line.

    Args:
        track_history: Track id to center-point history.
        line_y: Virtual counting line y coordinate.

    Returns:
        In/out crossing counts.

    Raises:
        None.
    """
    counts = {"in": 0, "out": 0}
    for points in track_history.values():
        if len(points) < 2:
            continue
        start, end = points[0][1], points[-1][1]
        if start < line_y <= end:
            counts["in"] += 1
        elif start > line_y >= end:
            counts["out"] += 1
    return counts
