"""Congestion analytics."""

from __future__ import annotations


def estimate_congestion_from_counts(vehicle_count: int) -> str:
    """Estimate congestion label from count.

    Args:
        vehicle_count: Number of visible vehicles.

    Returns:
        Congestion label.

    Raises:
        ValueError: If count is negative.
    """
    if vehicle_count < 0:
        raise ValueError("vehicle_count cannot be negative")
    if vehicle_count < 5:
        return "free_flow"
    if vehicle_count < 15:
        return "slow_moving"
    if vehicle_count < 25:
        return "heavy_traffic"
    return "standstill"
