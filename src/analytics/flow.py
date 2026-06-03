"""Traffic flow analytics."""

from __future__ import annotations


def summarize_flow(tracks: list[dict]) -> dict[str, float]:
    """Summarize tracked vehicle flow.

    Args:
        tracks: Track dictionaries.

    Returns:
        Flow summary.

    Raises:
        None.
    """
    if not tracks:
        return {"vehicle_count": 0, "average_speed_proxy": 0.0, "dominant_direction": "unknown"}
    speeds = [float(track.get("speed_proxy", 0.0)) for track in tracks]
    directions = [track.get("direction", "unknown") for track in tracks]
    dominant = max(set(directions), key=directions.count)
    return {"vehicle_count": len(tracks), "average_speed_proxy": sum(speeds) / len(speeds), "dominant_direction": dominant}
