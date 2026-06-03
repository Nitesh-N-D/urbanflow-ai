"""Simple IoU tracker fallback for vehicle movement analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Track:
    """A single active object track."""

    track_id: int
    bbox: list[float]
    age: int = 0
    hits: int = 1


class SimpleIOUTracker:
    """Minimal IoU tracker used when ByteTrack is unavailable."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30) -> None:
        """Initialize tracker.

        Args:
            iou_threshold: Minimum IoU for association.
            max_age: Maximum stale frames.

        Returns:
            None.

        Raises:
            ValueError: If thresholds are invalid.
        """
        if not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold must be in [0, 1]")
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: list[Track] = []
        self.next_id = 1

    def update(self, detections: list[dict]) -> list[Track]:
        """Update tracks from detections.

        Args:
            detections: Detection dictionaries with bbox keys.

        Returns:
            Active tracks.

        Raises:
            KeyError: If detections lack bbox fields.
        """
        unmatched = set(range(len(detections)))
        for track in self.tracks:
            best_idx = None
            best_iou = 0.0
            for idx in list(unmatched):
                score = _iou(track.bbox, detections[idx]["bbox"])
                if score > best_iou:
                    best_idx, best_iou = idx, score
            if best_idx is not None and best_iou >= self.iou_threshold:
                track.bbox = detections[best_idx]["bbox"]
                track.age = 0
                track.hits += 1
                unmatched.remove(best_idx)
            else:
                track.age += 1
        for idx in unmatched:
            self.tracks.append(Track(self.next_id, detections[idx]["bbox"]))
            self.next_id += 1
        self.tracks = [track for track in self.tracks if track.age <= self.max_age]
        return list(self.tracks)


def _iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)
