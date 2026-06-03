"""Post-processing helpers for detections."""

from __future__ import annotations

from typing import Any


def filter_detections(detections: list[dict[str, Any]], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    """Filter detections using class-specific thresholds.

    Args:
        detections: Detection dictionaries.
        thresholds: Class name to confidence threshold.

    Returns:
        Filtered detections.

    Raises:
        KeyError: If a detection lacks required keys.
    """
    return [det for det in detections if det["confidence"] >= thresholds.get(det["class_name"], 0.25)]


def weighted_box_fusion(detections: list[dict[str, Any]], image_size: tuple[int, int]) -> list[dict[str, Any]]:
    """Apply weighted box fusion when ensemble-boxes is installed.

    Args:
        detections: Detection dictionaries in pixel coordinates.
        image_size: Image width and height.

    Returns:
        Fused detections, or original detections if dependency is absent.

    Raises:
        ValueError: If image dimensions are invalid.
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be positive")
    try:
        from ensemble_boxes import weighted_boxes_fusion

        boxes = [[[d["bbox"][0] / width, d["bbox"][1] / height, d["bbox"][2] / width, d["bbox"][3] / height] for d in detections]]
        scores = [[float(d["confidence"]) for d in detections]]
        labels = [[int(d.get("class_id", 0)) for d in detections]]
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(boxes, scores, labels, iou_thr=0.55, skip_box_thr=0.001)
        output = []
        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            template = next((d for d in detections if int(d.get("class_id", 0)) == int(label)), detections[0] if detections else {})
            output.append({
                "bbox": [float(box[0] * width), float(box[1] * height), float(box[2] * width), float(box[3] * height)],
                "confidence": float(score),
                "class_id": int(label),
                "class_name": template.get("class_name", str(int(label))),
            })
        return output
    except Exception:
        return detections
