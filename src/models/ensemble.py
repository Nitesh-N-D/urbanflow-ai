"""Model fusion for congestion, detection, and violation summaries."""

from __future__ import annotations

import json
from collections import Counter, deque
from pathlib import Path
from typing import Any

from PIL import Image

from .classifier import CongestionClassifier
from .detector import TrafficDetector

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


class TrafficEnsemble:
    """Combines detector counts and scene classifier probabilities."""

    congestion_names = ["free_flow", "slow_moving", "heavy_traffic", "standstill"]

    def __init__(self, detector: TrafficDetector | None = None, classifier: CongestionClassifier | None = None) -> None:
        """Initialize ensemble.

        Args:
            detector: Optional detector instance.
            classifier: Optional classifier instance.

        Returns:
            None.

        Raises:
            None.
        """
        self.detector = detector or TrafficDetector()
        self.classifier = classifier or CongestionClassifier()

    def predict(self, image: str | Path | Image.Image) -> dict[str, Any]:
        """Predict complete traffic intelligence for one frame.

        Args:
            image: Image path or PIL image.

        Returns:
            Prediction payload.

        Raises:
            FileNotFoundError: If image path is missing.
        """
        detections = self.detector.predict(image)
        class_probs = self.classifier.predict_proba(image)
        count = len([d for d in detections if d["class_name"] in {"car", "motorcycle", "truck", "bus", "auto_rickshaw"}])
        density_probs = self._density_probs(count)
        fused = {name: 0.6 * class_probs.get(name, 0.0) + 0.4 * density_probs.get(name, 0.0) for name in self.congestion_names}
        final = max(fused, key=fused.get)
        breakdown = Counter(d["class_name"] for d in detections)
        violations = [
            {"type": "no_helmet", "bbox": d["bbox"], "confidence": d["confidence"]}
            for d in detections
            if d["class_name"] == "no_helmet"
        ]
        return {
            "congestion_level": final,
            "confidence": float(fused[final]),
            "vehicle_count": count,
            "vehicle_breakdown": dict(breakdown),
            "detections": detections,
            "violations": violations,
            "class_probabilities": fused,
        }

    def predict_video(self, video_path: str | Path, fps: float = 5.0, smoothing: int = 5) -> pd.DataFrame:
        """Predict congestion over a video file.

        Args:
            video_path: Video path.
            fps: Target processing frame rate.
            smoothing: Majority smoothing window size.

        Returns:
            Per-frame predictions.

        Raises:
            ImportError: If OpenCV is unavailable.
            FileNotFoundError: If video is missing.
        """
        if cv2 is None:
            raise ImportError("opencv-python-headless is required for video prediction")
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(path)
        cap = cv2.VideoCapture(str(path))
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        stride = max(int(native_fps / fps), 1)
        rows = []
        window: deque[str] = deque(maxlen=smoothing)
        frame_id = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_id % stride == 0:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pred = self.predict(image)
                window.append(pred["congestion_level"])
                smooth = Counter(window).most_common(1)[0][0]
                rows.append({
                    "timestamp": frame_id / native_fps,
                    "frame_id": frame_id,
                    "congestion_level": smooth,
                    "confidence": pred["confidence"],
                    "vehicle_count": pred["vehicle_count"],
                    "violation_count": len(pred["violations"]),
                })
            frame_id += 1
        cap.release()
        try:
            import pandas as pd

            return pd.DataFrame(rows)
        except ImportError:
            return rows

    def write_video_report(self, video_path: str | Path, output_dir: str | Path = "reports") -> tuple[Path, Path]:
        """Write CSV and JSON summary for a video.

        Args:
            video_path: Video path.
            output_dir: Output directory.

        Returns:
            CSV and JSON paths.

        Raises:
            OSError: If reports cannot be written.
        """
        df = self.predict_video(video_path)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / f"{Path(video_path).stem}_frames.csv"
        json_path = out / f"{Path(video_path).stem}_summary.json"
        if hasattr(df, "to_csv"):
            df.to_csv(csv_path, index=False)
            frame_count = int(len(df))
            mean_count = float(df["vehicle_count"].mean()) if len(df) else 0.0
            dominant = df["congestion_level"].mode().iloc[0] if len(df) else "unknown"
        else:
            import csv

            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["timestamp", "frame_id", "congestion_level", "confidence", "vehicle_count", "violation_count"])
                writer.writeheader()
                writer.writerows(df)
            frame_count = len(df)
            mean_count = sum(row["vehicle_count"] for row in df) / frame_count if frame_count else 0.0
            labels = [row["congestion_level"] for row in df]
            dominant = Counter(labels).most_common(1)[0][0] if labels else "unknown"
        summary = {
            "frames": frame_count,
            "mean_vehicle_count": mean_count,
            "dominant_congestion": dominant,
        }
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return csv_path, json_path

    def _density_probs(self, count: int) -> dict[str, float]:
        if count < 5:
            raw = [0.8, 0.15, 0.04, 0.01]
        elif count < 15:
            raw = [0.1, 0.72, 0.15, 0.03]
        elif count < 25:
            raw = [0.03, 0.17, 0.68, 0.12]
        else:
            raw = [0.01, 0.06, 0.23, 0.70]
        return dict(zip(self.congestion_names, raw))
