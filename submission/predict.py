"""Generate HackerEarth-style submissions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.pipeline import TrafficInferencePipeline
from submission.format_output import format_submission
from submission.validate_submission import validate_submission


def collect_inputs(test_dir: str | Path | None = None, test_csv: str | Path | None = None) -> list[dict[str, str]]:
    """Collect test image inputs.

    Args:
        test_dir: Directory containing images.
        test_csv: CSV with image_path or video_path column.

    Returns:
        Input rows.

    Raises:
        FileNotFoundError: If provided input path is missing.
    """
    if test_csv:
        csv_path = Path(test_csv)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        import csv

        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            columns = set(reader.fieldnames or [])
        path_col = "image_path" if "image_path" in columns else "video_path"
        return [{"image_id": str(row.get("image_id") or Path(row[path_col]).stem), "path": str(row[path_col])} for row in rows]
    root = Path(test_dir or "data/splits/test/images")
    if not root.exists():
        raise FileNotFoundError(root)
    return [{"image_id": path.stem, "path": str(path)} for path in root.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]


def generate_submission(output: str = "submission.csv", test_dir: str | None = None, test_csv: str | None = None) -> str:
    """Generate submission CSV and detailed CSV.

    Args:
        output: Output CSV path.
        test_dir: Optional image directory.
        test_csv: Optional test CSV.

    Returns:
        Output CSV path.

    Raises:
        FileNotFoundError: If test input is missing.
    """
    pipeline = TrafficInferencePipeline()
    rows = []
    detailed = []
    for item in collect_inputs(test_dir, test_csv):
        pred = pipeline.predict_image(item["path"])
        rows.append({"image_id": item["image_id"], "predicted_label": pred["congestion_level"], "confidence": pred["confidence"]})
        detailed.append({**rows[-1], "vehicle_count": pred["vehicle_count"], "violation_count": len(pred["violations"])})
    format_submission(rows, output)
    import csv

    detailed_path = Path(output).with_name("submission_detailed.csv")
    with detailed_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_id", "predicted_label", "confidence", "vehicle_count", "violation_count"])
        writer.writeheader()
        writer.writerows(detailed)
    validate_submission(output)
    return output


def main() -> None:
    """Run submission generation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()
    print(generate_submission(args.output, args.test_dir, args.test_csv))


if __name__ == "__main__":
    main()
