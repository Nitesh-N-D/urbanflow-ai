"""Traffic dataset ingestion with auto-detection and synthetic fallback."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from sklearn.model_selection import train_test_split
except ImportError:  # pragma: no cover
    train_test_split = None

from src.utils.io_utils import ensure_dir, load_yaml, set_seed

logger = logging.getLogger(__name__)


@dataclass
class DatasetItem:
    """One normalized image sample.

    Args:
        image_path: Path to the image.
        label_path: Optional YOLO label path.
        congestion_label: Optional scene label.
    """

    image_path: Path
    label_path: Path | None = None
    congestion_label: str | None = None


class TrafficDatasetIngestor:
    """Ingests BTP-like traffic datasets into YOLO-compatible splits."""

    SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".ts"}
    SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """Initialize the ingestor.

        Args:
            config_path: Path to the YAML configuration.

        Returns:
            None.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
        """
        self.config_path = Path(config_path)
        self.config = load_yaml(self.config_path)
        set_seed(int(self.config.get("project", {}).get("seed", 42)))
        self.raw_dir = Path(self.config["data"]["raw_dir"])
        self.processed_dir = ensure_dir(self.config["data"]["processed_dir"])
        self.splits_dir = ensure_dir(self.config["data"].get("splits_dir", "data/splits"))
        self.class_names = list(self.config["model"]["detector"]["class_names"])
        self.congestion_names = list(self.config["model"]["classifier"]["class_names"])

    def detect_format(self, data_dir: str | Path) -> str:
        """Auto-detect a dataset format.

        Args:
            data_dir: Dataset root directory.

        Returns:
            One of yolo, coco, voc, video, images_only, missing.

        Raises:
            ValueError: If the path points to a file instead of a directory.
        """
        root = Path(data_dir)
        if not root.exists():
            return "missing"
        if not root.is_dir():
            raise ValueError(f"Dataset path is not a directory: {root}")
        if (root / "labels").exists() and list((root / "labels").rglob("*.txt")):
            return "yolo"
        for json_path in [*root.glob("*.json"), *root.glob("annotations/*.json")]:
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                if {"images", "annotations"}.issubset(data.keys()):
                    return "coco"
            except (OSError, json.JSONDecodeError):
                continue
        if (root / "Annotations").exists() and list((root / "Annotations").glob("*.xml")):
            return "voc"
        if any(path.suffix.lower() in self.SUPPORTED_VIDEO_EXTS for path in root.iterdir()):
            return "video"
        if list(self._iter_images(root)):
            return "images_only"
        return "missing"

    def ingest(self, data_dir: str | Path | None = None) -> list[DatasetItem]:
        """Load or generate a normalized dataset.

        Args:
            data_dir: Optional dataset directory.

        Returns:
            List of normalized dataset items.

        Raises:
            RuntimeError: If all ingestion strategies fail.
        """
        root = Path(data_dir) if data_dir else self.raw_dir
        fmt = self.detect_format(root)
        logger.info("Detected dataset format: %s", fmt)
        try:
            if fmt == "yolo":
                items = self._load_yolo(root)
            elif fmt == "coco":
                items = self._convert_coco(root)
            elif fmt == "voc":
                items = self._convert_voc(root)
            elif fmt == "video":
                items = self._extract_video_frames(root)
            elif fmt == "images_only":
                items = [DatasetItem(path, self._write_empty_label(path)) for path in self._iter_images(root)]
            else:
                items = self.generate_synthetic_dataset()
            if not items:
                items = self.generate_synthetic_dataset()
            return items
        except Exception as exc:
            logger.warning("Ingestion failed, using synthetic fallback: %s", exc)
            return self.generate_synthetic_dataset()

    def create_splits(self, items: list[DatasetItem]) -> tuple[list[DatasetItem], list[DatasetItem], list[DatasetItem]]:
        """Create train, validation, and test splits.

        Args:
            items: Dataset items.

        Returns:
            Train, validation, and test item lists.

        Raises:
            ValueError: If the item list is empty.
        """
        if not items:
            raise ValueError("Cannot split an empty dataset")
        random.shuffle(items)
        if train_test_split:
            train, temp = train_test_split(items, train_size=float(self.config["data"]["train_split"]), random_state=42)
            val_fraction = float(self.config["data"]["val_split"]) / (
                float(self.config["data"]["val_split"]) + float(self.config["data"]["test_split"])
            )
            val, test = train_test_split(temp, train_size=val_fraction, random_state=42)
            return list(train), list(val), list(test)
        train_end = int(len(items) * float(self.config["data"]["train_split"]))
        val_end = train_end + int(len(items) * float(self.config["data"]["val_split"]))
        return items[:train_end], items[train_end:val_end], items[val_end:]

    def copy_splits_to_disk(
        self,
        train: list[DatasetItem],
        val: list[DatasetItem],
        test: list[DatasetItem],
        output_dir: str | Path | None = None,
    ) -> Path:
        """Copy split files to YOLO folder layout and write dataset.yaml.

        Args:
            train: Training samples.
            val: Validation samples.
            test: Test samples.
            output_dir: Output split directory.

        Returns:
            Path to dataset.yaml.

        Raises:
            FileNotFoundError: If source images are missing.
        """
        base = ensure_dir(output_dir or self.splits_dir)
        for split_name, split_items in {"train": train, "val": val, "test": test}.items():
            image_dir = ensure_dir(base / split_name / "images")
            label_dir = ensure_dir(base / split_name / "labels")
            for item in split_items:
                if not item.image_path.exists():
                    raise FileNotFoundError(f"Missing image: {item.image_path}")
                dest_image = image_dir / item.image_path.name
                shutil.copy2(item.image_path, dest_image)
                label = item.label_path or self._write_empty_label(item.image_path)
                shutil.copy2(label, label_dir / f"{dest_image.stem}.txt")
        yaml_path = self.generate_yolo_yaml(base)
        self._write_classification_csv(base, {"train": train, "val": val, "test": test})
        return yaml_path

    def generate_yolo_yaml(self, split_dir: str | Path) -> Path:
        """Generate Ultralytics dataset YAML.

        Args:
            split_dir: Split root directory.

        Returns:
            Path to written YAML file.

        Raises:
            FileNotFoundError: If train or validation images are missing.
        """
        base = Path(split_dir).resolve()
        content = {
            "path": str(base),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "names": {i: name for i, name in enumerate(self.class_names)},
        }
        for split in ["train", "val"]:
            split_path = base / content[split]
            if not split_path.exists():
                raise FileNotFoundError(f"Split not found: {split_path}")
        yaml_path = base / "dataset.yaml"
        if yaml is not None:
            text = yaml.safe_dump(content, sort_keys=False)
        else:
            names = "\n".join(f"  {idx}: {name}" for idx, name in content["names"].items())
            text = f"path: {content['path']}\ntrain: {content['train']}\nval: {content['val']}\ntest: {content['test']}\nnames:\n{names}\n"
        yaml_path.write_text(text, encoding="utf-8")
        return yaml_path

    def generate_synthetic_dataset(self, count: int | None = None) -> list[DatasetItem]:
        """Generate synthetic CCTV-like traffic frames and YOLO labels.

        Args:
            count: Optional number of samples.

        Returns:
            Generated dataset items.

        Raises:
            OSError: If files cannot be written.
        """
        total = count or int(self.config["data"].get("synthetic_count", 96))
        image_dir = ensure_dir(self.processed_dir / "synthetic" / "images")
        label_dir = ensure_dir(self.processed_dir / "synthetic" / "labels")
        items: list[DatasetItem] = []
        for idx in range(total):
            congestion = random.choice(self.congestion_names)
            vehicle_count = {"free_flow": 3, "slow_moving": 9, "heavy_traffic": 18, "standstill": 28}[congestion]
            image, boxes = self._draw_synthetic_frame(vehicle_count)
            image_path = image_dir / f"synthetic_{idx:04d}.jpg"
            label_path = label_dir / f"synthetic_{idx:04d}.txt"
            image.save(image_path, quality=92)
            label_path.write_text("\n".join(boxes) + "\n", encoding="utf-8")
            items.append(DatasetItem(image_path, label_path, congestion))
        return items

    def print_statistics(self, items: list[DatasetItem]) -> dict[str, int]:
        """Print and return simple dataset statistics.

        Args:
            items: Dataset items.

        Returns:
            Class distribution mapping.

        Raises:
            None.
        """
        counts = {name: 0 for name in self.congestion_names}
        for item in items:
            if item.congestion_label in counts:
                counts[item.congestion_label] += 1
        print(f"Total images: {len(items)}")
        print(f"Congestion distribution: {counts}")
        return counts

    def _load_yolo(self, root: Path) -> list[DatasetItem]:
        images = list(self._iter_images(root / "images")) or list(self._iter_images(root))
        return [DatasetItem(path, root / "labels" / f"{path.stem}.txt") for path in images]

    def _convert_coco(self, root: Path) -> list[DatasetItem]:
        json_path = next(iter([*root.glob("*.json"), *root.glob("annotations/*.json")]))
        data = json.loads(json_path.read_text(encoding="utf-8"))
        images = {img["id"]: img for img in data.get("images", [])}
        grouped: dict[int, list[dict]] = {}
        for ann in data.get("annotations", []):
            grouped.setdefault(int(ann["image_id"]), []).append(ann)
        out_labels = ensure_dir(self.processed_dir / "coco_labels")
        items = []
        for image_id, image_info in images.items():
            image_path = root / image_info.get("file_name", "")
            if not image_path.exists():
                matches = list(root.rglob(Path(image_info.get("file_name", "")).name))
                image_path = matches[0] if matches else image_path
            width = float(image_info.get("width", 1))
            height = float(image_info.get("height", 1))
            lines = []
            for ann in grouped.get(image_id, []):
                x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
                cls = max(0, int(ann.get("category_id", 1)) - 1)
                lines.append(f"{cls} {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} {w / width:.6f} {h / height:.6f}")
            label_path = out_labels / f"{image_path.stem}.txt"
            label_path.write_text("\n".join(lines), encoding="utf-8")
            if image_path.exists():
                items.append(DatasetItem(image_path, label_path))
        return items

    def _convert_voc(self, root: Path) -> list[DatasetItem]:
        out_labels = ensure_dir(self.processed_dir / "voc_labels")
        items = []
        for xml_path in (root / "Annotations").glob("*.xml"):
            tree = ET.parse(xml_path)
            node = tree.getroot()
            size = node.find("size")
            width = float(size.findtext("width", "1")) if size is not None else 1.0
            height = float(size.findtext("height", "1")) if size is not None else 1.0
            image_name = node.findtext("filename", f"{xml_path.stem}.jpg")
            matches = list(root.rglob(image_name))
            if not matches:
                continue
            lines = []
            for obj in node.findall("object"):
                name = obj.findtext("name", "car")
                cls = self.class_names.index(name) if name in self.class_names else 0
                box = obj.find("bndbox")
                if box is None:
                    continue
                xmin = float(box.findtext("xmin", "0"))
                ymin = float(box.findtext("ymin", "0"))
                xmax = float(box.findtext("xmax", "0"))
                ymax = float(box.findtext("ymax", "0"))
                lines.append(f"{cls} {((xmin + xmax) / 2) / width:.6f} {((ymin + ymax) / 2) / height:.6f} {(xmax - xmin) / width:.6f} {(ymax - ymin) / height:.6f}")
            label_path = out_labels / f"{xml_path.stem}.txt"
            label_path.write_text("\n".join(lines), encoding="utf-8")
            items.append(DatasetItem(matches[0], label_path))
        return items

    def _extract_video_frames(self, root: Path, fps: float = 5.0) -> list[DatasetItem]:
        if cv2 is None:
            raise ImportError("opencv-python-headless is required for video ingestion")
        out_images = ensure_dir(self.processed_dir / "video_frames")
        items = []
        for video_path in root.iterdir():
            if video_path.suffix.lower() not in self.SUPPORTED_VIDEO_EXTS:
                continue
            cap = cv2.VideoCapture(str(video_path))
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 25
            stride = max(int(native_fps / fps), 1)
            frame_id = 0
            saved = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_id % stride == 0:
                    dest = out_images / f"{video_path.stem}_{saved:05d}.jpg"
                    cv2.imwrite(str(dest), frame)
                    items.append(DatasetItem(dest, self._write_empty_label(dest)))
                    saved += 1
                frame_id += 1
            cap.release()
        return items

    def _iter_images(self, root: Path) -> Iterable[Path]:
        if not root.exists():
            return []
        return (path for path in root.rglob("*") if path.suffix.lower() in self.SUPPORTED_IMAGE_EXTS)

    def _write_empty_label(self, image_path: Path) -> Path:
        label_dir = ensure_dir(self.processed_dir / "empty_labels")
        label_path = label_dir / f"{image_path.stem}.txt"
        label_path.write_text("", encoding="utf-8")
        return label_path

    def _draw_synthetic_frame(self, vehicle_count: int) -> tuple[Image.Image, list[str]]:
        width, height = 640, 384
        image = Image.new("RGB", (width, height), (90, 93, 94))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width, 80), fill=(130, 151, 164))
        for lane_x in [160, 320, 480]:
            draw.line((lane_x, 95, lane_x, height), fill=(225, 225, 180), width=3)
        boxes = []
        for _ in range(vehicle_count):
            cls = random.randint(0, min(4, len(self.class_names) - 1))
            w = random.randint(28, 78)
            h = random.randint(18, 46)
            x = random.randint(10, width - w - 10)
            y = random.randint(95, height - h - 8)
            color = random.choice([(32, 104, 211), (238, 196, 62), (215, 67, 63), (68, 171, 113), (242, 242, 242)])
            draw.rounded_rectangle((x, y, x + w, y + h), radius=4, fill=color, outline=(22, 22, 22))
            boxes.append(f"{cls} {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} {w / width:.6f} {h / height:.6f}")
        return image, boxes

    def _write_classification_csv(self, base: Path, splits: dict[str, list[DatasetItem]]) -> None:
        with (base / "classification_labels.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["split", "image_path", "label"])
            for split, items in splits.items():
                for item in items:
                    label = item.congestion_label or "free_flow"
                    writer.writerow([split, item.image_path, label])


def main() -> None:
    """Run ingestion from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    ingestor = TrafficDatasetIngestor(args.config)
    items = ingestor.ingest(args.data_dir)
    splits = ingestor.create_splits(items)
    yaml_path = ingestor.copy_splits_to_disk(*splits)
    ingestor.print_statistics(items)
    print(f"YOLO dataset YAML: {yaml_path}")


if __name__ == "__main__":
    main()
