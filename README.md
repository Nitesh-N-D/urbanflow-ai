# Gridlock 2.0 - Bengaluru Traffic Intelligence

![Python](https://img.shields.io/badge/Python-3.10-blue) ![YOLOv8](https://img.shields.io/badge/YOLOv8-ready-green) ![Streamlit](https://img.shields.io/badge/Streamlit-demo-red) ![FastAPI](https://img.shields.io/badge/FastAPI-api-teal) ![License](https://img.shields.io/badge/License-MIT-yellow)

It detects vehicles and safety violations, classifies congestion, estimates movement patterns, and exposes the result through training scripts, leaderboard submission tools, a FastAPI service, and a Streamlit dashboard.

## Architecture

```text
CCTV frames/video
      |
      +--> Data ingestion: YOLO / COCO / VOC / video / synthetic fallback
      |
      +--> YOLOv8 detector --------+
      |                            |
      +--> Congestion classifier --+--> Ensemble + analytics --> CSV / API / dashboard
      |                            |
      +--> Tracker + flow tools ---+
```



## Prerequisites

Python 3.10+, 8GB RAM, and an optional CUDA GPU. The code falls back to CPU and synthetic data when real data or model dependencies are unavailable.

## Installation

```bash
cd gridlock2
python -m pip install -r requirements.txt
```

## Dataset Setup

Place the BTP/ASTraM dataset under `data/raw` or set `BTP_DATA_DIR`. The ingestor auto-detects YOLO, COCO, VOC, video, or image-only layouts. Until the real dataset is available, it generates synthetic traffic frames.

```bash
python -m src.data.ingestion --config config/config.yaml --data-dir data/raw
```

## Training

```bash
python training/train_detector.py --config config/config.yaml --dry-run
python training/train_detector.py --config config/config.yaml
python training/train_classifier.py --config config/config.yaml --dry-run
```

## Inference And Submission

```bash
python submission/predict.py --test-dir data/splits/test/images --output submission.csv
python submission/validate_submission.py --csv submission.csv
```

## Demo

```bash
uvicorn demo.api:app --reload
streamlit run demo/app.py
python demo/gradio_app.py
```

## Results

| Model | mAP@0.5 | mAP@0.5:0.95 | Congestion F1 | FPS (T4) |
|---|---:|---:|---:|---:|
| YOLOv8m + classifier ensemble | TBD after BTP training | TBD | TBD | TBD |

## Supplemental Datasets

The project includes cached Roboflow download helpers for IDD-style Indian roads and helmet detection data. Downloads require `ROBOFLOW_API_KEY`; otherwise the pipeline continues without crashing.

## License

MIT
