"""FastAPI serving layer for Gridlock AI."""

from __future__ import annotations

import io
import sys
import time
from collections import Counter
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.pipeline import TrafficInferencePipeline

app = FastAPI(title="Gridlock AI API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
pipeline = TrafficInferencePipeline()
stats: Counter[str] = Counter()


@app.get("/health")
async def health() -> dict[str, Any]:
    """Return service health."""
    return {"status": "ok", "model_loaded": True, "version": "1.0.0"}


@app.get("/stats")
async def get_stats() -> dict[str, int]:
    """Return in-memory prediction stats."""
    return dict(stats)


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)) -> dict[str, Any]:
    """Predict traffic state from an uploaded image.

    Args:
        file: Uploaded image.

    Returns:
        Prediction payload.

    Raises:
        HTTPException: If the image cannot be decoded.
    """
    start = time.perf_counter()
    try:
        data = await file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc
    pred = pipeline.predict_image(image)
    stats[pred["congestion_level"]] += 1
    return {
        "congestion_level": pred["congestion_level"],
        "confidence": pred["confidence"],
        "vehicle_count": pred["vehicle_count"],
        "vehicle_breakdown": pred["vehicle_breakdown"],
        "violations": pred["violations"],
        "processing_time_ms": round((time.perf_counter() - start) * 1000, 2),
    }
