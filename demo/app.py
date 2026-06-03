"""Streamlit dashboard for Gridlock AI."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["DISPLAY"] = ""

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.pipeline import TrafficInferencePipeline
from src.utils.visualize import draw_detections


def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(page_title="Gridlock AI", layout="wide")
    st.title("Gridlock AI - Bengaluru Traffic Intelligence")
    threshold = st.sidebar.slider("Confidence threshold", 0.30, 0.95, 0.50)
    source = st.sidebar.radio("Upload source", ["image", "video", "webcam"])
    pipeline = TrafficInferencePipeline()
    tabs = st.tabs(["Live Analysis", "Historical Trends", "Junction Map", "Model Info"])
    st.session_state.setdefault("history", [])
    with tabs[0]:
        uploaded = st.file_uploader("Upload", type=["jpg", "jpeg", "png"] if source == "image" else ["mp4", "avi", "mov"])
        if uploaded and source == "image":
            image = Image.open(uploaded).convert("RGB")
            pred = pipeline.predict_image(image)
            pred["detections"] = [d for d in pred["detections"] if d["confidence"] >= threshold]
            st.image(draw_detections(image, pred["detections"]), use_column_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Congestion Level", pred["congestion_level"])
            c2.metric("Vehicle Count", pred["vehicle_count"])
            c3.metric("Active Violations", len(pred["violations"]))
            st.session_state.history.append(pred)
    with tabs[1]:
        hist = st.session_state.history
        if hist:
            df = pd.DataFrame({"index": range(len(hist)), "congestion": [h["congestion_level"] for h in hist], "vehicles": [h["vehicle_count"] for h in hist]})
            st.plotly_chart(px.line(df, x="index", y="vehicles", color="congestion"), use_container_width=True)
            st.plotly_chart(px.histogram(df, x="congestion"), use_container_width=True)
        else:
            st.info("Run live analysis to populate trends.")
    with tabs[2]:
        url = os.environ.get("MAPMYINDIA_EMBED_URL", "https://maps.mapmyindia.com/")
        st.components.v1.iframe(url, height=520)
    with tabs[3]:
        st.write("YOLOv8 detector + EfficientNet-style congestion classifier + count-based ensemble.")
        st.write("Synthetic fallback keeps the demo and tests runnable before BTP data is released.")


if __name__ == "__main__":
    main()
