"""Gradio Spaces app for Gridlock AI."""

from __future__ import annotations

from src.inference.pipeline import TrafficInferencePipeline


def predict(image):
    """Predict a Gradio image."""
    pipeline = TrafficInferencePipeline()
    result = pipeline.predict_image(image)
    return result["congestion_level"], result["confidence"], result["vehicle_count"]


try:
    import gradio as gr

    demo = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs=["text", "number", "number"], title="Gridlock AI")
except Exception:
    demo = None


if __name__ == "__main__" and demo is not None:
    demo.launch()
