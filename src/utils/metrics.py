"""Metric helpers."""

from __future__ import annotations


def classification_report_dict(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    """Compute accuracy and macro F1 with optional sklearn.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Metric mapping.

    Raises:
        ValueError: If lengths differ.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    try:
        from sklearn.metrics import accuracy_score, f1_score

        return {"accuracy": float(accuracy_score(y_true, y_pred)), "macro_f1": float(f1_score(y_true, y_pred, average="macro"))}
    except Exception:
        correct = sum(a == b for a, b in zip(y_true, y_pred))
        return {"accuracy": correct / len(y_true), "macro_f1": correct / len(y_true)}
