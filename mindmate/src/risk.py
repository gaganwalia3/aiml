"""Wellbeing risk check using the tabular model. A screening aid, NOT a diagnosis."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import config as C

_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(C.TABULAR_MODEL_PATH)
    return _bundle


def is_synthetic() -> bool:
    return bool(_load().get("synthetic", False))


def defaults() -> dict:
    return _load().get("defaults", {})


def predict(inputs: dict) -> dict:
    b = _load()
    row = pd.DataFrame([{c: inputs.get(c, np.nan) for c in b["num_cols"] + b["cat_cols"]}])
    p = float(b["model"].predict_proba(row)[0, 1])
    level = "low" if p <= C.RISK_LOW_MAX else "moderate" if p <= C.RISK_MODERATE_MAX else "elevated"
    return {"probability": p, "level": level, "model": b["name"], "synthetic": is_synthetic()}


ADVICE = {
    "low": "Your answers show few of the patterns linked with distress in the training data. Keep up healthy routines.",
    "moderate": "Some of your answers resemble patterns linked with distress. Consider small changes (sleep, workload) and talking to someone you trust.",
    "elevated": "Several of your answers resemble patterns linked with distress. This is not a diagnosis, but please consider talking to a counsellor or doctor.",
}
