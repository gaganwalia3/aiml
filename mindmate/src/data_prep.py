"""Load, clean and prepare the tabular (student depression) dataset.

If data/student_depression.csv is missing, a synthetic stand-in with the same schema
and *some built-in signal* is generated so the whole pipeline can run. Metrics from
synthetic data are NOT real results -- replace with the Kaggle CSV before reporting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

SLEEP_ORDER = ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"]


# ----------------------------------------------------------------------------
# synthetic fallback
# ----------------------------------------------------------------------------
def make_synthetic(n: int = 2000, seed: int = C.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Gender": rng.choice(["Male", "Female"], n),
        "Age": rng.integers(18, 31, n),
        "City": rng.choice(["Delhi", "Ludhiana", "Mumbai", "Bangalore", "Chandigarh"], n),
        "Academic Pressure": rng.integers(0, 6, n).astype(float),
        "CGPA": np.round(rng.uniform(5.0, 10.0, n), 2),
        "Study Satisfaction": rng.integers(0, 6, n).astype(float),
        "Sleep Duration": rng.choice(SLEEP_ORDER, n, p=[0.25, 0.3, 0.3, 0.15]),
        "Dietary Habits": rng.choice(["Healthy", "Moderate", "Unhealthy"], n),
        "Degree": rng.choice(["B.Tech", "BSc", "BA", "MBA", "MSc"], n),
        "Have you ever had suicidal thoughts?": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "Work/Study Hours": rng.integers(0, 13, n),
        "Financial Stress": rng.integers(1, 6, n).astype(float),
        "Family History of Mental Illness": rng.choice(["Yes", "No"], n, p=[0.25, 0.75]),
    })
    # latent risk score -> label (so models have something real to learn)
    z = (
        0.55 * df["Academic Pressure"]
        + 0.45 * df["Financial Stress"]
        - 0.35 * df["Study Satisfaction"]
        + 0.10 * df["Work/Study Hours"]
        + 0.9 * (df["Sleep Duration"] == "Less than 5 hours")
        + 0.4 * (df["Sleep Duration"] == "5-6 hours")
        + 0.6 * (df["Dietary Habits"] == "Unhealthy")
        + 0.5 * (df["Family History of Mental Illness"] == "Yes")
        + 1.5 * (df["Have you ever had suicidal thoughts?"] == "Yes")
        - 0.05 * (df["Age"] - 24)
    )
    z = (z - z.mean()) / z.std()
    p = 1 / (1 + np.exp(-(1.4 * z + rng.normal(0, 0.8, n))))
    df["Depression"] = (rng.random(n) < p).astype(int)
    # realistic mess
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)
    for col in ["CGPA", "Academic Pressure", "Sleep Duration"]:
        idx = rng.choice(df.index, size=int(0.02 * len(df)), replace=False)
        df.loc[idx, col] = np.nan
    return df


# ----------------------------------------------------------------------------
# loading + cleaning
# ----------------------------------------------------------------------------
def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    # The Kaggle file writes "Have you ever had suicidal thoughts ?" (space before ?)
    df = df.copy()
    df.columns = [c.strip().replace(" ?", "?") for c in df.columns]
    return df


def load_raw() -> tuple[pd.DataFrame, bool]:
    """Return (dataframe, is_synthetic)."""
    if C.TABULAR_PATH.exists():
        return _normalise_columns(pd.read_csv(C.TABULAR_PATH)), False
    return make_synthetic(), True


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Dedupe, tidy strings, coerce numerics, keep only columns we model on."""
    df = _normalise_columns(df).drop_duplicates().reset_index(drop=True)

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip().str.strip("'\"")

    # real file has quoted sleep labels and an 'Others' bucket; map '?' etc. to NaN
    if "Sleep Duration" in df:
        df["Sleep Duration"] = df["Sleep Duration"].replace({"Others": pd.NA})
    if "Gender" in df:
        df["Gender"] = df["Gender"].str.title()

    for col in C.NUMERIC_FEATURES:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep = C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES + [C.TARGET]
    if C.USE_SUICIDAL_FEATURE and C.SUICIDAL_COL in df:
        keep.append(C.SUICIDAL_COL)
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in data: {missing}. Check the CSV header.")
    df = df[keep].copy()
    # sklearn imputers need plain object columns with np.nan (not pandas' pd.NA)
    for col in df.columns:
        if col not in C.NUMERIC_FEATURES and col != C.TARGET:
            df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    df[C.TARGET] = pd.to_numeric(df[C.TARGET], errors="coerce")
    return df.dropna(subset=[C.TARGET]).astype({C.TARGET: int})


def get_clean_data() -> tuple[pd.DataFrame, bool]:
    raw, synthetic = load_raw()
    return clean(raw), synthetic


def feature_lists() -> tuple[list[str], list[str]]:
    cat = list(C.CATEGORICAL_FEATURES)
    if C.USE_SUICIDAL_FEATURE:
        cat.append(C.SUICIDAL_COL)
    return list(C.NUMERIC_FEATURES), cat
