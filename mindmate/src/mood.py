"""Mood journal: simple CSV-backed log of daily mood, sleep, stress and a note.

Stored locally in data/journal.csv. Notes are screened by the safety layer in the app.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from . import config as C

COLUMNS = ["timestamp", "mood", "sleep_hours", "stress", "note"]


def load() -> pd.DataFrame:
    if not C.JOURNAL_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(C.JOURNAL_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", errors="coerce")
    return df.dropna(subset=["timestamp"])


def add_entry(mood: int, sleep_hours: float, stress: int, note: str = "") -> None:
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "mood": int(mood),
           "sleep_hours": float(sleep_hours), "stress": int(stress), "note": note.strip()}
    df = pd.concat([load(), pd.DataFrame([row])], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601").dt.strftime("%Y-%m-%dT%H:%M:%S")
    C.JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.JOURNAL_PATH, index=False)


def clear() -> None:
    if C.JOURNAL_PATH.exists():
        C.JOURNAL_PATH.unlink()


def insights(df: pd.DataFrame) -> list[str]:
    """Plain-language observations from the last 7 entries. Not a diagnosis."""
    out: list[str] = []
    if len(df) < 3:
        return ["Add a few more entries (3+) to see trends."]
    recent = df.sort_values("timestamp").tail(7)
    if recent["mood"].mean() <= 2.5:
        out.append("Your mood has been low recently. Consider talking to someone you trust or a counsellor.")
    if len(recent) >= 4:
        half = len(recent) // 2
        diff = recent["mood"].iloc[-half:].mean() - recent["mood"].iloc[:half].mean()
        if diff >= 0.7:
            out.append("Your mood is trending up. Nice.")
        elif diff <= -0.7:
            out.append("Your mood has dipped lately. Be kind to yourself and check in with someone.")
    if recent["sleep_hours"].mean() < 6:
        out.append(f"You've averaged {recent['sleep_hours'].mean():.1f}h of sleep, which is on the low side.")
    if recent["mood"].corr(recent["sleep_hours"]) > 0.5:
        out.append("On days you sleep more, your mood tends to be better.")
    if recent["mood"].corr(recent["stress"]) < -0.5:
        out.append("Higher-stress days tend to be lower-mood days for you.")
    return out or ["No notable patterns yet. Keep logging."]
