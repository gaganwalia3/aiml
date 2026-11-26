"""Crisis-detection layer. Runs BEFORE any ML model and cannot be overridden by it.

Deliberately rule-based: for self-harm language we prefer false alarms over misses,
and rules are transparent and testable. Levels:
    "high"     explicit suicidal / self-harm intent  -> crisis message + helplines
    "elevated" hopelessness / burden language        -> gentle check-in + helplines
    "none"
"""
from __future__ import annotations

import json
import re

from . import config as C

_HIGH = [
    r"\b(kill|end|take)\s+(my\s+own\s+life|my\s+life|myself)\b",
    r"\bsuicid(e|al)\b",
    r"\b(want|wanna|going|plan(ning)?|thinking(\s+of|\s+about)?)\s+(to\s+)?(die|end\s+it(\s+all)?)\b",
    r"\bwant\s+to\s+die\b",
    r"\bdon'?t\s+want\s+to\s+(live|be\s+alive|exist)(\s+anymore|\s+any\s+more)?\b",
    r"\b(hurt|harm|cut)\s+myself\b",
    r"\bself[- ]?harm\b",
    r"\bwish\s+i\s+(was|were)n'?t\s+(alive|here|born)\b",
    r"\bwish\s+i\s+(was|were)\s+dead\b",
    r"\b(better|easier)\s+off\s+(dead|without\s+me)\b",
    r"\blife\s+(is\s+)?(not|isn'?t)\s+worth\s+living\b",
    r"\bno\s+reason\s+to\s+(live|go\s+on)\b",
    r"\bend\s+it\s+all\b",
    r"\boverdose\b",
]
_ELEVATED = [
    r"\bno\s+(point|use)\s+(in\s+)?(living|going\s+on|anything)\b",
    r"\bcan'?t\s+(go\s+on|take\s+(it|this)\s+anymore|do\s+this\s+anymore)\b",
    r"\bgive\s+up\s+on\s+(life|everything)\b",
    r"\bi'?m\s+a\s+burden\b",
    r"\beveryone\s+(would\s+be\s+)?better\s+without\s+me\b",
    r"\bhopeless\b",
    r"\bdisappear\s+forever\b",
    r"\bnothing\s+matters\s+anymore\b",
]
_HIGH_RE = [re.compile(p, re.I) for p in _HIGH]
_ELEV_RE = [re.compile(p, re.I) for p in _ELEVATED]


def assess(text: str) -> str:
    """Return 'high', 'elevated' or 'none'."""
    t = text.lower().replace("’", "'")
    if any(r.search(t) for r in _HIGH_RE):
        return "high"
    if any(r.search(t) for r in _ELEV_RE):
        return "elevated"
    return "none"


def load_helplines() -> list[dict]:
    try:
        return json.loads(C.HELPLINES_PATH.read_text())["helplines"]
    except (OSError, KeyError, ValueError):
        return [{"name": "Tele-MANAS (India, free, 24x7)", "phone": "14416 or 1800-891-4416"},
                {"name": "Emergency services (India)", "phone": "112"}]


def _helpline_block() -> str:
    return "\n".join(f"- **{h['name']}**: {h['phone']}" for h in load_helplines())


def crisis_response(level: str) -> str:
    if level == "high":
        return (
            "I'm really glad you told me, and I'm worried about you. What you're feeling matters, "
            "and you don't have to face it alone.\n\n"
            "**Please reach out to someone right now:**\n" + _helpline_block() + "\n\n"
            "If you are in immediate danger, call 112 or go to the nearest hospital. "
            "If you can, tell someone near you (a friend, warden, family member) so you're not alone tonight.\n\n"
            "I'm an AI and can't replace a person who can really be there with you. "
            "I'll stay here if you'd like to keep talking, but please contact one of these as well."
        )
    return (
        "It sounds like things feel really heavy right now, and I'm taking that seriously. "
        "Would you like to tell me more about what's going on?\n\n"
        "If these feelings get stronger, or you ever think about hurting yourself, please contact:\n"
        + _helpline_block()
    )
