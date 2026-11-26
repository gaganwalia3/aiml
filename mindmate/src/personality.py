"""Personality module: 10-item Big Five quiz (TIPI format) -> response style.

The quiz uses the Ten-Item Personality Inventory *format* (Gosling, Rentfrow & Swann, 2003):
two statements per trait, answered 1-7, with reverse-scoring. TIPI is freely available for
research and educational use; cite it in your report.

The mapping from traits to response style is a *design heuristic* for this project. It is
not clinically validated, and users should be able to ignore or reset it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .coping import SUGGESTIONS

# (statement, trait, reverse_scored)
ITEMS = [
    ("I see myself as extraverted, enthusiastic.", "extraversion", False),
    ("I see myself as critical, quarrelsome.", "agreeableness", True),
    ("I see myself as dependable, self-disciplined.", "conscientiousness", False),
    ("I see myself as anxious, easily upset.", "emotional_stability", True),
    ("I see myself as open to new experiences, complex.", "openness", False),
    ("I see myself as reserved, quiet.", "extraversion", True),
    ("I see myself as sympathetic, warm.", "agreeableness", False),
    ("I see myself as disorganized, careless.", "conscientiousness", True),
    ("I see myself as calm, emotionally stable.", "emotional_stability", False),
    ("I see myself as conventional, uncreative.", "openness", True),
]
TRAITS = ["extraversion", "agreeableness", "conscientiousness", "emotional_stability", "openness"]
SCALE = {1: "Disagree strongly", 2: "Disagree moderately", 3: "Disagree a little",
         4: "Neither agree nor disagree", 5: "Agree a little", 6: "Agree moderately",
         7: "Agree strongly"}

LOW_MAX, HIGH_MIN = 3.5, 5.5


def score(answers: list[int]) -> dict[str, float]:
    """answers: ten integers 1-7 in ITEMS order -> trait scores on 1-7."""
    if len(answers) != len(ITEMS) or not all(1 <= a <= 7 for a in answers):
        raise ValueError("Need 10 answers between 1 and 7")
    buckets: dict[str, list[int]] = {t: [] for t in TRAITS}
    for a, (_, trait, rev) in zip(answers, ITEMS):
        buckets[trait].append(8 - a if rev else a)
    return {t: sum(v) / len(v) for t, v in buckets.items()}


def level(value: float) -> str:
    if value <= LOW_MAX:
        return "low"
    return "high" if value >= HIGH_MIN else "mid"


@dataclass
class Profile:
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def levels(self) -> dict[str, str]:
        return {t: level(v) for t, v in self.scores.items()}

    @property
    def has_data(self) -> bool:
        return bool(self.scores)

    @classmethod
    def from_answers(cls, answers: list[int]) -> "Profile":
        return cls(score(answers))

    def describe(self) -> str:
        if not self.has_data:
            return "No personality profile yet."
        names = {"extraversion": "Extraversion", "agreeableness": "Agreeableness",
                 "conscientiousness": "Conscientiousness",
                 "emotional_stability": "Emotional stability", "openness": "Openness"}
        return ", ".join(f"{names[t]}: {self.levels[t]} ({self.scores[t]:.1f}/7)" for t in TRAITS)


NEGATIVE_INTENTS = {"stress", "anxiety", "sadness", "loneliness", "self_esteem", "anger",
                    "family_pressure", "relationships", "sleep", "procrastination"}


def pick_suggestion(intent: str, profile: Profile | None) -> str | None:
    options = SUGGESTIONS.get(intent)
    if not options:
        return None
    levels = profile.levels if profile and profile.has_data else {}
    best, best_n = None, -1
    for req, text in options:
        n = len(req) if req and all(levels.get(t) == lv for t, lv in req.items()) else (0 if not req else -1)
        if n > best_n:
            best, best_n = text, n
    return best


def personalize(reply: str, intent: str, profile: Profile | None) -> str:
    """Adapt tone and add a trait-matched suggestion to a base reply."""
    if intent not in NEGATIVE_INTENTS:
        return reply
    levels = profile.levels if profile and profile.has_data else {}

    # emotionally sensitive users: lead with extra validation, keep it shorter
    if levels.get("emotional_stability") == "low":
        reply = "That sounds really hard, and it makes sense that it hurts. " + reply
    tip = pick_suggestion(intent, profile)
    if tip:
        reply += f"\n\n**An idea that might suit you:** {tip}"
    return reply
