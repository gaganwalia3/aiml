"""MindMate chatbot: safety layer -> intent model -> personality-aware reply."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

import joblib

from . import config as C
from . import safety
from .coping import EXERCISES, INTENT_EXERCISE
from .personality import Profile, personalize
from .train_intent import normalise

YES = {"yes", "yeah", "yep", "sure", "ok", "okay", "please", "yes please", "y", "go ahead"}
NO = {"no", "nope", "nah", "not now", "maybe later", "later", "no thanks", "skip"}

FALLBACKS = [
    "I'm listening. Even if it's hard to put into words, feel free to share a little more—is this mostly about stress, sleep, mood, focus, or relationships?",
    "Thanks for sharing that with me. Tell me a bit more about what's been weighing on you today, or pick a topic like stress, anxiety, or sleep.",
    "I hear you, and I want to respond helpfully. Could you give me one more sentence about what's happening right now?",
]

# Lightweight keyword rules used only when the intent classifier's confidence is below
# the threshold, so natural student phrasing still routes to a helpful topic.
_KEYWORD_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("loneliness", (r"\blonely\b", r"\balone\b", r"\bisolated\b", r"\bno friends\b", r"\bleft out\b", r"\bmiss home\b", r"\bhomesick\b")),
    ("anxiety", (r"\banxious\b", r"\banxiety\b", r"\bpanic\b", r"\boverthink", r"\bworried\b", r"\bworrying\b", r"\bnervous\b", r"\bscared\b")),
    ("stress", (r"\bstress", r"\boverwhelm", r"\bdeadline", r"\bburnout\b", r"\bburnt out\b", r"\bexam", r"\bpressure\b", r"\bworkload\b")),
    ("sleep", (r"\bsleep", r"\binsomnia\b", r"\bawake\b", r"\bnightmare", r"\bcant rest\b", r"\btired\b", r"\bexhausted\b")),
    ("procrastination", (r"\bprocrastinat", r"\bfocus\b", r"\bconcentrat", r"\bmotivat", r"\bput(ting)? off\b", r"\bcant start\b", r"\blazy\b")),
    ("sadness", (r"\bsad\b", r"\bcry", r"\bdown\b", r"\bempty\b", r"\bmiserable\b", r"\bdepress", r"\bheartbroken\b", r"\bheavy\b")),
    ("self_esteem", (r"\bworthless\b", r"\bfailure\b", r"\bfailed\b", r"\bnot good enough\b", r"\bhate myself\b", r"\binsecure\b", r"\bdumb\b")),
    ("anger", (r"\bangry\b", r"\bmad\b", r"\bfurious\b", r"\bfrustrat", r"\birritat", r"\bannoyed\b")),
    ("family_pressure", (r"\bparent", r"\bmom\b", r"\bmother\b", r"\bdad\b", r"\bfather\b", r"\bfamily\b", r"\bexpectation")),
    ("relationships", (r"\bfriend", r"\bbreakup\b", r"\bboyfriend\b", r"\bgirlfriend\b", r"\bpartner\b", r"\bargument\b", r"\bfight\b", r"\bcrush\b")),
    ("professional_help", (r"\btherap", r"\bcounsell?or\b", r"\bpsychologist\b", r"\bdoctor\b", r"\bprofessional help\b")),
    ("positive_mood", (r"\bhappy\b", r"\bgreat\b", r"\bawesome\b", r"\bfeeling good\b", r"\bfeeling better\b", r"\bcalm\b", r"\bproud\b")),
    ("about_bot", (r"\bwho are you\b", r"\bwhat are you\b", r"\bwhat can you do\b", r"\bhow do you work\b", r"\bare you real\b")),
    ("thanks", (r"\bthank", r"\bappreciate\b", r"\bhelped\b")),
    ("goodbye", (r"\bbye\b", r"\bgoodbye\b", r"\bsee you\b", r"\bgood night\b", r"\bgotta go\b")),
    ("greeting", (r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\bnamaste\b", r"\bgood morning\b", r"\bgood evening\b")),
]
_COMPILED_KEYWORDS = [(tag, [re.compile(p) for p in pats]) for tag, pats in _KEYWORD_RULES]


def _keyword_intent(norm_text: str) -> str | None:
    for tag, patterns in _COMPILED_KEYWORDS:
        if any(p.search(norm_text) for p in patterns):
            return tag
    return None


@dataclass
class Reply:
    text: str
    intent: str = "none"
    confidence: float = 0.0
    crisis: str = "none"


@dataclass
class MindMateBot:
    profile: Profile | None = None
    pending_exercise: str | None = None
    history: list = field(default_factory=list)
    _bundle: dict | None = None

    def _model(self):
        if self._bundle is None:
            self._bundle = joblib.load(C.INTENT_MODEL_PATH)
        return self._bundle

    def respond(self, message: str) -> Reply:
        message = message.strip()
        if not message:
            return Reply("I'm here whenever you're ready to talk.")

        # 1) Safety always first
        level = safety.assess(message)
        if level != "none":
            self.pending_exercise = None
            return Reply(safety.crisis_response(level), intent="crisis", confidence=1.0, crisis=level)

        norm = normalise(message)

        # 2) Handle response to an offered exercise
        if self.pending_exercise:
            if norm in YES:
                ex = EXERCISES[self.pending_exercise]["text"]
                self.pending_exercise = None
                return Reply(ex, intent="exercise", confidence=1.0)
            if norm in NO:
                self.pending_exercise = None
                return Reply(
                    "No problem at all—we don't have to do an exercise. Feel free to keep sharing whatever is on your mind.",
                    intent="acknowledgement",
                    confidence=1.0,
                )
        self.pending_exercise = None

        # 3) Intent classification
        b = self._model()
        pipe = b["pipeline"]
        proba = pipe.predict_proba([norm])[0]
        order = proba.argsort()[::-1]
        i1 = int(order[0])
        i2 = int(order[1]) if len(order) > 1 else i1
        intent, conf = str(pipe.classes_[i1]), float(proba[i1])
        second_conf = float(proba[i2])

        # 4) Low-confidence handling: try keyword match, then clear-margin match, else warm fallback
        if conf < C.INTENT_CONFIDENCE_THRESHOLD:
            kw_intent = _keyword_intent(norm)
            if kw_intent and kw_intent in b["responses"]:
                intent = kw_intent
                conf = max(conf, C.INTENT_CONFIDENCE_THRESHOLD)
            elif conf >= 0.20 and (second_conf == 0 or conf >= 1.45 * second_conf) and intent != "crisis":
                pass  # clear winner over second-best class
            else:
                return Reply(random.choice(FALLBACKS), intent="unknown", confidence=conf)

        if intent == "crisis":  # classifier flagged crisis the rules missed: be cautious
            return Reply(safety.crisis_response("elevated"), intent="crisis", confidence=conf, crisis="elevated")

        responses = b["responses"].get(intent) or FALLBACKS
        text = personalize(random.choice(responses), intent, self.profile)

        ex_key = INTENT_EXERCISE.get(intent)
        if ex_key:
            text += "\n\n" + EXERCISES[ex_key]["offer"]
            self.pending_exercise = ex_key
        return Reply(text, intent=intent, confidence=conf)
