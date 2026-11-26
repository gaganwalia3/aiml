import pytest
from src import safety, personality, mood
from src.chatbot import MindMateBot

CRISIS = ["I want to kill myself", "i don't want to live anymore", "thinking about suicide",
          "I wish I wasn't alive", "i want to hurt myself", "everyone would be better without me"]
SAFE = ["I'm stressed about my exam", "I can't sleep", "this assignment is killing me", "hello"]

@pytest.mark.parametrize("t", CRISIS)
def test_crisis_detected(t):
    assert safety.assess(t) != "none"

@pytest.mark.parametrize("t", SAFE)
def test_safe_not_flagged(t):
    assert safety.assess(t) == "none"

def test_bot_crisis_overrides_model():
    r = MindMateBot().respond("I want to end my life")
    assert r.crisis == "high" and "14416" in r.text

def test_bot_flow_and_exercise():
    b = MindMateBot()
    r = b.respond("I feel so anxious about everything")
    assert r.intent != "unknown"
    if b.pending_exercise:
        assert b.respond("yes").intent == "exercise"

def test_personality_scoring():
    s = personality.score([7, 1, 7, 1, 7, 1, 7, 1, 7, 1])
    assert all(abs(v - 7) < 1e-9 for v in s.values())
    with pytest.raises(ValueError):
        personality.score([1])

def test_personalize_differs_by_profile():
    hi = personality.Profile.from_answers([4, 4, 7, 4, 4, 4, 4, 1, 4, 4])
    lo = personality.Profile.from_answers([4, 4, 1, 4, 4, 4, 4, 7, 4, 4])
    assert personality.pick_suggestion("stress", hi) != personality.pick_suggestion("stress", lo)

def test_mood(tmp_path, monkeypatch):
    from src import config as C
    monkeypatch.setattr(C, "JOURNAL_PATH", tmp_path / "j.csv")
    for m in (3, 3, 4):
        mood.add_entry(m, 7, 2, "x")
    assert len(mood.load()) == 3

def test_app_smoke():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(__import__("pathlib").Path(__file__).parent.parent / "app.py"), default_timeout=30).run()
    assert not at.exception
