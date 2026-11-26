# MindMate: AI-Driven Personality-Aware Mental Wellness Assistant for Students

Chitkara University, AIML (24CSE0316). CE-2 / final project.

> **Disclaimer:** educational project, not a medical device or therapist. Crisis: Tele-MANAS 14416, emergency 112.

## What it contains
| Module | File | What it does |
|---|---|---|
| Risk model | `src/train_tabular.py`, `src/risk.py` | Compares LR/RF/SVM/GB (5-fold CV + GridSearch) on student depression survey data |
| Intent classifier | `src/train_intent.py` | TF-IDF (word+char) + Logistic Regression on intents JSON |
| Safety layer | `src/safety.py` | Rule-based crisis detection, runs before any ML, editable helplines (`data/helplines.json`) |
| Personality | `src/personality.py`, `src/coping.py` | 10-item Big Five (TIPI format) -> tailored suggestions and tone |
| Chatbot | `src/chatbot.py` | Safety -> intent -> personalised reply + guided exercise |
| Mood journal | `src/mood.py` | Local CSV log, trends |
| App | `app.py` | Streamlit UI (Chat, Risk check, Personality, Journal, About) |

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
pytest -q tests
```

## IMPORTANT: use the real datasets
This package ships with **synthetic / placeholder data** so it runs out of the box. Metrics produced from it are **not real results**. Before reporting anything:
1. Download Student Depression Dataset (Kaggle, hopesb) -> `data/student_depression.csv`
2. Download the Mental Health Chatbot intents (Kaggle, elvis23) -> `data/mental_health_intents.json`
3. Retrain: `python -m src.train_tabular` and `python -m src.train_intent`
4. Reports/plots appear in `reports/`.

The "Have you ever had suicidal thoughts?" column is excluded by default (`USE_SUICIDAL_FEATURE=False` in `src/config.py`) because it would dominate the model and makes the tool circular. Edit `data/helplines.json` (add your campus counsellor) and verify numbers before deploying.

## Limitations (put these in your report)
- Survey-based risk model: screening only, not diagnosis; dataset biases (single country, self-report).
- Small intent dataset; unknown inputs get a clarifying fallback (confidence < 0.30).
- Rule-based crisis detection favours false alarms; it can miss indirect or non-English phrasing.
- Personality->style mapping is a design heuristic, not clinically validated.
- Journal is stored unencrypted locally; add auth/encryption before real deployment.
