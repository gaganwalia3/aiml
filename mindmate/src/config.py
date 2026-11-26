"""Central configuration: paths, feature lists, thresholds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
for _d in (DATA_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(exist_ok=True)

TABULAR_PATH = DATA_DIR / "student_depression.csv"          # real Kaggle CSV goes here
INTENTS_PATH = DATA_DIR / "mental_health_intents.json"      # real Kaggle JSON goes here
JOURNAL_PATH = DATA_DIR / "journal.csv"                     # mood journal (local only)
HELPLINES_PATH = DATA_DIR / "helplines.json"                # editable list of crisis resources

TABULAR_MODEL_PATH = MODEL_DIR / "risk_model.joblib"
INTENT_MODEL_PATH = MODEL_DIR / "intent_model.joblib"

TARGET = "Depression"

# Features the risk model is allowed to use. These are the things a student can
# easily self-report in the app. City/Degree are excluded (too many sparse
# categories, little use for a self-check form).
NUMERIC_FEATURES = [
    "Age", "Academic Pressure", "CGPA", "Study Satisfaction",
    "Work/Study Hours", "Financial Stress",
]
CATEGORICAL_FEATURES = [
    "Gender", "Sleep Duration", "Dietary Habits", "Family History of Mental Illness",
]

# The "suicidal thoughts" column is a screening item, not a lifestyle factor. In the
# real Kaggle data it dominates the model. Keep it OFF by default so the risk check
# is based on modifiable lifestyle factors; flip to True to compare (see report notes).
USE_SUICIDAL_FEATURE = False
SUICIDAL_COL = "Have you ever had suicidal thoughts?"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Intent classifier: below this confidence the bot asks a clarifying question.
INTENT_CONFIDENCE_THRESHOLD = 0.30

# Risk thresholds on predicted probability of the "depressed" class.
RISK_LOW_MAX = 0.35
RISK_MODERATE_MAX = 0.65
