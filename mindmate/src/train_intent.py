"""Train the intent classifier that drives the chatbot's coping responses.

Run:  python -m src.train_intent
Outputs: models/intent_model.joblib, reports/intent_*.png / intent_metrics.json
"""
from __future__ import annotations

import json
import re

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from . import config as C


def normalise(text: str) -> str:
    text = text.lower().replace("’", "'")
    return re.sub(r"[^a-z0-9\s']", " ", text).strip()


def load_intents() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    data = json.loads(C.INTENTS_PATH.read_text())
    rows, responses = [], {}
    for it in data["intents"]:
        responses[it["tag"]] = it.get("responses", [])
        rows += [{"pattern": normalise(p), "tag": it["tag"]} for p in it["patterns"]]
    df = pd.DataFrame(rows).drop_duplicates("pattern").reset_index(drop=True)
    return df, responses


def make_features() -> FeatureUnion:
    # word n-grams capture meaning, char n-grams make it robust to typos ("anxios")
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)),
    ])


def candidates() -> dict:
    rs = C.RANDOM_STATE
    return {
        "Logistic Regression": LogisticRegression(C=15, max_iter=3000, class_weight="balanced", random_state=rs),
        "Linear SVM": LinearSVC(C=1.0, class_weight="balanced", random_state=rs),
        "Naive Bayes": MultinomialNB(alpha=0.3),
    }


def main() -> dict:
    df, responses = load_intents()
    X, y = df["pattern"], df["tag"]
    counts = y.value_counts()
    print(f"{len(df)} patterns, {y.nunique()} intents (smallest class: {counts.min()})")

    n_splits = int(min(5, counts.min()))
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=C.RANDOM_STATE)

    results, preds = [], {}
    for name, clf in candidates().items():
        pipe = Pipeline([("feat", make_features()), ("clf", clf)])
        pred = cross_val_predict(pipe, X, y, cv=cv)
        preds[name] = pred
        results.append({"model": name, "accuracy": float((pred == y).mean()),
                        "macro_f1": float(f1_score(y, pred, average="macro"))})
        print(f"  {name:<20} acc={results[-1]['accuracy']:.3f}  macro-F1={results[-1]['macro_f1']:.3f}")

    best = max(results, key=lambda r: r["macro_f1"])["model"]
    # The chatbot needs probabilities for its confidence threshold -> keep LR as deployed
    # model unless it is clearly worse; report which one scored best.
    deploy = "Logistic Regression"
    print(f"\nBest CV model: {best}; deploying {deploy} (needs predict_proba for confidence).")
    print(classification_report(y, preds[deploy], zero_division=0))

    labels = sorted(y.unique())
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(y, preds[deploy], labels=labels, xticks_rotation=60,
                                            cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"Intent classifier ({deploy}), cross-validated", fontsize=11)
    fig.tight_layout(); fig.savefig(C.REPORT_DIR / "intent_confusion_matrix.png", dpi=140); plt.close(fig)

    final = Pipeline([("feat", make_features()), ("clf", candidates()[deploy])]).fit(X, y)
    joblib.dump({"pipeline": final, "responses": responses, "classes": list(final.classes_)},
                C.INTENT_MODEL_PATH)
    metrics = {"n_patterns": int(len(df)), "n_intents": int(y.nunique()), "cv_folds": n_splits,
               "results": results, "deployed": deploy,
               "caveat": "Small placeholder dataset; re-evaluate on the real Kaggle intents file."}
    (C.REPORT_DIR / "intent_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model -> {C.INTENT_MODEL_PATH}")
    return metrics


if __name__ == "__main__":
    main()
