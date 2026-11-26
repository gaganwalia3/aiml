"""Train and compare depression-risk classifiers on the tabular dataset.

Run:  python -m src.train_tabular
Outputs: models/risk_model.joblib, reports/tabular_*.png, reports/tabular_metrics.json
"""
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             classification_report, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from . import config as C
from .data_prep import feature_lists, get_clean_data


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)])


def candidate_models() -> dict:
    rs = C.RANDOM_STATE
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=rs),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                random_state=rs, n_jobs=-1),
        "SVM (RBF)": CalibratedClassifierCV(SVC(class_weight="balanced", random_state=rs),
                                            cv=3, ensemble=False),
        "Gradient Boosting": GradientBoostingClassifier(random_state=rs),
    }


PARAM_GRIDS = {
    "Logistic Regression": {"clf__C": [0.1, 1, 10]},
    "Random Forest": {"clf__max_depth": [None, 8, 14], "clf__min_samples_leaf": [1, 3, 6]},
    "SVM (RBF)": {"clf__estimator__C": [0.5, 1, 4], "clf__estimator__gamma": ["scale", 0.05]},
    "Gradient Boosting": {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
                          "clf__max_depth": [2, 3]},
}


def _scores(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def main() -> dict:
    df, synthetic = get_clean_data()
    num_cols, cat_cols = feature_lists()
    X, y = df[num_cols + cat_cols], df[C.TARGET]
    tag = " [SYNTHETIC DATA]" if synthetic else ""
    print(f"Data: {df.shape}, depressed share = {y.mean():.2%}{tag}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)
    cv = StratifiedKFold(5, shuffle=True, random_state=C.RANDOM_STATE)

    # ---- 1. baseline comparison with cross-validation --------------------------------
    rows = []
    for name, model in candidate_models().items():
        pipe = Pipeline([("pre", build_preprocessor(num_cols, cat_cols)), ("clf", model)])
        res = cross_validate(pipe, X_tr, y_tr, cv=cv, n_jobs=-1,
                             scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
        rows.append({"model": name, **{m: res[f"test_{m}"].mean()
                                       for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]}})
        print(f"  CV  {name:<20} F1={rows[-1]['f1']:.3f}  recall={rows[-1]['recall']:.3f}  AUC={rows[-1]['roc_auc']:.3f}")
    cv_df = pd.DataFrame(rows).set_index("model").round(4)

    # ---- 2. hyper-parameter tuning (optimise F1; recall is tracked separately) -------
    tuned, tune_rows = {}, []
    for name, model in candidate_models().items():
        pipe = Pipeline([("pre", build_preprocessor(num_cols, cat_cols)), ("clf", model)])
        gs = GridSearchCV(pipe, PARAM_GRIDS[name], scoring="f1", cv=cv, n_jobs=-1).fit(X_tr, y_tr)
        tuned[name] = gs.best_estimator_
        tune_rows.append({"model": name, "best_cv_f1": round(gs.best_score_, 4),
                          "best_params": {k.replace("clf__estimator__", "").replace("clf__", ""): v for k, v in gs.best_params_.items()}})
        print(f"  TUNE {name:<20} best CV F1={gs.best_score_:.3f}  {gs.best_params_}")

    # ---- 3. held-out test evaluation -------------------------------------------------
    test_rows, probas = [], {}
    for name, est in tuned.items():
        proba = est.predict_proba(X_te)[:, 1]
        probas[name] = proba
        test_rows.append({"model": name, **_scores(y_te, (proba >= 0.5).astype(int), proba)})
    test_df = pd.DataFrame(test_rows).set_index("model").round(4)
    best_name = max(tune_rows, key=lambda r: r["best_cv_f1"])["model"]
    best = tuned[best_name]
    print(f"\nBest model (CV F1): {best_name}")
    print(test_df.to_string())
    print("\n" + classification_report(y_te, (probas[best_name] >= 0.5).astype(int),
                                        target_names=["Not depressed", "Depressed"]))

    # ---- 4. plots --------------------------------------------------------------------
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_te, (probas[best_name] >= 0.5).astype(int), display_labels=["No", "Yes"],
        cmap="Blues", ax=ax)
    ax.set_title(f"Confusion matrix - {best_name}{tag}", fontsize=10)
    fig.tight_layout(); fig.savefig(C.REPORT_DIR / "tabular_confusion_matrix.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in probas.items():
        RocCurveDisplay.from_predictions(y_te, p, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title(f"ROC curves (held-out test){tag}", fontsize=10)
    fig.tight_layout(); fig.savefig(C.REPORT_DIR / "tabular_roc.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    test_df[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot.bar(ax=ax, rot=15)
    ax.set_ylim(0, 1); ax.set_title(f"Model comparison (test set){tag}", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(C.REPORT_DIR / "tabular_model_comparison.png", dpi=150); plt.close(fig)

    imp = permutation_importance(best, X_te, y_te, scoring="f1", n_repeats=15,
                                 random_state=C.RANDOM_STATE, n_jobs=-1)
    imp_df = (pd.DataFrame({"feature": X_te.columns, "importance": imp.importances_mean,
                            "std": imp.importances_std})
              .sort_values("importance"))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp_df["feature"], imp_df["importance"], xerr=imp_df["std"], color="#4C78A8")
    ax.set_xlabel("Drop in F1 when feature is shuffled")
    ax.set_title(f"Permutation importance - {best_name}{tag}", fontsize=10)
    fig.tight_layout(); fig.savefig(C.REPORT_DIR / "tabular_feature_importance.png", dpi=150); plt.close(fig)

    # ---- 5. save ---------------------------------------------------------------------
    joblib.dump({"model": best, "name": best_name, "num_cols": num_cols, "cat_cols": cat_cols,
                 "synthetic": synthetic, "defaults": X_tr.median(numeric_only=True).to_dict()},
                C.TABULAR_MODEL_PATH)
    metrics = {
        "synthetic_data": synthetic, "n_rows": int(len(df)), "positive_rate": float(y.mean()),
        "best_model": best_name, "cv": cv_df.reset_index().to_dict("records"),
        "tuning": tune_rows, "test": test_df.reset_index().to_dict("records"),
        "feature_importance": imp_df.sort_values("importance", ascending=False)
                                    .round(4).to_dict("records"),
        "uses_suicidal_feature": C.USE_SUICIDAL_FEATURE,
    }
    (C.REPORT_DIR / "tabular_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    cv_df.to_csv(C.REPORT_DIR / "tabular_cv_results.csv")
    test_df.to_csv(C.REPORT_DIR / "tabular_test_results.csv")
    print(f"\nSaved model -> {C.TABULAR_MODEL_PATH}")
    return metrics


if __name__ == "__main__":
    main()
