"""
Layer 3 — Predictive Model
SmartDischarge: 30-Day Hospital Readmission Prediction

Production ML pipeline:
  - XGBoost classifier with calibrated probabilities (Platt scaling)
  - SHAP explainability — global + per-prediction top-3 risk drivers
  - Rigorous evaluation: AUROC, AUPRC, F1, calibration curve
  - Fairness audit: AUROC by race and age group
  - 3-tier risk routing: High / Medium / Low
  - Model artefacts saved for API serving

Data Scientist role signal: evaluation rigour, explainability, fairness,
calibration — not just accuracy.
"""

import json
import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

warnings.filterwarnings("ignore")

log = logging.getLogger("smartdischarge.model")

MODEL_DIR  = Path(__file__).parents[2] / "data" / "processed"
REPORT_DIR = Path(__file__).parents[2] / "docs"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ── Risk tier thresholds (calibrated on validation set) ──────────────────────
THRESHOLD_HIGH   = 0.35   # ≥ 35% predicted prob → HIGH risk
THRESHOLD_MEDIUM = 0.15   # 15-35% → MEDIUM risk; <15% → LOW


def train_model(X_train, y_train):
    """Train XGBoost with class balancing, then apply Platt scaling calibration."""
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    log.info("Class imbalance ratio (scale_pos_weight): %.2f", scale_pos)

    base_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        min_child_weight=5,
        gamma=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    # Calibrated wrapper — ensures probabilities are well-calibrated (not just ranked)
    model = CalibratedClassifierCV(base_model, cv=3, method="sigmoid")
    model.fit(X_train, y_train)
    log.info("Model trained and calibrated")
    return model


def evaluate(model, X_test, y_test, feature_names) -> dict:
    """Comprehensive evaluation: AUROC, AUPRC, calibration, classification report."""
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= THRESHOLD_HIGH).astype(int)

    auroc  = roc_auc_score(y_test, proba)
    auprc  = average_precision_score(y_test, proba)
    report = classification_report(y_test, preds, output_dict=True)

    log.info("AUROC: %.4f  |  AUPRC: %.4f", auroc, auprc)

    # Calibration
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
    calibration_error = float(np.mean(np.abs(frac_pos - mean_pred)))
    log.info("Mean calibration error: %.4f", calibration_error)

    # ROC curve points (for dashboard)
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_points = {"fpr": fpr.tolist()[::10], "tpr": tpr.tolist()[::10]}

    return {
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "calibration_error": round(calibration_error, 4),
        "classification_report": report,
        "roc_curve": roc_points,
        "n_test": int(len(y_test)),
        "n_positive": int(y_test.sum()),
        "positive_rate_pct": round(y_test.mean() * 100, 2),
    }


def fairness_audit(model, X_test, y_test, feat_test) -> dict:
    """
    AUROC by demographic subgroup — race and age band.
    Flags if any group's AUROC drops > 0.05 below overall.
    """
    proba = model.predict_proba(X_test)[:, 1]
    overall = roc_auc_score(y_test, proba)
    results = {"overall_auroc": round(overall, 4), "subgroups": {}}

    for col in ["race", "age"]:
        if col not in feat_test.columns:
            continue
        for group in feat_test[col].dropna().unique():
            mask = feat_test[col] == group
            if mask.sum() < 50 or y_test[mask].sum() < 10:
                continue
            try:
                grp_auroc = roc_auc_score(y_test[mask], proba[mask])
                gap = overall - grp_auroc
                results["subgroups"][f"{col}={group}"] = {
                    "auroc": round(grp_auroc, 4),
                    "gap_from_overall": round(gap, 4),
                    "n": int(mask.sum()),
                    "flag": gap > 0.05,
                }
            except Exception:
                pass

    flagged = [k for k, v in results["subgroups"].items() if v.get("flag")]
    results["flagged_groups"] = flagged
    if flagged:
        log.warning("Fairness flags (AUROC gap > 0.05): %s", flagged)
    else:
        log.info("Fairness audit PASSED — no group AUROC gap > 0.05")
    return results


def compute_shap(model, X_train, X_test, feature_names):
    """
    Compute SHAP values on base XGBoost estimator.
    Returns global importance + function for per-prediction explanations.
    """
    # Extract base estimator from calibrated wrapper
    base = model.calibrated_classifiers_[0].estimator
    explainer = shap.TreeExplainer(base)
    shap_values = explainer.shap_values(X_test)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    log.info("Top 5 SHAP features:\n%s",
             importance_df.head(5).to_string(index=False))

    return explainer, shap_values, importance_df


def predict_with_explanation(model, explainer, X_single: pd.DataFrame,
                              feature_names: list) -> dict:
    """
    Single-patient prediction with top-3 risk drivers (SHAP).
    Used by the FastAPI serving layer.
    """
    proba = float(model.predict_proba(X_single)[0, 1])

    if proba >= THRESHOLD_HIGH:
        risk_tier = "HIGH"
    elif proba >= THRESHOLD_MEDIUM:
        risk_tier = "MEDIUM"
    else:
        risk_tier = "LOW"

    # SHAP for this patient
    base = model.calibrated_classifiers_[0].estimator
    sv = explainer.shap_values(X_single)[0]
    top_idx = np.argsort(np.abs(sv))[::-1][:3]
    drivers = [
        {
            "feature": feature_names[i],
            "shap_value": round(float(sv[i]), 4),
            "direction": "increases" if sv[i] > 0 else "decreases",
        }
        for i in top_idx
    ]

    return {
        "readmit_probability": round(proba, 4),
        "risk_tier": risk_tier,
        "top_risk_drivers": drivers,
        "needs_review": risk_tier == "HIGH",
        "threshold_high": THRESHOLD_HIGH,
        "threshold_medium": THRESHOLD_MEDIUM,
    }


def run_training_pipeline():
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2]))

    from src.features.engineer import run_feature_pipeline

    log.info("Loading feature mart...")
    X, y, feature_names, feat = run_feature_pipeline()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    feat_test = feat.iloc[y_test.index]
    log.info("Train: %d  Test: %d  Positive rate: %.2f%%",
             len(X_train), len(X_test), y_test.mean() * 100)

    # 5-fold CV before final training
    cv_model = xgb.XGBClassifier(n_estimators=200, max_depth=5, random_state=42,
                                   use_label_encoder=False, eval_metric="logloss", n_jobs=-1)
    cv_scores = cross_val_score(cv_model, X_train, y_train, cv=StratifiedKFold(5),
                                 scoring="roc_auc", n_jobs=-1)
    log.info("5-fold CV AUROC: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    model = train_model(X_train, y_train)

    metrics = evaluate(model, X_test, y_test, feature_names)
    fairness = fairness_audit(model, X_test, y_test, feat_test)
    explainer, shap_vals, importance_df = compute_shap(model, X_train, X_test, feature_names)

    # Save artefacts
    with open(MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODEL_DIR / "explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)
    importance_df.to_csv(MODEL_DIR / "shap_importance.csv", index=False)

    report = {
        "model": "XGBoost + Platt calibration",
        "cv_auroc_mean": round(float(cv_scores.mean()), 4),
        "cv_auroc_std": round(float(cv_scores.std()), 4),
        "test_metrics": metrics,
        "fairness_audit": fairness,
        "feature_names": feature_names,
        "top_10_features": importance_df.head(10).to_dict(orient="records"),
        "thresholds": {
            "high": THRESHOLD_HIGH,
            "medium": THRESHOLD_MEDIUM,
        },
    }
    with open(REPORT_DIR / "model_report.json", "w") as f:
        json.dump(report, f, indent=2)
    log.info("Model report → %s", REPORT_DIR / "model_report.json")

    log.info("Training complete. AUROC=%.4f  AUPRC=%.4f",
             metrics["auroc"], metrics["auprc"])

    return model, explainer, feature_names, report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    model, explainer, feature_names, report = run_training_pipeline()
    print("\n" + "=" * 50)
    print(f"AUROC  : {report['test_metrics']['auroc']}")
    print(f"AUPRC  : {report['test_metrics']['auprc']}")
    print(f"CV AUROC: {report['cv_auroc_mean']} ± {report['cv_auroc_std']}")
    print(f"Fairness flags: {report['fairness_audit']['flagged_groups']}")
    print("\nTop 5 risk drivers (SHAP):")
    for r in report["top_10_features"][:5]:
        print(f"  {r['feature']:<35} SHAP={r['mean_abs_shap']:.4f}")
