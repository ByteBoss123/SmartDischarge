"""Train, compare, calibrate and evaluate readmission models.

Design:
- 60/20/20 stratified split. Each patient appears once (see data.py), so no patient spans two splits.
- Three candidates, chosen on the validation set only:
    * rule baseline: rank patients by prior-year inpatient visits (what a care team could do with no model)
    * logistic regression
    * XGBoost with early stopping
- The chosen model is calibrated on the validation set (Platt / sigmoid) and scored once on the test set.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import build_cohort, load_raw
from .drift import psi
from .features import CATEGORICAL, NUMERIC, build_features

SEED = 42


def preprocessor(scale: bool) -> ColumnTransformer:
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [("num", num, NUMERIC),
         ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False), CATEGORICAL)]
    )


def make_logreg() -> Pipeline:
    return Pipeline([("prep", preprocessor(scale=True)),
                     ("clf", LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))])


def make_xgb(n_estimators: int = 2000) -> Pipeline:
    return Pipeline([("prep", preprocessor(scale=False)),
                     ("clf", xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=0.03, max_depth=4,
                                               min_child_weight=20, subsample=0.8, colsample_bytree=0.7,
                                               reg_lambda=5.0, eval_metric="auc", random_state=SEED,
                                               n_jobs=2))])


def bootstrap_auc_ci(y, p, n=1000, seed=SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def top_fraction_stats(y, p, frac: float) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    k = int(round(len(y) * frac))
    top = np.argsort(-p, kind="stable")[:k]
    hits = int(y[top].sum())
    return {"share_of_discharges": frac, "flagged": k, "readmissions_caught": hits,
            "precision": round(hits / k, 4), "recall": round(hits / y.sum(), 4),
            "lift_vs_random": round((hits / k) / y.mean(), 2),
            "calls_per_readmission_caught": round(k / max(hits, 1), 1)}


def subgroup_auc(y, p, groups: pd.Series, min_pos: int = 30) -> dict:
    out = {}
    for g in sorted(groups.unique()):
        m = (groups == g).to_numpy()
        yy, pp = np.asarray(y)[m], np.asarray(p)[m]
        if yy.sum() < min_pos or yy.sum() == len(yy):
            out[g] = {"n": int(m.sum()), "positives": int(yy.sum()), "auroc": None,
                      "note": f"fewer than {min_pos} positives, not scored"}
            continue
        lo, hi = bootstrap_auc_ci(yy, pp, n=500)
        out[g] = {"n": int(m.sum()), "positives": int(yy.sum()),
                  "auroc": round(roc_auc_score(yy, pp), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    return out


def shap_importance(model: Pipeline, X: pd.DataFrame, n: int = 2000) -> list[dict]:
    import shap
    sample = X.sample(min(n, len(X)), random_state=SEED)
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    Xt = prep.transform(sample)
    names = prep.get_feature_names_out()
    values = shap.TreeExplainer(clf).shap_values(Xt)
    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=names)
    # fold one-hot columns back into their source feature
    source = [n.split("__", 1)[1] for n in names]
    source = [next((c for c in CATEGORICAL if s.startswith(c + "_")), s) for s in source]
    agg = mean_abs.groupby(source).sum().sort_values(ascending=False)
    return [{"feature": f, "mean_abs_shap": round(float(v), 4)} for f, v in agg.head(10).items()]


def run(data_path: str | Path, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort, steps = build_cohort(load_raw(data_path))
    y_all = cohort["label"].to_numpy()
    idx = np.arange(len(cohort))
    tr, tmp = train_test_split(idx, test_size=0.4, stratify=y_all, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.5, stratify=y_all[tmp], random_state=SEED)

    X_tr, vocab = build_features(cohort.iloc[tr])
    X_va, _ = build_features(cohort.iloc[va], **vocab)
    X_te, _ = build_features(cohort.iloc[te], **vocab)
    y_tr, y_va, y_te = y_all[tr], y_all[va], y_all[te]

    # Candidates, compared on validation only
    rule_va = X_va["number_inpatient"].to_numpy()
    logreg = make_logreg().fit(X_tr, y_tr)
    xgb_model = make_xgb()
    prep = xgb_model.named_steps["prep"].fit(X_tr)
    xgb_model.named_steps["clf"].set_params(early_stopping_rounds=100)
    xgb_model.named_steps["clf"].fit(prep.transform(X_tr), y_tr,
                                     eval_set=[(prep.transform(X_va), y_va)], verbose=False)
    best_iter = int(xgb_model.named_steps["clf"].best_iteration)

    val_scores = {
        "rule: prior inpatient visits": roc_auc_score(y_va, rule_va),
        "logistic regression": roc_auc_score(y_va, logreg.predict_proba(X_va)[:, 1]),
        "xgboost": roc_auc_score(y_va, xgb_model.predict_proba(X_va)[:, 1]),
    }
    chosen_name = max(["logistic regression", "xgboost"], key=val_scores.get)
    chosen = logreg if chosen_name == "logistic regression" else xgb_model

    calibrated = CalibratedClassifierCV(FrozenEstimator(chosen), method="sigmoid").fit(X_va, y_va)

    # Test set: scored once
    p_te = calibrated.predict_proba(X_te)[:, 1]
    test_scores = {
        "rule: prior inpatient visits": roc_auc_score(y_te, X_te["number_inpatient"]),
        "logistic regression": roc_auc_score(y_te, logreg.predict_proba(X_te)[:, 1]),
        "xgboost": roc_auc_score(y_te, xgb_model.predict_proba(X_te)[:, 1]),
    }
    lo, hi = bootstrap_auc_ci(y_te, p_te)
    frac_pos, mean_pred = calibration_curve(y_te, p_te, n_bins=10, strategy="quantile")

    # Stability: 5-fold CV of the chosen model family on train+val
    trva = np.concatenate([tr, va])
    X_trva, _ = build_features(cohort.iloc[trva], **vocab)
    y_trva = y_all[trva]
    cv = []
    for f_tr, f_te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X_trva, y_trva):
        m = make_logreg() if chosen_name == "logistic regression" else make_xgb(n_estimators=best_iter + 1)
        m.fit(X_trva.iloc[f_tr], y_trva[f_tr])
        cv.append(roc_auc_score(y_trva[f_te], m.predict_proba(X_trva.iloc[f_te])[:, 1]))

    ages = X_te["age_years"].map(lambda a: "<50" if a < 50 else ("50-69" if a < 70 else "70+"))
    report = {
        "data": {"source": "UCI Diabetes 130-US Hospitals 1999-2008 (Strack et al., 2014)",
                 "cohort_steps": steps, "base_rate": round(float(y_all.mean()), 4),
                 "split": {"train": int(len(tr)), "validation": int(len(va)), "test": int(len(te)),
                           "test_positives": int(y_te.sum())}},
        "model_selection_validation_auroc": {k: round(v, 4) for k, v in val_scores.items()},
        "chosen_model": chosen_name,
        "xgboost_best_iteration": best_iter,
        "test": {
            "auroc": round(roc_auc_score(y_te, p_te), 4), "auroc_ci95": [round(lo, 4), round(hi, 4)],
            "auprc": round(average_precision_score(y_te, p_te), 4),
            "auprc_random_baseline": round(float(y_te.mean()), 4),
            "brier": round(brier_score_loss(y_te, p_te), 5),
            "brier_base_rate_only": round(brier_score_loss(y_te, np.full(len(y_te), y_tr.mean())), 5),
            "uncalibrated_auroc_all_candidates": {k: round(v, 4) for k, v in test_scores.items()},
            "calibration_deciles": [{"mean_predicted": round(float(m), 4), "observed": round(float(f), 4)}
                                    for m, f in zip(mean_pred, frac_pos)],
        },
        "cv5_auroc_train_val": {"mean": round(float(np.mean(cv)), 4), "std": round(float(np.std(cv)), 4)},
        "capacity_scenarios": [top_fraction_stats(y_te, p_te, f) for f in (0.05, 0.10, 0.20)],
        "subgroup_auroc": {"race": subgroup_auc(y_te, p_te, X_te["race"]),
                           "gender": subgroup_auc(y_te, p_te, X_te["gender"]),
                           "age_band": subgroup_auc(y_te, p_te, ages)},
        "score_psi_test_vs_validation": round(psi(calibrated.predict_proba(X_va)[:, 1], p_te), 4),
    }
    if chosen_name == "xgboost":
        report["top_features_mean_abs_shap"] = shap_importance(xgb_model, X_te)

    # Tier cut-offs from validation scores (not test): top 10% "follow up", next 10% "monitor"
    p_va = calibrated.predict_proba(X_va)[:, 1]
    cutoffs = (float(np.quantile(p_va, 0.90)), float(np.quantile(p_va, 0.80)))
    report["tier_cutoffs_from_validation"] = {"follow_up_min_prob": round(cutoffs[0], 4),
                                             "monitor_min_prob": round(cutoffs[1], 4)}
    joblib.dump({"model": calibrated, "vocab": vocab, "chosen": chosen_name, "tier_cutoffs": cutoffs},
                out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    _plots(y_te, p_te, frac_pos, mean_pred, out_dir)
    return report


def _plots(y, p, frac_pos, mean_pred, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    fpr, tpr, _ = roc_curve(y, p)
    ax[0].plot(fpr, tpr, color="#2a6f97")
    ax[0].plot([0, 1], [0, 1], ls="--", color="#999")
    ax[0].set(title=f"ROC, test set (AUROC {roc_auc_score(y, p):.3f})", xlabel="False positive rate",
              ylabel="True positive rate")
    ax[1].plot(mean_pred, frac_pos, marker="o", color="#2a6f97")
    lim = max(mean_pred.max(), frac_pos.max()) * 1.1
    ax[1].plot([0, lim], [0, lim], ls="--", color="#999")
    ax[1].set(title="Calibration by decile, test set", xlabel="Mean predicted risk",
              ylabel="Observed 30-day readmission rate")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_calibration.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/diabetic_data.csv")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    r = run(a.data, a.out)
    print(json.dumps({k: r[k] for k in ("chosen_model", "model_selection_validation_auroc", "test")}, indent=2)[:3000])
