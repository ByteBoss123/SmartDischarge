"""
Layer 4b — Model Monitoring & Drift Detection
SmartDischarge: 30-Day Hospital Readmission Prediction

Production monitoring:
  - Feature distribution drift (Population Stability Index)
  - Prediction score drift (score distribution shift)
  - Model performance degradation alerts
  - Automated retraining trigger logic

ML Engineer role signal: production ML ops beyond deployment —
drift detection, alerting, retraining gates.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("smartdischarge.monitoring")

REPORT_DIR = Path(__file__).parents[2] / "docs"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PSI_ALERT_THRESHOLD   = 0.20   # PSI > 0.20 → significant drift
PSI_WARNING_THRESHOLD = 0.10   # PSI 0.10–0.20 → minor drift
AUROC_DROP_THRESHOLD  = 0.05   # AUROC drop > 0.05 → alert


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    PSI measures how much a feature distribution has shifted.
    PSI < 0.10  → no significant change
    PSI 0.10–0.20 → moderate change (monitor)
    PSI > 0.20  → significant change (alert — retrain)
    """
    eps = 1e-6
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0] -= eps
    bins[-1] += eps

    exp_counts = np.histogram(expected, bins=bins)[0] + eps
    act_counts = np.histogram(actual,   bins=bins)[0] + eps

    exp_pct = exp_counts / exp_counts.sum()
    act_pct = act_counts / act_counts.sum()

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return round(float(psi), 4)


def monitor_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_features: list,
) -> dict:
    """Compute PSI for each numeric feature between reference and current window."""
    results = {}
    alerts = []
    warnings = []

    for feat in numeric_features:
        if feat not in reference_df.columns or feat not in current_df.columns:
            continue
        psi = population_stability_index(
            reference_df[feat].dropna().values,
            current_df[feat].dropna().values,
        )
        level = ("ALERT" if psi > PSI_ALERT_THRESHOLD
                 else "WARNING" if psi > PSI_WARNING_THRESHOLD
                 else "OK")
        results[feat] = {"psi": psi, "status": level}
        if level == "ALERT":
            alerts.append(feat)
        elif level == "WARNING":
            warnings.append(feat)

    if alerts:
        log.warning("DRIFT ALERT — significant distribution shift: %s", alerts)
    if warnings:
        log.info("Drift warning — minor shift: %s", warnings)
    else:
        log.info("Feature drift monitor: no significant drift detected")

    return {
        "features": results,
        "alert_features": alerts,
        "warning_features": warnings,
        "recommend_retrain": len(alerts) > 0,
    }


def monitor_score_drift(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
) -> dict:
    """
    Monitor prediction score distribution shift.
    A PSI on the score distribution catches silent model degradation
    even when we don't have ground-truth labels yet.
    """
    psi = population_stability_index(reference_scores, current_scores)
    mean_shift = float(current_scores.mean() - reference_scores.mean())
    high_risk_rate_ref = float((reference_scores >= 0.35).mean())
    high_risk_rate_cur = float((current_scores >= 0.35).mean())

    status = ("ALERT" if psi > PSI_ALERT_THRESHOLD
              else "WARNING" if psi > PSI_WARNING_THRESHOLD
              else "OK")

    result = {
        "psi": psi,
        "status": status,
        "mean_score_shift": round(mean_shift, 4),
        "high_risk_rate_reference": round(high_risk_rate_ref, 4),
        "high_risk_rate_current": round(high_risk_rate_cur, 4),
        "high_risk_rate_shift": round(high_risk_rate_cur - high_risk_rate_ref, 4),
    }

    if status == "ALERT":
        log.warning("SCORE DRIFT ALERT: PSI=%.4f, mean shift=%.4f", psi, mean_shift)
    return result


def check_performance_degradation(
    baseline_auroc: float,
    current_auroc: float,
) -> dict:
    """Compare current window AUROC against training baseline."""
    drop = baseline_auroc - current_auroc
    alert = drop > AUROC_DROP_THRESHOLD
    result = {
        "baseline_auroc": round(baseline_auroc, 4),
        "current_auroc": round(current_auroc, 4),
        "auroc_drop": round(drop, 4),
        "alert": alert,
    }
    if alert:
        log.warning(
            "PERFORMANCE ALERT: AUROC dropped %.4f (%.4f → %.4f)",
            drop, baseline_auroc, current_auroc
        )
    else:
        log.info("Performance check OK: AUROC drop=%.4f", drop)
    return result


def run_monitoring_report(
    feature_mart_path: Path,
    report_out_path: Path = REPORT_DIR / "monitoring_report.json",
) -> dict:
    """
    Simulate a monitoring run:
    Split feature mart into reference (first 70%) and current (last 30%)
    to mimic temporal drift detection.
    """
    df = pd.read_parquet(feature_mart_path)

    numeric_features = [
        "time_in_hospital", "number_inpatient", "number_emergency",
        "num_medications", "num_lab_procedures", "number_diagnoses",
        "age_midpoint", "charlson_proxy", "clinical_risk_score",
    ]
    available = [f for f in numeric_features if f in df.columns]

    split = int(len(df) * 0.70)
    reference = df.iloc[:split]
    current   = df.iloc[split:]

    # Simulate score drift with slight positive shift in current window
    ref_scores = np.random.beta(2, 8, len(reference))
    cur_scores = np.random.beta(2.2, 7.5, len(current))

    feature_drift  = monitor_feature_drift(reference, current, available)
    score_drift    = monitor_score_drift(ref_scores, cur_scores)

    # Simulate slight AUROC degradation
    import json
    with open(REPORT_DIR / "model_report.json") as f:
        model_report = json.load(f)
    baseline_auroc = model_report["test_metrics"]["auroc"]
    current_auroc  = baseline_auroc - np.random.uniform(0.005, 0.025)
    perf_check     = check_performance_degradation(baseline_auroc, current_auroc)

    report = {
        "monitoring_timestamp": pd.Timestamp.utcnow().isoformat(),
        "reference_window_rows": len(reference),
        "current_window_rows": len(current),
        "feature_drift": feature_drift,
        "score_drift": score_drift,
        "performance_check": perf_check,
        "overall_status": (
            "ALERT" if (feature_drift["recommend_retrain"] or
                        score_drift["status"] == "ALERT" or
                        perf_check["alert"])
            else "OK"
        ),
        "recommendation": (
            "Trigger retraining pipeline" if feature_drift["recommend_retrain"]
            else "Continue monitoring"
        ),
    }

    with open(report_out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Monitoring report → %s", report_out_path)
    log.info("Overall status: %s | Recommendation: %s",
             report["overall_status"], report["recommendation"])
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    processed = Path(__file__).parents[2] / "data" / "processed" / "feature_mart.parquet"
    report = run_monitoring_report(processed)
    print(json.dumps(report, indent=2))
