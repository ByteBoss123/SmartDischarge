"""
SmartDischarge — Master Pipeline Runner
Orchestrates all 5 layers end-to-end:
  1. Data ingestion + validation
  2. Feature engineering
  3. Model training (XGBoost + SHAP + calibration + fairness)
  4. Monitoring report
  5. Analytics dashboard

Run: python run_pipeline.py
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("smartdischarge.pipeline")

PROCESSED = Path(__file__).parent / "data" / "processed"
DOCS      = Path(__file__).parent / "docs"


def banner(msg):
    log.info("")
    log.info("=" * 60)
    log.info("  %s", msg)
    log.info("=" * 60)


def main():
    t_start = time.time()
    banner("SmartDischarge — Full Pipeline Starting")

    # ── Layer 1: Ingestion ──────────────────────────────────────────────────
    banner("Layer 1 — Data Ingestion & Validation")
    from src.ingestion.ingest import run_ingestion
    df_clean = run_ingestion()

    # ── Layer 2: Feature Engineering ────────────────────────────────────────
    banner("Layer 2 — Feature Engineering (Charlson, utilisation, meds)")
    from src.features.engineer import build_features, prepare_model_input
    feat = build_features(df_clean)
    X, y, feature_names = prepare_model_input(feat)
    feat_path = PROCESSED / "feature_mart.parquet"
    feat.to_parquet(feat_path, index=False, engine="pyarrow")
    log.info("Feature mart: %d rows × %d features saved", *X.shape)

    # ── Layer 3: Model Training ──────────────────────────────────────────────
    banner("Layer 3 — Model Training (XGBoost + SHAP + Calibration + Fairness)")
    from src.model.train import run_training_pipeline
    model, explainer, feature_names, report = run_training_pipeline()
    log.info(
        "Model ready — AUROC=%.4f  AUPRC=%.4f  CV=%.4f±%.4f",
        report["test_metrics"]["auroc"],
        report["test_metrics"]["auprc"],
        report["cv_auroc_mean"],
        report["cv_auroc_std"],
    )

    # ── Layer 4b: Monitoring ─────────────────────────────────────────────────
    banner("Layer 4b — Drift Detection & Monitoring Report")
    from src.monitoring.drift import run_monitoring_report
    monitoring = run_monitoring_report(feat_path)
    log.info("Monitoring status: %s | %s",
             monitoring["overall_status"], monitoring["recommendation"])

    # ── Layer 5: Dashboard ───────────────────────────────────────────────────
    banner("Layer 5 — Analytics Dashboard & CMS Business Impact")
    from src.dashboard.analytics import generate_dashboard
    report_path = DOCS / "model_report.json"
    impact = generate_dashboard(feat_path, report_path)

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    banner(f"Pipeline Complete in {elapsed:.1f}s")

    print("\n" + "─" * 60)
    print("  SMARTDISCHARGE — RESULTS SUMMARY")
    print("─" * 60)
    print(f"  Dataset               {len(df_clean):>12,} clean encounters")
    print(f"  Features engineered   {len(feature_names):>12,}")
    print(f"  Readmit-30d rate      {y.mean()*100:>11.2f}%")
    print(f"")
    print(f"  MODEL PERFORMANCE")
    print(f"  AUROC                 {report['test_metrics']['auroc']:>12.4f}")
    print(f"  AUPRC                 {report['test_metrics']['auprc']:>12.4f}")
    print(f"  CV AUROC (5-fold)     {report['cv_auroc_mean']:>8.4f} ± {report['cv_auroc_std']:.4f}")
    print(f"  Calibration error     {report['test_metrics']['calibration_error']:>12.4f}")
    print(f"")
    print(f"  FAIRNESS")
    flags = report["fairness_audit"]["flagged_groups"]
    print(f"  Flagged subgroups     {flags if flags else 'None (PASSED)':>12}")
    print(f"")
    print(f"  BUSINESS IMPACT (300-bed hospital, annual)")
    print(f"  Readmit rate baseline {impact['baseline_readmit_rate_pct']:>10.1f}%")
    print(f"  Readmit rate (model)  {impact['model_readmit_rate_pct']:>10.1f}%")
    print(f"  Prevented readmits    {impact['prevented_readmissions_annual']:>12,}")
    print(f"  Direct savings        ${impact['direct_savings_usd']:>11,}")
    print(f"  CMS penalty avoided   ${impact['cms_penalty_reduction_usd']:>11,}")
    print(f"  Total annual impact   ${impact['total_annual_impact_usd']:>11,}")
    print(f"")
    print(f"  ARTEFACTS")
    print(f"  Model                 data/processed/model.pkl")
    print(f"  SHAP explainer        data/processed/explainer.pkl")
    print(f"  Feature mart          data/processed/feature_mart.parquet")
    print(f"  Model report          docs/model_report.json")
    print(f"  Monitoring report     docs/monitoring_report.json")
    print(f"  Dashboard             docs/dashboard.html")
    print("─" * 60)
    print(f"  API: uvicorn src.api.server:app --host 0.0.0.0 --port 8000")
    print("─" * 60)


if __name__ == "__main__":
    main()
