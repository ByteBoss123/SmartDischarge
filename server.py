"""
Layer 4 — Production API
SmartDischarge: 30-Day Hospital Readmission Prediction

FastAPI serving layer:
  - POST /predict       → risk score + tier + SHAP top-3 drivers
  - GET  /review-queue  → high-risk patients awaiting review
  - GET  /stats         → live model performance metrics
  - GET  /health        → liveness check
  - GET  /model-info    → version + thresholds + feature list

ML Engineer role signal: production deployment, latency tracking,
structured logging, versioning, health monitoring.
"""

import json
import logging
import pickle
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator

from src.kafka.producer import kafka_producer

log = logging.getLogger("smartdischarge.api")

MODEL_DIR  = Path(__file__).parents[2] / "data" / "processed"
REPORT_DIR = Path(__file__).parents[2] / "docs"

# ── Load artefacts at startup ─────────────────────────────────────────────────
def _load_artefacts():
    model_path    = MODEL_DIR / "model.pkl"
    explainer_path = MODEL_DIR / "explainer.pkl"
    report_path   = REPORT_DIR / "model_report.json"

    if not model_path.exists():
        raise RuntimeError(
            "Model artefacts not found. Run `python -m src.model.train` first."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(explainer_path, "rb") as f:
        explainer = pickle.load(f)
    with open(report_path) as f:
        report = json.load(f)

    return model, explainer, report


# ── In-memory review queue (production: replace with DB) ─────────────────────
review_queue: deque = deque(maxlen=1000)
prediction_log: deque = deque(maxlen=5000)

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PatientEncounter(BaseModel):
    """Input schema — mirrors the feature mart columns."""
    model_config = {"json_schema_extra": {"example": {
        "encounter_id": "ENC-20240115-001",
        "time_in_hospital": 5,
        "number_inpatient": 1,
        "number_emergency": 0,
        "number_outpatient": 0,
        "number_diagnoses": 7,
        "num_medications": 12,
        "num_lab_procedures": 30,
        "age_midpoint": 65.0,
        "diabetesMed": "Yes",
        "change": "Ch",
        "A1Cresult": ">7",
    }}}

    encounter_id: Optional[str]     = Field(None)
    time_in_hospital: int           = Field(..., ge=1, le=14)
    number_inpatient: int           = Field(..., ge=0, le=21)
    number_emergency: int           = Field(..., ge=0, le=76)
    number_outpatient: int          = Field(..., ge=0, le=42)
    number_diagnoses: int           = Field(..., ge=1, le=16)
    num_medications: int            = Field(..., ge=1, le=81)
    num_lab_procedures: int         = Field(..., ge=1, le=132)
    num_procedures: int             = Field(0, ge=0, le=6)
    age_midpoint: float             = Field(..., ge=5, le=95)
    diabetesMed: str                = Field(...)
    change: str                     = Field(...)
    A1Cresult: str                  = Field(...)
    max_glu_serum: str              = Field("None")
    admission_type_id: int          = Field(1, ge=1, le=8)
    charlson_proxy: Optional[float] = Field(None)

    @field_validator("diabetesMed")
    @classmethod
    def valid_diabetes_med(cls, v: str) -> str:
        if v not in ("Yes", "No"):
            raise ValueError("diabetesMed must be 'Yes' or 'No'")
        return v

    @field_validator("change")
    @classmethod
    def valid_change(cls, v: str) -> str:
        if v not in ("Ch", "No"):
            raise ValueError("change must be 'Ch' or 'No'")
        return v


class PredictionResponse(BaseModel):
    request_id: str
    encounter_id: Optional[str]
    readmit_probability: float
    risk_tier: str
    needs_review: bool
    top_risk_drivers: list
    latency_ms: float
    timestamp: str
    model_version: str


class BatchRequest(BaseModel):
    encounters: list[PatientEncounter]


# ── Feature vector construction ───────────────────────────────────────────────

FEATURE_ORDER = [
    "charlson_proxy", "has_diabetes_primary", "has_circulatory_dx",
    "number_diagnoses", "time_in_hospital", "number_inpatient",
    "number_emergency", "number_outpatient", "prior_visits_total",
    "has_prior_inpatient", "has_prior_emergency", "high_prior_utilisation",
    "long_los", "los_x_diagnoses", "num_medications", "num_lab_procedures",
    "num_procedures", "a1c_tested", "a1c_abnormal", "glucose_tested",
    "on_insulin", "med_changed", "on_diabetes_med", "polypharmacy",
    "high_lab_burden", "emergency_admission", "urgent_admission",
    "age_midpoint", "elderly", "working_age", "clinical_risk_score",
]


def encounter_to_features(enc: PatientEncounter) -> pd.DataFrame:
    """Convert a PatientEncounter object into the model's feature vector."""
    prior_visits = enc.number_outpatient + enc.number_emergency + enc.number_inpatient
    a1c_tested   = int(enc.A1Cresult != "None")
    a1c_abnormal = int(enc.A1Cresult in (">7", ">8"))
    charlson     = enc.charlson_proxy if enc.charlson_proxy is not None else 1.0

    risk_score = (
        charlson * 0.25 +
        int(enc.number_inpatient > 0) * 0.20 +
        int(enc.number_emergency > 0) * 0.15 +
        int(enc.time_in_hospital > 7) * 0.10 +
        a1c_abnormal * 0.10 +
        int(enc.change == "Ch") * 0.08 +
        int(enc.num_medications > 15) * 0.07 +
        int(enc.admission_type_id == 1) * 0.05
    )

    row = {
        "charlson_proxy":         charlson,
        "has_diabetes_primary":   0,
        "has_circulatory_dx":     0,
        "number_diagnoses":       enc.number_diagnoses,
        "time_in_hospital":       enc.time_in_hospital,
        "number_inpatient":       enc.number_inpatient,
        "number_emergency":       enc.number_emergency,
        "number_outpatient":      enc.number_outpatient,
        "prior_visits_total":     prior_visits,
        "has_prior_inpatient":    int(enc.number_inpatient > 0),
        "has_prior_emergency":    int(enc.number_emergency > 0),
        "high_prior_utilisation": int(prior_visits > 5),
        "long_los":               int(enc.time_in_hospital > 7),
        "los_x_diagnoses":        enc.time_in_hospital * enc.number_diagnoses,
        "num_medications":        enc.num_medications,
        "num_lab_procedures":     enc.num_lab_procedures,
        "num_procedures":         enc.num_procedures,
        "a1c_tested":             a1c_tested,
        "a1c_abnormal":           a1c_abnormal,
        "glucose_tested":         int(enc.max_glu_serum != "None"),
        "on_insulin":             0,
        "med_changed":            int(enc.change == "Ch"),
        "on_diabetes_med":        int(enc.diabetesMed == "Yes"),
        "polypharmacy":           int(enc.num_medications > 15),
        "high_lab_burden":        int(enc.num_lab_procedures > 50),
        "emergency_admission":    int(enc.admission_type_id == 1),
        "urgent_admission":       int(enc.admission_type_id == 2),
        "age_midpoint":           enc.age_midpoint,
        "elderly":                int(enc.age_midpoint >= 70),
        "working_age":            int(40 <= enc.age_midpoint < 70),
        "clinical_risk_score":    risk_score,
    }
    return pd.DataFrame([{k: row.get(k, 0) for k in FEATURE_ORDER}])


# ── FastAPI app ────────────────────────────────────────────────────────────────

_model, _explainer, _report = None, None, None
_model_version = "smartdischarge-v1.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _explainer, _report
    try:
        _model, _explainer, _report = _load_artefacts()
        log.info("Model artefacts loaded. AUROC=%.4f", _report["test_metrics"]["auroc"])
    except RuntimeError as e:
        log.error("Startup error: %s", e)
    yield
    kafka_producer.close()
    log.info("Shutting down SmartDischarge API")


app = FastAPI(
    title="SmartDischarge API",
    description="30-Day Hospital Readmission Prediction — production serving layer",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "healthy" if _model is not None else "degraded",
        "model_loaded": _model is not None,
        "timestamp": datetime.utcnow().isoformat(),
        "version": _model_version,
    }


@app.get("/model-info")
def model_info():
    if _report is None:
        raise HTTPException(503, "Model not loaded")
    return {
        "model_version": _model_version,
        "algorithm": _report["model"],
        "auroc": _report["test_metrics"]["auroc"],
        "auprc": _report["test_metrics"]["auprc"],
        "cv_auroc": f"{_report['cv_auroc_mean']} ± {_report['cv_auroc_std']}",
        "thresholds": _report["thresholds"],
        "top_features": _report["top_10_features"][:5],
        "fairness_flags": _report["fairness_audit"]["flagged_groups"],
        "n_features": len(_report["feature_names"]),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(enc: PatientEncounter):
    if _model is None:
        raise HTTPException(503, "Model not loaded")

    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())[:8]

    X = encounter_to_features(enc)
    proba = float(_model.predict_proba(X)[0, 1])

    from src.model.train import THRESHOLD_HIGH, THRESHOLD_MEDIUM
    risk_tier = ("HIGH" if proba >= THRESHOLD_HIGH
                 else "MEDIUM" if proba >= THRESHOLD_MEDIUM
                 else "LOW")

    # SHAP top-3 drivers
    base = _model.calibrated_classifiers_[0].estimator
    sv = _explainer.shap_values(X)[0]
    top_idx = np.argsort(np.abs(sv))[::-1][:3]
    drivers = [
        {
            "feature": FEATURE_ORDER[i],
            "shap_value": round(float(sv[i]), 4),
            "direction": "increases risk" if sv[i] > 0 else "decreases risk",
            "value": round(float(X.iloc[0, i]), 2),
        }
        for i in top_idx
    ]

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    response = PredictionResponse(
        request_id=request_id,
        encounter_id=enc.encounter_id,
        readmit_probability=round(proba, 4),
        risk_tier=risk_tier,
        needs_review=(risk_tier == "HIGH"),
        top_risk_drivers=drivers,
        latency_ms=latency_ms,
        timestamp=datetime.utcnow().isoformat(),
        model_version=_model_version,
    )

    prediction_log.appendleft(response.dict())
    if risk_tier == "HIGH":
        review_queue.appendleft({**response.dict(), "reviewed": False})

    # Publish to Kafka — downstream dashboards and care-team alert consumers
    kafka_producer.emit_prediction(response.dict())

    return response


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    """Batch prediction for discharge workflow integration."""
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    results = [predict(enc) for enc in req.encounters]
    summary = {
        "total": len(results),
        "high_risk": sum(1 for r in results if r.risk_tier == "HIGH"),
        "medium_risk": sum(1 for r in results if r.risk_tier == "MEDIUM"),
        "low_risk": sum(1 for r in results if r.risk_tier == "LOW"),
    }
    return {"summary": summary, "predictions": results}


@app.get("/review-queue")
def get_review_queue(limit: int = 20):
    return {
        "count": len(review_queue),
        "items": list(review_queue)[:limit],
    }


@app.get("/stats")
def get_stats():
    if not prediction_log:
        return {"message": "No predictions yet"}
    recent = list(prediction_log)
    tiers = [r["risk_tier"] for r in recent]
    latencies = [r["latency_ms"] for r in recent]
    return {
        "total_predictions": len(recent),
        "risk_distribution": {
            "HIGH":   tiers.count("HIGH"),
            "MEDIUM": tiers.count("MEDIUM"),
            "LOW":    tiers.count("LOW"),
        },
        "latency_ms": {
            "mean": round(np.mean(latencies), 2),
            "p95":  round(np.percentile(latencies, 95), 2),
            "max":  round(max(latencies), 2),
        },
        "review_queue_length": len(review_queue),
        "model_version": _model_version,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=False)
