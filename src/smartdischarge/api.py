"""FastAPI scoring service.

POST /score with one encounter in the raw UCI column format; returns a calibrated 30-day
readmission probability and a follow-up tier. Tier cut-offs are set on the validation set
(top 10% of scores = "follow up", next 10% = "monitor"); see reports/metrics.json.
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .features import MEDS, build_features

MODEL_PATH = Path(os.environ.get("SMARTDISCHARGE_MODEL", "reports/model.joblib"))


class Encounter(BaseModel):
    race: str | None = None
    gender: str = "Female"
    age: str = Field("[70-80)", pattern=r"^\[\d+-\d+\)$")
    admission_type_id: int = 1
    discharge_disposition_id: int = 1
    admission_source_id: int = 7
    time_in_hospital: int = Field(3, ge=1, le=14)
    payer_code: str | None = None
    medical_specialty: str | None = None
    num_lab_procedures: int = Field(40, ge=0)
    num_procedures: int = Field(1, ge=0)
    num_medications: int = Field(15, ge=0)
    number_outpatient: int = Field(0, ge=0)
    number_emergency: int = Field(0, ge=0)
    number_inpatient: int = Field(0, ge=0)
    diag_1: str | None = None
    diag_2: str | None = None
    diag_3: str | None = None
    number_diagnoses: int = Field(8, ge=1)
    max_glu_serum: str | None = None
    A1Cresult: str | None = None
    insulin: str = "No"
    change: str = "No"
    diabetesMed: str = "Yes"
    other_meds: dict[str, str] = Field(default_factory=dict,
                                       description="Any of the other 22 diabetes drug columns: No/Steady/Up/Down")


app = FastAPI(title="SmartDischarge")
_state: dict = {}


def _load():
    if "bundle" not in _state:
        if not MODEL_PATH.exists():
            raise HTTPException(503, f"model not found at {MODEL_PATH}; run the training pipeline first")
        bundle = joblib.load(MODEL_PATH)
        _state["bundle"] = bundle
    return _state["bundle"]


def score_encounter(enc: Encounter) -> dict:
    bundle = _load()
    row = enc.model_dump()
    meds = {m: "No" for m in MEDS}
    meds["insulin"] = row.pop("insulin")
    meds.update(row.pop("other_meds") or {})
    row.update(meds)
    df = pd.DataFrame([row]).replace({None: np.nan})
    X, _ = build_features(df, **bundle["vocab"])
    p = float(bundle["model"].predict_proba(X)[:, 1][0])
    cut_follow, cut_monitor = bundle.get("tier_cutoffs", (np.inf, np.inf))
    tier = "follow up" if p >= cut_follow else ("monitor" if p >= cut_monitor else "standard")
    return {"readmission_30d_probability": round(p, 4), "tier": tier, "model": bundle["chosen"]}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/score")
def score(enc: Encounter):
    return score_encounter(enc)
