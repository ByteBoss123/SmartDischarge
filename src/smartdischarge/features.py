"""Feature engineering. Every feature is known at discharge time.

Diagnosis grouping follows Strack et al. (2014), the paper that released this dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MEDS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
    "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

NUMERIC = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "age_years", "n_meds_active", "n_meds_changed", "prior_visits_total",
]
CATEGORICAL = [
    "race", "gender", "admission_type", "admission_source", "discharge_group", "diag_1_group",
    "diag_2_group", "diag_3_group", "A1Cresult", "max_glu_serum", "insulin", "change",
    "diabetesMed", "medical_specialty_top", "payer_code_top",
]


def icd9_group(code) -> str:
    if pd.isna(code):
        return "Missing"
    s = str(code)
    if s.startswith(("V", "E")):
        return "Other"
    try:
        v = float(s)
    except ValueError:
        return "Other"
    if 390 <= v <= 459 or int(v) == 785:
        return "Circulatory"
    if 460 <= v <= 519 or int(v) == 786:
        return "Respiratory"
    if 520 <= v <= 579 or int(v) == 787:
        return "Digestive"
    if int(v) == 250:
        return "Diabetes"
    if 800 <= v <= 999:
        return "Injury"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if 580 <= v <= 629 or int(v) == 788:
        return "Genitourinary"
    if 140 <= v <= 239:
        return "Neoplasms"
    return "Other"


def _admission_type(x) -> str:
    return {1: "Emergency", 2: "Urgent", 3: "Elective"}.get(x, "Other/Unknown")


def _admission_source(x) -> str:
    if x == 7:
        return "Emergency room"
    if x in (1, 2, 3):
        return "Referral"
    if x in (4, 5, 6, 10, 22, 25):
        return "Transfer"
    return "Other/Unknown"


def _discharge_group(x) -> str:
    if x == 1:
        return "Home"
    if x in (6, 8):
        return "Home health"
    if x in (3, 4, 24):
        return "Skilled nursing / ICF"
    if x in (2, 5, 9, 10, 15, 16, 17, 22, 23, 27, 28, 29, 30):
        return "Other facility / transfer"
    if x == 7:
        return "Left against advice"
    return "Other/Unknown"


def _top(series: pd.Series, keep: list[str]) -> pd.Series:
    return series.where(series.isin(keep), np.where(series.isna(), "Missing", "Other"))


def build_features(df: pd.DataFrame, top_specialties: list[str] | None = None,
                   top_payers: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Return (features, fitted_vocab). Pass the vocab learned on training data when scoring."""
    out = pd.DataFrame(index=df.index)
    for c in ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
              "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]:
        out[c] = df[c].astype(float)

    # "[70-80)" -> 75
    out["age_years"] = df["age"].str.extract(r"\[(\d+)-")[0].astype(float) + 5
    meds = df[[m for m in MEDS if m in df.columns]]
    out["n_meds_active"] = (meds != "No").sum(axis=1).astype(float)
    out["n_meds_changed"] = meds.isin(["Up", "Down"]).sum(axis=1).astype(float)
    out["prior_visits_total"] = out[["number_outpatient", "number_emergency", "number_inpatient"]].sum(axis=1)

    out["race"] = df["race"].fillna("Missing")
    out["gender"] = df["gender"].replace({"Unknown/Invalid": "Missing"}).fillna("Missing")
    out["admission_type"] = df["admission_type_id"].map(_admission_type)
    out["admission_source"] = df["admission_source_id"].map(_admission_source)
    out["discharge_group"] = df["discharge_disposition_id"].map(_discharge_group)
    for i in (1, 2, 3):
        out[f"diag_{i}_group"] = df[f"diag_{i}"].map(icd9_group)
    for c in ["A1Cresult", "max_glu_serum"]:
        out[c] = df[c].fillna("None").astype(str)
    out["insulin"] = df["insulin"]
    out["change"] = df["change"]
    out["diabetesMed"] = df["diabetesMed"]

    if top_specialties is None:
        top_specialties = df["medical_specialty"].value_counts().head(10).index.tolist()
    if top_payers is None:
        top_payers = df["payer_code"].value_counts().head(8).index.tolist()
    out["medical_specialty_top"] = _top(df["medical_specialty"], top_specialties)
    out["payer_code_top"] = _top(df["payer_code"], top_payers)

    for c in CATEGORICAL:
        out[c] = out[c].astype(str)
    return out[NUMERIC + CATEGORICAL], {"top_specialties": top_specialties, "top_payers": top_payers}
