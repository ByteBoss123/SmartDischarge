"""Unit tests run on a handful of hand-written encounters (shape checks only, no results claimed).
Tests marked `real_data` run on the real UCI file when data/diabetic_data.csv is present."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smartdischarge.data import EXPIRED_OR_HOSPICE, build_cohort
from smartdischarge.drift import psi
from smartdischarge.features import CATEGORICAL, MEDS, NUMERIC, build_features, icd9_group

DATA = Path(__file__).parents[1] / "data" / "diabetic_data.csv"


def _encounter(eid, pid, readmitted="NO", disposition=1, **kw):
    row = {"encounter_id": eid, "patient_nbr": pid, "race": "Caucasian", "gender": "Female",
           "age": "[70-80)", "admission_type_id": 1, "discharge_disposition_id": disposition,
           "admission_source_id": 7, "time_in_hospital": 3, "payer_code": np.nan,
           "medical_specialty": np.nan, "num_lab_procedures": 40, "num_procedures": 1,
           "num_medications": 15, "number_outpatient": 0, "number_emergency": 0,
           "number_inpatient": 0, "diag_1": "428", "diag_2": "250.02", "diag_3": "V45",
           "number_diagnoses": 8, "max_glu_serum": np.nan, "A1Cresult": np.nan, "change": "No",
           "diabetesMed": "Yes", "readmitted": readmitted}
    row.update({m: "No" for m in MEDS})
    row.update(kw)
    return row


def test_cohort_keeps_first_encounter_and_drops_expired_and_hospice():
    raw = pd.DataFrame([
        _encounter(2, 100, "<30"), _encounter(1, 100, "NO"),       # same patient: keep encounter 1
        _encounter(3, 200, "<30", disposition=11),                 # expired
        _encounter(4, 300, "<30", disposition=13),                 # hospice
        _encounter(5, 400, ">30"),
    ])
    cohort, steps = build_cohort(raw)
    assert list(cohort["encounter_id"]) == [1, 5]
    assert cohort["label"].tolist() == [0, 0]
    assert not cohort["discharge_disposition_id"].isin(EXPIRED_OR_HOSPICE).any()
    assert steps[0]["rows"] == 5 and steps[1]["rows"] == 4 and steps[2]["rows"] == 2


@pytest.mark.parametrize("code,group", [("428", "Circulatory"), ("250.02", "Diabetes"),
                                        ("486", "Respiratory"), ("V45", "Other"),
                                        ("820", "Injury"), (np.nan, "Missing")])
def test_icd9_grouping(code, group):
    assert icd9_group(code) == group


def test_features_have_no_missing_values_and_reuse_vocab():
    raw = pd.DataFrame([_encounter(i, i, insulin="Up", metformin="Steady") for i in range(5)])
    X, vocab = build_features(raw)
    assert list(X.columns) == NUMERIC + CATEGORICAL
    assert not X.isna().any().any()
    assert X.loc[0, "n_meds_active"] == 2 and X.loc[0, "n_meds_changed"] == 1
    assert X.loc[0, "age_years"] == 75
    X2, _ = build_features(raw.assign(medical_specialty="Cardiology"), **vocab)
    assert (X2["medical_specialty_top"] == "Other").all()   # unseen at fit time


def test_psi_is_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=20000)
    assert psi(a, rng.normal(size=20000)) < 0.01
    assert psi(a, rng.normal(loc=1.0, size=20000)) > 0.25


real_data = pytest.mark.skipif(not DATA.exists(), reason="run data/download.sh first")


@real_data
def test_real_cohort_matches_documented_counts():
    from smartdischarge.data import load_raw
    cohort, steps = build_cohort(load_raw(DATA))
    assert [s["rows"] for s in steps] == [101766, 71518, 69973, 6277]
    assert cohort["patient_nbr"].is_unique


@real_data
def test_api_scores_a_real_encounter(tmp_path):
    model = Path(__file__).parents[1] / "reports" / "model.joblib"
    if not model.exists():
        pytest.skip("train the model first: python -m smartdischarge.train")
    from fastapi.testclient import TestClient
    from smartdischarge.api import app
    client = TestClient(app)
    r = client.post("/score", json={"number_inpatient": 3, "discharge_disposition_id": 3,
                                    "diag_1": "428", "time_in_hospital": 9})
    assert r.status_code == 200
    body = r.json()
    assert 0 < body["readmission_30d_probability"] < 1
    assert body["tier"] in {"follow up", "monitor", "standard"}
    low = client.post("/score", json={}).json()["readmission_30d_probability"]
    assert body["readmission_30d_probability"] > low
    assert client.post("/score", json={"time_in_hospital": 40}).status_code == 422
