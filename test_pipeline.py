"""
Unit tests for SmartDischarge pipeline components.
Tests data contracts, feature shapes, model behaviour, and API responses.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))


# ── Test: Feature engineering ────────────────────────────────────────────────

class TestFeatureEngineering:

    def _make_encounter(self, **kwargs):
        defaults = dict(
            encounter_id=1, patient_nbr=100, race="Caucasian", gender="Male",
            age="[70-80)", weight="?", admission_type_id=1,
            discharge_disposition_id=1, admission_source_id=7,
            time_in_hospital=8, payer_code="MC", medical_specialty="Cardiology",
            num_lab_procedures=55, num_procedures=2, num_medications=18,
            number_outpatient=0, number_emergency=1, number_inpatient=2,
            diag_1="428", diag_2="250", diag_3="486",
            number_diagnoses=9, max_glu_serum="None", A1Cresult=">7",
            change="Ch", diabetesMed="Yes", insulin="Steady",
            readmitted="<30", readmit_30d=1, age_midpoint=75,
        )
        defaults.update(kwargs)
        return pd.DataFrame([defaults])

    def test_charlson_proxy_positive(self):
        from src.features.engineer import compute_charlson_proxy
        df = self._make_encounter(diag_1="428", diag_2="250", diag_3="486")
        score = compute_charlson_proxy(df)
        assert score.iloc[0] > 0, "Charlson score should be positive for comorbid patient"

    def test_charlson_proxy_clipped(self):
        from src.features.engineer import compute_charlson_proxy
        df = self._make_encounter()
        score = compute_charlson_proxy(df)
        assert 0 <= score.iloc[0] <= 6, "Charlson proxy should be in [0, 6]"

    def test_icd9_categorisation(self):
        from src.features.engineer import categorise_icd9
        assert categorise_icd9("428")   == "circulatory"
        assert categorise_icd9("250")   == "diabetes"
        assert categorise_icd9("486")   == "respiratory"
        assert categorise_icd9("99999") == "other"
        assert categorise_icd9("?")     == "other"

    def test_build_features_shape(self):
        from src.features.engineer import build_features
        df = self._make_encounter()
        feat = build_features(df)
        assert len(feat.columns) > len(df.columns), "Feature mart must add columns"
        assert "clinical_risk_score" in feat.columns
        assert "prior_visits_total" in feat.columns
        assert "long_los" in feat.columns

    def test_risk_score_high_risk_patient(self):
        from src.features.engineer import build_features
        # Patient with multiple risk factors — should have high score
        df = self._make_encounter(
            number_inpatient=3, number_emergency=2,
            time_in_hospital=10, A1Cresult="None",
            change="Ch", num_medications=20,
        )
        feat = build_features(df)
        assert feat["clinical_risk_score"].iloc[0] > 0.5

    def test_prepare_model_input_no_nulls(self):
        from src.features.engineer import build_features, prepare_model_input
        df = self._make_encounter()
        feat = build_features(df)
        X, y, names = prepare_model_input(feat)
        assert X.isnull().sum().sum() == 0, "Model input must have no nulls"
        assert len(y) == len(X)


# ── Test: Ingestion ───────────────────────────────────────────────────────────

class TestIngestion:

    def test_validate_schema_passes_good_data(self):
        from src.ingestion.ingest import validate_schema, REQUIRED_COLUMNS
        df = pd.DataFrame({col: [1] for col in REQUIRED_COLUMNS})
        # Set string columns
        for col in ["race","gender","age","diag_1","diag_2","diag_3",
                    "max_glu_serum","A1Cresult","change","diabetesMed","readmitted"]:
            if col in df.columns:
                df[col] = "test"
        df["time_in_hospital"] = 3
        df["num_lab_procedures"] = 10
        df["num_procedures"] = 1
        df["num_medications"] = 5
        df["number_outpatient"] = 0
        df["number_emergency"] = 0
        df["number_inpatient"] = 0
        df["number_diagnoses"] = 3
        result = validate_schema(df)
        assert result["passed"] is True

    def test_validate_schema_catches_out_of_bounds(self):
        from src.ingestion.ingest import validate_schema, REQUIRED_COLUMNS
        df = pd.DataFrame({col: ["x"] for col in REQUIRED_COLUMNS})
        df["time_in_hospital"] = 999  # out of bounds
        df["num_lab_procedures"] = 5
        df["num_procedures"] = 1
        df["num_medications"] = 5
        df["number_outpatient"] = 0
        df["number_emergency"] = 0
        df["number_inpatient"] = 0
        df["number_diagnoses"] = 3
        df["encounter_id"] = range(len(df))
        result = validate_schema(df)
        assert result["passed"] is False


# ── Test: API request/response schema ─────────────────────────────────────────

class TestAPISchemas:

    def test_patient_encounter_valid(self):
        from src.api.server import PatientEncounter
        enc = PatientEncounter(
            time_in_hospital=5,
            number_inpatient=1,
            number_emergency=0,
            number_outpatient=2,
            number_diagnoses=7,
            num_medications=12,
            num_lab_procedures=30,
            age_midpoint=65.0,
            diabetesMed="Yes",
            change="No",
            A1Cresult="None",
        )
        assert enc.time_in_hospital == 5
        assert enc.diabetesMed == "Yes"

    def test_patient_encounter_rejects_bad_diabetes_med(self):
        from src.api.server import PatientEncounter
        from pydantic import ValidationError
        with pytest.raises((ValueError, ValidationError)):
            PatientEncounter(
                time_in_hospital=5, number_inpatient=0, number_emergency=0,
                number_outpatient=0, number_diagnoses=3, num_medications=5,
                num_lab_procedures=10, age_midpoint=60.0,
                diabetesMed="INVALID",
                change="No", A1Cresult="None",
            )

    def test_encounter_to_features_shape(self):
        from src.api.server import PatientEncounter, encounter_to_features, FEATURE_ORDER
        enc = PatientEncounter(
            time_in_hospital=5, number_inpatient=1, number_emergency=0,
            number_outpatient=0, number_diagnoses=7, num_medications=12,
            num_lab_procedures=30, age_midpoint=65.0, diabetesMed="Yes",
            change="Ch", A1Cresult=">7",
        )
        X = encounter_to_features(enc)
        assert X.shape == (1, len(FEATURE_ORDER))
        assert X.isnull().sum().sum() == 0


# ── Test: Monitoring ──────────────────────────────────────────────────────────

class TestMonitoring:

    def test_psi_identical_distributions(self):
        from src.monitoring.drift import population_stability_index
        data = np.random.normal(0, 1, 1000)
        psi = population_stability_index(data, data.copy())
        assert psi < 0.01, "PSI of identical distributions should be ~0"

    def test_psi_shifted_distributions(self):
        from src.monitoring.drift import population_stability_index
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(2, 1, 1000)  # large shift
        psi = population_stability_index(ref, cur)
        assert psi > 0.20, "PSI of very different distributions should be > 0.20"

    def test_performance_degradation_alert(self):
        from src.monitoring.drift import check_performance_degradation
        result = check_performance_degradation(baseline_auroc=0.70, current_auroc=0.62)
        assert result["alert"] is True
        assert result["auroc_drop"] > 0.05

    def test_performance_degradation_ok(self):
        from src.monitoring.drift import check_performance_degradation
        result = check_performance_degradation(baseline_auroc=0.70, current_auroc=0.69)
        assert result["alert"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
