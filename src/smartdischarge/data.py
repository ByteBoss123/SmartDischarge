"""Load the raw encounters and build the modeling cohort.

Cohort rules (each one is logged with its row count in the run report):
1. One encounter per patient: the first by encounter_id. Later encounters of the same
   patient are not independent and would leak the outcome across a train/test split.
2. Drop discharges where a 30-day readmission cannot happen or is not a quality signal:
   expired (11, 19, 20, 21) and hospice (13, 14).
3. Label: readmitted within 30 days ("<30") = 1; ">30" and "NO" = 0.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

EXPIRED_OR_HOSPICE = {11, 13, 14, 19, 20, 21}


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["?"], low_memory=False)
    required = {"encounter_id", "patient_nbr", "readmitted", "discharge_disposition_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return df


def build_cohort(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    steps = [{"step": "raw encounters", "rows": int(len(raw))}]

    df = raw.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    steps.append({"step": "first encounter per patient", "rows": int(len(df))})

    df = df[~df["discharge_disposition_id"].isin(EXPIRED_OR_HOSPICE)]
    steps.append({"step": "exclude expired and hospice discharges", "rows": int(len(df))})

    df = df.copy()
    df["label"] = (df["readmitted"] == "<30").astype(int)
    steps.append({"step": "positives (readmitted <30 days)", "rows": int(df["label"].sum())})
    return df.reset_index(drop=True), steps
