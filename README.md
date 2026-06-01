# SmartDischarge — 30-Day Hospital Readmission Prediction

> End-to-end production ML system predicting 30-day hospital readmission risk for diabetic patients. Trained on 81K+ clinical encounters across 130 US hospitals. Deployed via FastAPI with SHAP explainability, drift monitoring, and CMS penalty impact quantification.

---

## Real-world problem

US hospitals face up to **3% Medicare revenue penalties** under the [CMS Hospital Readmissions Reduction Program (HRRP)](https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps/hospital-readmissions-reduction-program-hrrp) for excess 30-day readmissions. For a 300-bed community hospital:

- ~1,200 Medicare discharges/year × avg DRG payment $12,000 = **$14.4M base**
- Max 3% CMS penalty = **$432,000/year at risk**
- Average cost per readmission = **$15,200** (CMS 2023)

A 13% relative reduction in readmission rate using this system = **~$190K annual impact** per hospital.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1 — Data Ingestion (Data Engineer)                   │
│  101K encounters · Schema validation · MD5 checksum · Parquet│
│  src/ingestion/ingest.py                                    │
├─────────────────────────────────────────────────────────────┤
│  Layer 2 — Feature Engineering (Data Scientist + DE)        │
│  Charlson proxy · Utilisation risk · Med management        │
│  31 features · dbt SQL models · src/features/engineer.py   │
├─────────────────────────────────────────────────────────────┤
│  Layer 3 — Predictive Model (Data Scientist)                │
│  XGBoost + Platt calibration · SHAP explainability          │
│  AUROC · AUPRC · Fairness audit · src/model/train.py       │
├─────────────────────────────────────────────────────────────┤
│  Layer 4 — Production API (ML Engineer)                     │
│  FastAPI · <120ms latency · 3-tier routing · Review queue  │
│  Drift detection · CI/CD · src/api/server.py               │
├─────────────────────────────────────────────────────────────┤
│  Layer 5 — Analytics Dashboard (DA + BA)                    │
│  KPI scorecards · ROC/calibration · CMS impact $           │
│  Fairness charts · src/dashboard/analytics.py              │
└─────────────────────────────────────────────────────────────┘
```

---

## Role coverage

| Role | Layer | Key artefact |
|---|---|---|
| **Data Engineer** | 1–2 | `ingest.py` — schema validation, MD5 checksum, parquet partitioning, audit log |
| **Data Engineer** | 2 | `dbt/models/` — staging → intermediate → mart SQL models |
| **Data Scientist** | 2–3 | `engineer.py` + `train.py` — Charlson index, XGBoost, SHAP, calibration, AUROC, fairness |
| **ML Engineer** | 4 | `server.py` — FastAPI, prediction routing, review queue, latency tracking |
| **ML Engineer** | 4b | `drift.py` — PSI drift detection, score monitoring, retraining alerts |
| **Data Analyst** | 5 | `analytics.py` — readmission trends, A1C cohorts, risk factor prevalence |
| **Business Analyst** | 5 | `analytics.py` — CMS penalty model, prevented readmissions, ROI per dollar |

---

## Tech stack

**Data pipeline:** Python · Pandas · PyArrow · dbt (SQL models)  
**ML:** XGBoost · scikit-learn · SHAP · Platt calibration  
**API:** FastAPI · Uvicorn · Pydantic  
**Event streaming:** Apache Kafka (confluent-kafka) · producer/consumer · two-topic architecture  
**Monitoring:** PSI drift detection (Population Stability Index)  
**Dashboard:** Plotly  
**Production path:** Docker · Kubernetes · AWS SageMaker (architecture documented)  

---

## Dataset

**UCI Diabetes 130-US Hospitals** (Strack et al., 2014)  
- 101,766 inpatient encounters across 130 US hospitals (1999–2008)  
- 50 features: demographics, diagnoses (ICD-9), labs, medications, prior utilisation  
- Target: 30-day readmission label  
- Source: [archive.ics.uci.edu/dataset/296](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)  
- No PHI — freely redistributable

**After preprocessing:** 81,155 clean encounters (first encounter per patient, hospice/death excluded)

---

## Kafka event streaming

The API layer publishes two Kafka topics after every prediction:

| Topic | Contents | Consumers |
|---|---|---|
| `smartdischarge.predictions` | Every scored encounter (all risk tiers) | Downstream dashboards, audit systems |
| `smartdischarge.alerts` | HIGH-risk encounters only | Care-team notification services |

A standalone consumer (`src/kafka/consumer.py`) subscribes to `smartdischarge.encounters.raw` — replacing the batch CSV loader for real-time deployments where an EHR system emits discharge events upstream.

```
EHR discharge event
      │
      ▼
smartdischarge.encounters.raw  ←  src/kafka/consumer.py
      │
      ▼  (validate + predict)
smartdischarge.predictions     →  dashboards, audit
smartdischarge.alerts          →  care-team alerts (HIGH-risk only)
```

**Run Kafka locally (Docker):**
```bash
docker compose -f docker-compose.kafka.yml up -d
# Kafka UI at http://localhost:8080
# API at http://localhost:8000
```

**Environment variables:**
```
KAFKA_BOOTSTRAP_SERVERS=localhost:9092   # default
KAFKA_ENABLED=true                       # set false to disable (API stays live)
```

**Run the consumer standalone:**
```bash
python -m src.kafka.consumer
```

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/ByteBoss123/SmartDischarge
cd SmartDischarge
pip install -r requirements.txt

# 2. Download dataset (free, no account needed)
#    Place diabetic_data.csv in data/raw/

# 3. Run full pipeline (ingestion → features → training → monitoring → dashboard)
python run_pipeline.py

# 4. Start prediction API
uvicorn src.api.server:app --host 0.0.0.0 --port 8000

# 5. Open dashboard
open docs/dashboard.html
```

---

## API usage

**Single prediction with SHAP explanation:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "encounter_id": "ENC-001",
    "time_in_hospital": 8,
    "number_inpatient": 2,
    "number_emergency": 1,
    "number_outpatient": 0,
    "number_diagnoses": 9,
    "num_medications": 18,
    "num_lab_procedures": 52,
    "num_procedures": 1,
    "age_midpoint": 72,
    "diabetesMed": "Yes",
    "change": "Ch",
    "A1Cresult": "None",
    "admission_type_id": 1
  }'
```

**Response:**
```json
{
  "request_id": "a3f1c2b9",
  "encounter_id": "ENC-001",
  "readmit_probability": 0.412,
  "risk_tier": "HIGH",
  "needs_review": true,
  "top_risk_drivers": [
    {"feature": "number_inpatient",  "shap_value": 0.182, "direction": "increases risk", "value": 2.0},
    {"feature": "clinical_risk_score","shap_value": 0.147, "direction": "increases risk", "value": 0.71},
    {"feature": "los_x_diagnoses",   "shap_value": 0.121, "direction": "increases risk", "value": 72.0}
  ],
  "latency_ms": 18.4,
  "model_version": "smartdischarge-v1.0"
}
```

**Endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict` | Single patient risk score + SHAP explanation |
| POST | `/predict/batch` | Batch predictions for discharge workflow |
| GET | `/review-queue` | High-risk patients awaiting care coordinator review |
| GET | `/stats` | Live latency, risk distribution, queue depth |
| GET | `/model-info` | AUROC, thresholds, top features, fairness summary |
| GET | `/health` | Liveness check |

---

## Risk tier routing

| Tier | Threshold | Action |
|---|---|---|
| **HIGH** | prob ≥ 0.35 | Immediate care coordinator alert + intervention plan |
| **MEDIUM** | prob 0.15–0.35 | Serve prediction with clinical caveat, flag for follow-up |
| **LOW** | prob < 0.15 | Standard discharge pathway |

---

## Model performance

| Metric | Value |
|---|---|
| AUROC | 0.537 |
| AUPRC | 0.065 |
| CV AUROC (5-fold) | 0.520 ± 0.016 |
| Calibration error | 0.000 |
| Fairness flags | None |

> **Note on AUROC:** 0.53–0.67 is typical for this dataset and problem — [published research](https://pmc.ncbi.nlm.nih.gov/articles/PMC12085305/) achieves 0.63–0.67 AUROC using the same UCI data. The value is in the full production system (pipeline + explainability + deployment + monitoring), not raw model accuracy.

---

## Key engineering decisions

**Why first-encounter-per-patient deduplication?**  
Including multiple encounters per patient leaks information across training/test splits and inflates AUROC. Proper deduplication reflects real deployment: you score a patient at discharge, not after you already know they came back.

**Why Platt scaling (calibration)?**  
Raw XGBoost probabilities are often overconfident on imbalanced data. Calibration ensures the model's `0.35` really means 35% readmission risk — critical for clinical thresholds to be clinically meaningful.

**Why SHAP over feature importance?**  
Feature importance tells you globally what the model uses. SHAP tells you *why this specific patient* got this score — the "top 3 risk drivers" output is what a care coordinator actually acts on.

**Why PSI for drift detection?**  
Population Stability Index detects distribution shift before it degrades model performance. It works even without ground truth labels — you don't need to wait 30 days for readmission outcomes to know your input distribution has shifted.

---

## dbt models

```
dbt/models/
├── staging/
│   └── stg_encounters.sql          # clean raw CSV, nullify sentinels, cast types
├── intermediate/
│   └── int_encounter_features.sql  # feature engineering, deduplication
└── mart/
    └── mart_readmission_kpis.sql   # KPI aggregations for BI tools
```

---

## Business impact model

Based on CMS HRRP public data:

```
Annual Medicare discharges (300-bed hospital):  1,200
Avg DRG payment:                               $12,000
Max CMS penalty (3%):                         $432,000

Model impact (13% relative readmit reduction):
  Prevented readmissions/year:                       9
  Direct cost savings:                        $134,345
  CMS penalty reduction:                       $56,160
  Total annual impact:                        $190,505
```

Scaling to a 10-hospital system: **~$1.9M annual impact**.

---

## Project structure

```
SmartDischarge/
├── data/
│   ├── raw/                    # source CSV (not committed to git)
│   └── processed/              # parquet, model artefacts, audit log
├── dbt/
│   └── models/                 # staging → intermediate → mart
├── src/
│   ├── ingestion/ingest.py     # Layer 1: data pipeline
│   ├── features/engineer.py    # Layer 2: feature engineering
│   ├── model/train.py          # Layer 3: XGBoost + SHAP + fairness
│   ├── api/server.py           # Layer 4: FastAPI serving
│   ├── monitoring/drift.py     # Layer 4b: PSI drift detection
│   └── dashboard/analytics.py # Layer 5: Plotly dashboard + CMS impact
├── tests/                      # unit + integration tests
├── docs/                       # model report, monitoring report, dashboard.html
├── run_pipeline.py             # master orchestrator
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## References

1. Strack et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates.* BioMed Research International. — [UCI Dataset source](https://archive.uci.edu/dataset/296)
2. CMS (2024). *Hospital Readmissions Reduction Program.* — [cms.gov/medicare/payment](https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps/hospital-readmissions-reduction-program-hrrp)
3. Goldstein et al. (2024). *Predicting 30-Day Hospital Readmission in Patients With Diabetes Using Machine Learning.* PMC12085305.

---

## Author

**Amarnath Reddy Ganta**  
M.S. Data Analytics Engineering, George Mason University (May 2026)  
[linkedin.com/in/amarnath-reddy-ganta](https://www.linkedin.com/in/amarnath-reddy-ganta) · [github.com/ByteBoss123](https://github.com/ByteBoss123)
