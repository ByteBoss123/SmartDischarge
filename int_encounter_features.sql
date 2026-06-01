-- dbt/models/intermediate/int_encounter_features.sql
-- SmartDischarge: intermediate feature engineering layer
-- Builds clinical risk features from staged encounter data.
-- Materialized as table — referenced by mart layer and BI tools.

{{ config(materialized='table') }}

with base as (
    select * from {{ ref('stg_encounters') }}
),

-- Keep only first encounter per patient (prevent data leakage)
deduped as (
    select *
    from base
    qualify row_number() over (
        partition by patient_nbr
        order by encounter_id asc
    ) = 1
),

feature_mart as (
    select
        encounter_id,
        patient_nbr,
        age_midpoint,
        race,
        gender,
        time_in_hospital,
        num_lab_procedures,
        num_procedures,
        num_medications,
        number_outpatient,
        number_emergency,
        number_inpatient,
        number_diagnoses,
        max_glu_serum,
        A1Cresult,
        change,
        diabetesMed,
        readmit_30d,

        -- ── Utilisation risk features ───────────────────────────────────────
        number_outpatient + number_emergency + number_inpatient
            as prior_visits_total,
        case when number_inpatient > 0 then 1 else 0 end
            as has_prior_inpatient,
        case when number_emergency > 0 then 1 else 0 end
            as has_prior_emergency,
        case when (number_outpatient + number_emergency + number_inpatient) > 5
             then 1 else 0 end
            as high_prior_utilisation,
        case when time_in_hospital > 7 then 1 else 0 end
            as long_los,
        time_in_hospital * number_diagnoses
            as los_x_diagnoses,

        -- ── Medication & lab features ───────────────────────────────────────
        case when A1Cresult != 'None' then 1 else 0 end
            as a1c_tested,
        case when A1Cresult in ('>7', '>8') then 1 else 0 end
            as a1c_abnormal,
        case when max_glu_serum != 'None' then 1 else 0 end
            as glucose_tested,
        case when change = 'Ch' then 1 else 0 end
            as med_changed,
        case when diabetesMed = 'Yes' then 1 else 0 end
            as on_diabetes_med,
        case when num_medications > 15 then 1 else 0 end
            as polypharmacy,
        case when num_lab_procedures > 50 then 1 else 0 end
            as high_lab_burden,

        -- ── Admission features ──────────────────────────────────────────────
        case when admission_type_id = 1 then 1 else 0 end
            as emergency_admission,
        case when admission_type_id = 2 then 1 else 0 end
            as urgent_admission,

        -- ── Demographic features ─────────────────────────────────────────────
        case when age_midpoint >= 70 then 1 else 0 end as elderly,
        case when age_midpoint between 40 and 69 then 1 else 0 end as working_age

    from deduped
)

select * from feature_mart
