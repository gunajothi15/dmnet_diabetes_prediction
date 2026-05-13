"""
Phase 1: Longitudinal Diabetes Dataset Generation
==================================================
Generates synthetic EHR-style time-series patient data.
Each patient has multiple timesteps (months) simulating real-world
longitudinal health monitoring.

Features:
  - age, bmi, glucose, insulin, blood_pressure
  - pregnancies, hba1c, fasting_glucose
  - physical_activity, smoking_history, family_history

Label: diabetic (0 = No, 1 = Yes)
"""

import numpy as np
import pandas as pd
import os

# ─── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ─── Config ────────────────────────────────────────────────────────────────────
N_PATIENTS    = 1000   # number of unique patients
N_TIMESTEPS   = 12     # months per patient (1 year)
DIABETIC_RATE = 0.35   # ~35% diabetic prevalence


def generate_patient_base(patient_id: int, is_diabetic: bool) -> dict:
    """
    Generate baseline (static) features for one patient.
    Diabetic patients have higher means for risk factors.
    """
    if is_diabetic:
        age            = np.random.randint(40, 75)
        bmi            = np.random.uniform(27, 42)
        family_history = np.random.choice([0, 1], p=[0.3, 0.7])
        smoking        = np.random.choice([0, 1], p=[0.4, 0.6])
        pregnancies    = np.random.randint(0, 8)
    else:
        age            = np.random.randint(20, 65)
        bmi            = np.random.uniform(18, 30)
        family_history = np.random.choice([0, 1], p=[0.7, 0.3])
        smoking        = np.random.choice([0, 1], p=[0.7, 0.3])
        pregnancies    = np.random.randint(0, 5)

    return {
        "patient_id"    : patient_id,
        "age"           : age,
        "bmi"           : round(bmi, 1),
        "pregnancies"   : pregnancies,
        "family_history": family_history,
        "smoking_history": smoking,
        "label"         : int(is_diabetic),
    }


def generate_timestep_features(base: dict, t: int) -> dict:
    """
    Generate time-varying features for one patient at timestep t.
    Diabetic patients show deteriorating glucose/HbA1c trends.
    """
    is_d = base["label"]

    # Glucose drifts upward over time for diabetic patients
    glucose_base = 140 + t * 2 if is_d else 90 + t * 0.5
    glucose      = np.clip(np.random.normal(glucose_base, 15), 60, 300)

    hba1c_base   = 7.5 + t * 0.1 if is_d else 5.0 + t * 0.02
    hba1c        = np.clip(np.random.normal(hba1c_base, 0.5), 4.0, 14.0)

    fasting_base = 130 + t * 1.5 if is_d else 85 + t * 0.3
    fasting_gluc = np.clip(np.random.normal(fasting_base, 12), 60, 250)

    insulin_base = 180 if is_d else 80
    insulin      = np.clip(np.random.normal(insulin_base, 30), 10, 400)

    bp_base      = 85 if is_d else 70
    blood_press  = np.clip(np.random.normal(bp_base, 10), 50, 140)

    phys_act     = np.random.choice([0, 1], p=[0.6, 0.4] if is_d else [0.3, 0.7])

    return {
        "patient_id"      : base["patient_id"],
        "timestep"        : t,
        "age"             : base["age"] + t // 12,   # age increases yearly
        "bmi"             : round(base["bmi"] + np.random.normal(0, 0.3), 1),
        "glucose"         : round(glucose, 1),
        "insulin"         : round(insulin, 1),
        "blood_pressure"  : round(blood_press, 1),
        "pregnancies"     : base["pregnancies"],
        "hba1c"           : round(hba1c, 2),
        "fasting_glucose" : round(fasting_gluc, 1),
        "physical_activity": phys_act,
        "smoking_history" : base["smoking_history"],
        "family_history"  : base["family_history"],
        "label"           : base["label"],
    }


def generate_dataset() -> pd.DataFrame:
    """
    Generate full longitudinal dataset for all patients.
    Returns a flat DataFrame with shape: (N_PATIENTS * N_TIMESTEPS, features).
    """
    records = []

    for pid in range(N_PATIENTS):
        is_diabetic = np.random.rand() < DIABETIC_RATE
        base        = generate_patient_base(pid, is_diabetic)

        for t in range(N_TIMESTEPS):
            row = generate_timestep_features(base, t)
            records.append(row)

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    print("=" * 50)
    print("  Generating Longitudinal Diabetes Dataset")
    print("=" * 50)

    df = generate_dataset()

    # Save to CSV
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "diabetes_longitudinal.csv")
    df.to_csv(out_path, index=False)

    print(f"✅ Dataset saved → {out_path}")
    print(f"   Shape    : {df.shape}")
    print(f"   Patients : {df['patient_id'].nunique()}")
    print(f"   Timesteps: {df['timestep'].nunique()}")
    print(f"\nLabel distribution:")
    print(df.groupby("patient_id")["label"].first().value_counts())
    print(f"\nSample rows:")
    print(df.head(3).to_string())
