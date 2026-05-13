"""
Phase 2: Preprocessing Pipeline
================================
Handles:
  1. Loading the longitudinal CSV
  2. Scaling numerical features (MinMaxScaler)
  3. Creating 3D sequences  → shape: (samples, timesteps, features)
  4. Train / test split
  5. Saving scaler for inference

Theory:
  CNN-LSTM requires input shape (batch, timesteps, features).
  Each "sample" is one patient's full 12-month window.
  We reshape the flat CSV into 3D tensors.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ─── Feature columns used for the model ────────────────────────────────────────
FEATURE_COLS = [
    "age", "bmi", "glucose", "insulin", "blood_pressure",
    "pregnancies", "hba1c", "fasting_glucose",
    "physical_activity", "smoking_history", "family_history",
]

TARGET_COL  = "label"
N_TIMESTEPS = 12   # sequence length per patient


def load_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and verify expected columns exist."""
    df = pd.read_csv(csv_path)
    required = FEATURE_COLS + [TARGET_COL, "patient_id", "timestep"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    print(f"✅ Loaded dataset: {df.shape}")
    return df


def scale_features(df: pd.DataFrame, scaler_path: str = None, fit: bool = True):
    """
    MinMax-scale all numerical feature columns.
    If fit=True  → fit + transform (training).
    If fit=False → transform only (inference).
    Returns scaled DataFrame and the fitted scaler.
    """
    scaler = MinMaxScaler()

    if fit:
        df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
        os.makedirs(os.path.dirname(scaler_path) if os.path.dirname(scaler_path) else ".", exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"✅ Scaler fitted and saved → {scaler_path}")
    else:
        scaler = joblib.load(scaler_path)
        df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])
        print(f"✅ Scaler loaded and applied from → {scaler_path}")

    return df, scaler


def create_sequences(df: pd.DataFrame):
    """
    Reshape flat longitudinal DataFrame into 3D numpy arrays.

    Input  : (N_patients * N_timesteps, features)
    Output : X → (N_patients, N_timesteps, n_features)
             y → (N_patients,)

    Each patient's records are sorted by timestep before stacking.
    """
    patients = df["patient_id"].unique()
    X_list, y_list = [], []

    for pid in patients:
        pdata = df[df["patient_id"] == pid].sort_values("timestep")
        if len(pdata) < N_TIMESTEPS:
            continue   # skip incomplete sequences

        x_seq = pdata[FEATURE_COLS].values[:N_TIMESTEPS]   # (12, 11)
        y_val = pdata[TARGET_COL].iloc[0]                   # label is constant

        X_list.append(x_seq)
        y_list.append(y_val)

    X = np.array(X_list, dtype=np.float32)   # (n_patients, 12, 11)
    y = np.array(y_list, dtype=np.float32)   # (n_patients,)

    print(f"✅ Sequences created: X={X.shape}, y={y.shape}")
    return X, y


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    """Standard 80/20 stratified train-test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    print(f"✅ Split → Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"   Train positives: {y_train.sum():.0f} / {len(y_train)}")
    print(f"   Test  positives: {y_test.sum():.0f} / {len(y_test)}")
    return X_train, X_test, y_train, y_test


def run_preprocessing(
    csv_path    : str = "data/diabetes_longitudinal.csv",
    scaler_path : str = "models/scaler.pkl",
    save_arrays : bool = True,
):
    """
    Full preprocessing pipeline.
    Returns X_train, X_test, y_train, y_test ready for model training.
    """
    df             = load_data(csv_path)
    df, scaler     = scale_features(df, scaler_path=scaler_path, fit=True)
    X, y           = create_sequences(df)
    X_tr, X_te, y_tr, y_te = split_data(X, y)

    if save_arrays:
        os.makedirs("data", exist_ok=True)
        np.save("data/X_train.npy", X_tr)
        np.save("data/X_test.npy",  X_te)
        np.save("data/y_train.npy", y_tr)
        np.save("data/y_test.npy",  y_te)
        print("✅ Arrays saved to data/")

    return X_tr, X_te, y_tr, y_te, scaler


if __name__ == "__main__":
    print("=" * 50)
    print("  Running Preprocessing Pipeline")
    print("=" * 50)
    run_preprocessing()
