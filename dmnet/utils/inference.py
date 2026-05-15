"""
utils/inference.py
==================
Shared inference utilities used by both the backend API and the Streamlit frontend.

Handles:
  - Loading the trained DMNet model
  - Loading the fitted scaler
  - Preprocessing a single patient's feature dict
  - Making predictions with risk categorization
  - Generating LIME explanation for one patient
"""

import os
import numpy as np
import joblib
import tensorflow as tf
from typing import Dict, Any

# ── Feature configuration ──────────────────────────────────────────────────────
FEATURE_COLS = [
    "age", "bmi", "glucose", "insulin", "blood_pressure",
    "pregnancies", "hba1c", "fasting_glucose",
    "physical_activity", "smoking_history", "family_history",
]

N_TIMESTEPS = 12   # model expects 12 timestep sequences
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "dmnet_best.h5")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
DATA_DIR    = os.path.join(BASE_DIR, "data")


# ── Lazy-load model and scaler once per process ────────────────────────────────
_model  = None
_scaler = None


def get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def get_scaler():
    global _scaler
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)
    return _scaler


def preprocess_patient(features: Dict[str, float]) -> np.ndarray:
    """
    Convert a flat feature dict into a 3D tensor ready for DMNet.

    Strategy: replicate the single snapshot across all N_TIMESTEPS.
    (In a real system you'd pass actual historical readings.)

    Returns:
        np.ndarray of shape (1, N_TIMESTEPS, n_features)
    """
    scaler = get_scaler()

    # Build ordered feature vector
    feat_vec = np.array([[features[col] for col in FEATURE_COLS]], dtype=np.float32)

    # Scale using the fitted scaler
    feat_scaled = scaler.transform(feat_vec)  # (1, n_features)

    # Tile across timesteps: (1, 12, 11)
    seq = np.tile(feat_scaled[:, np.newaxis, :], (1, N_TIMESTEPS, 1))
    return seq.astype(np.float32)


def predict(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Full prediction pipeline for a single patient.

    Args:
        features: dict with keys matching FEATURE_COLS

    Returns:
        {
          "probability"  : float  (0–1),
          "prediction"   : int    (0 = Non-Diabetic, 1 = Diabetic),
          "label"        : str,
          "risk_category": str    (Low / Medium / High),
        }
    """
    model = get_model()
    seq   = preprocess_patient(features)

    prob  = float(model.predict(seq, verbose=0)[0][0])
    pred  = int(prob >= 0.5)

    # Risk categorization thresholds
    if prob < 0.45:
        risk = "Low"
    elif prob < 0.70:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "probability"  : round(prob, 4),
        "prediction"   : pred,
        "label"        : "Diabetic" if pred == 1 else "Non-Diabetic",
        "risk_category": risk,
    }


def explain_prediction(features: Dict[str, float]) -> Dict[str, float]:
    """
    Generate a LIME explanation for a single patient snapshot.

    Returns a dict mapping feature name → contribution score.
    Positive = increases diabetes risk.
    Negative = decreases diabetes risk.
    """
    from lime import lime_tabular

    model  = get_model()
    scaler = get_scaler()

    # Load training data for LIME background
    X_train      = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_train_flat = X_train.mean(axis=1)  # (n, features)

    # Scale the input
    feat_vec    = np.array([[features[col] for col in FEATURE_COLS]])
    feat_scaled = scaler.transform(feat_vec).flatten()

    # Predict function for LIME
    def predict_proba_flat(X_flat):
        n_t = N_TIMESTEPS
        X3d = np.tile(X_flat[:, np.newaxis, :], (1, n_t, 1)).astype(np.float32)
        p   = model.predict(X3d, verbose=0).flatten()
        return np.column_stack([1 - p, p])

    explainer   = lime_tabular.LimeTabularExplainer(
        training_data = X_train_flat,
        feature_names = FEATURE_COLS,
        class_names   = ["Non-Diabetic", "Diabetic"],
        mode          = "classification",
        random_state  = 42,
    )
    explanation = explainer.explain_instance(
        data_row    = feat_scaled,
        predict_fn  = predict_proba_flat,
        num_features= len(FEATURE_COLS),
        num_samples = 300,
    )

    contribs = explanation.as_list(label=1)
    return {feat: round(float(val), 4) for feat, val in contribs}
