"""
utils/inference.py  —  FIXED VERSION
"""

import os
import numpy as np
import joblib
import tensorflow as tf
from typing import Dict, Any

# ── MUST match EXACTLY what preprocessing.py used during training ──────────────
FEATURE_COLS = [
    "age", "bmi", "glucose", "insulin", "blood_pressure",
    "pregnancies", "hba1c", "fasting_glucose",
    "physical_activity", "smoking", "family_history",   # ← "smoking" not "smoking_history"
]

N_TIMESTEPS = 12
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "dmnet_best.h5")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
DATA_DIR    = os.path.join(BASE_DIR, "data")

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
    Convert flat feature dict → 3D tensor (1, 12, 11) for DMNet.
    
    FIX: Add small realistic noise across timesteps instead of
    identical copies — prevents sigmoid saturation.
    """
    scaler = get_scaler()

    # Map incoming keys — handle both "smoking" and "smoking_history"
    normalized = {}
    for col in FEATURE_COLS:
        if col in features:
            normalized[col] = features[col]
        elif col == "smoking" and "smoking_history" in features:
            normalized[col] = features["smoking_history"]
        elif col == "smoking_history" and "smoking" in features:
            normalized[col] = features["smoking"]
        else:
            normalized[col] = 0.0

    # Build base feature vector
    feat_vec = np.array([[normalized[col] for col in FEATURE_COLS]], dtype=np.float32)

    # Scale using fitted scaler
    feat_scaled = scaler.transform(feat_vec)  # (1, 11)

    # ── FIX: build sequence with small noise instead of identical copies ──────
    # Training data had natural variation; identical timesteps = out-of-distribution
    np.random.seed(42)
    noise_scale = 0.01  # tiny — just enough to break identical pattern
    sequence = np.zeros((1, N_TIMESTEPS, len(FEATURE_COLS)), dtype=np.float32)
    for t in range(N_TIMESTEPS):
        noise = np.random.normal(0, noise_scale, feat_scaled.shape)
        sequence[0, t, :] = np.clip(feat_scaled + noise, 0.0, 1.0)

    # Center timestep gets exact values (no noise)
    sequence[0, N_TIMESTEPS // 2, :] = feat_scaled

    return sequence


def predict(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Full prediction pipeline for a single patient.
    """
    model = get_model()
    seq   = preprocess_patient(features)

    prob  = float(model.predict(seq, verbose=0)[0][0])
    pred  = int(prob >= 0.5)

    if prob < 0.40:
        risk = "Low"
    elif prob < 0.65:
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
    LIME explanation — fixed to use same key normalization.
    """
    from lime import lime_tabular

    model  = get_model()
    scaler = get_scaler()

    # Normalize keys same way as preprocess_patient
    normalized = {}
    for col in FEATURE_COLS:
        if col in features:
            normalized[col] = features[col]
        elif col == "smoking" and "smoking_history" in features:
            normalized[col] = features["smoking_history"]
        else:
            normalized[col] = 0.0

    feat_vec    = np.array([[normalized[col] for col in FEATURE_COLS]])
    feat_scaled = scaler.transform(feat_vec).flatten()

    # Load training background data
    X_train      = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_train_flat = X_train.mean(axis=1)  # (n, 11) — average across timesteps

    def predict_proba_flat(X_flat):
        """LIME calls this with (n_samples, n_features) — expand to 3D."""
        np.random.seed(42)
        n = X_flat.shape[0]
        X3d = np.zeros((n, N_TIMESTEPS, len(FEATURE_COLS)), dtype=np.float32)
        for t in range(N_TIMESTEPS):
            noise = np.random.normal(0, 0.01, X_flat.shape)
            X3d[:, t, :] = np.clip(X_flat + noise, 0.0, 1.0)
        X3d[:, N_TIMESTEPS // 2, :] = X_flat  # center = exact
        p = model.predict(X3d, verbose=0).flatten()
        return np.column_stack([1 - p, p])

    explainer = lime_tabular.LimeTabularExplainer(
        training_data = X_train_flat,
        feature_names = FEATURE_COLS,
        class_names   = ["Non-Diabetic", "Diabetic"],
        mode          = "classification",
        random_state  = 42,
    )
    explanation = explainer.explain_instance(
        data_row     = feat_scaled,
        predict_fn   = predict_proba_flat,
        num_features = len(FEATURE_COLS),
        num_samples  = 300,
    )

    contribs = explanation.as_list(label=1)
    return {feat: round(float(val), 4) for feat, val in contribs}