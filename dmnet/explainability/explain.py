"""
Phase 6: Explainability — SHAP + LIME
======================================
Provides model-agnostic explanations for DMNet predictions.

SHAP (SHapley Additive exPlanations):
  - Theoretically grounded in game theory (Shapley values)
  - Tells us: "How much did each feature contribute to moving the
    prediction away from the average prediction?"
  - Global: average |SHAP| across all patients
  - Local : per-patient feature attribution bar chart

LIME (Local Interpretable Model-agnostic Explanations):
  - Fits a simple linear model in the local neighborhood of one prediction
  - Tells us: "In this region of input space, which features matter most?"
  - Better for explaining individual predictions to clinical users

Run: python explainability/explain.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tensorflow as tf
import shap
from lime import lime_tabular

FEATURE_COLS = [
    "age", "bmi", "glucose", "insulin", "blood_pressure",
    "pregnancies", "hba1c", "fasting_glucose",
    "physical_activity", "smoking_history", "family_history",
]

EXPL_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(EXPL_DIR, "..", "models", "dmnet_best.h5")
DATA_DIR   = os.path.join(EXPL_DIR, "..", "data")


def load_model_and_data():
    """Load trained model and test data."""
    model  = tf.keras.models.load_model(MODEL_PATH)
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    return model, X_test, y_test


def flatten_sequences(X):
    """Mean across timesteps: (n, timesteps, features) -> (n, features)"""
    return X.mean(axis=1)


def predict_proba_flat(X_flat, model):
    """
    Convert flat 2D -> tiled 3D sequences for DMNet prediction.
    Returns (n, 2) probability array for LIME compatibility.
    """
    n_timesteps = 12
    X_3d = np.tile(X_flat[:, np.newaxis, :], (1, n_timesteps, 1)).astype(np.float32)
    probs = model.predict(X_3d, verbose=0).flatten()
    return np.column_stack([1 - probs, probs])


# ──────────────────────────────────────────────────────────────────────────────
#  SHAP Explanations
# ──────────────────────────────────────────────────────────────────────────────

def run_shap(model, X_test, save_dir, n_background=100):
    """Compute SHAP values using KernelExplainer (model-agnostic)."""
    print("\n[SHAP] Computing SHAP values (this may take ~1 minute)...")

    X_flat     = flatten_sequences(X_test)
    background = shap.kmeans(X_flat[:n_background], 20)

    def predict_fn(X):
        return predict_proba_flat(X, model)[:, 1]

    explainer   = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_flat[:100], nsamples=200)

    # Global bar plot
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_flat[:100],
                      feature_names=FEATURE_COLS, plot_type="bar", show=False)
    plt.title("SHAP Global Feature Importance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(save_dir, "shap_global_bar.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("OK  SHAP global bar chart ->", out)

    # Beeswarm plot
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_flat[:100],
                      feature_names=FEATURE_COLS, show=False)
    plt.title("SHAP Beeswarm - Feature Impact Distribution", fontsize=12)
    plt.tight_layout()
    out2 = os.path.join(save_dir, "shap_beeswarm.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print("OK  SHAP beeswarm ->", out2)

    # Waterfall plot for patient 0
    plt.figure(figsize=(9, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values        = shap_values[0],
            base_values   = explainer.expected_value,
            data          = X_flat[0],
            feature_names = FEATURE_COLS,
        ),
        show=False,
    )
    plt.title("SHAP Waterfall - Patient #0", fontsize=12)
    plt.tight_layout()
    out3 = os.path.join(save_dir, "shap_local_waterfall.png")
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print("OK  SHAP waterfall ->", out3)

    return shap_values, explainer.expected_value


# ──────────────────────────────────────────────────────────────────────────────
#  LIME Explanations
# ──────────────────────────────────────────────────────────────────────────────

def run_lime(model, X_test, save_dir):
    """Compute LIME local explanation for patient #0."""
    print("\n[LIME] Computing LIME explanation...")

    X_flat = flatten_sequences(X_test)

    explainer = lime_tabular.LimeTabularExplainer(
        training_data = X_flat,
        feature_names = FEATURE_COLS,
        class_names   = ["Non-Diabetic", "Diabetic"],
        mode          = "classification",
        random_state  = 42,
    )

    explain_fn  = lambda X: predict_proba_flat(X, model)
    explanation = explainer.explain_instance(
        data_row     = X_flat[0],
        predict_fn   = explain_fn,
        num_features = len(FEATURE_COLS),
        num_samples  = 500,
    )

    # Save plot
    fig = explanation.as_pyplot_figure(label=1)
    fig.set_size_inches(9, 5)
    plt.title("LIME Local Explanation - Patient #0", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(save_dir, "lime_local_explanation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("OK  LIME explanation ->", out)

    # Save contributions as text
    # utf-8 encoding avoids Windows cp1252 UnicodeEncodeError
    contributions = explanation.as_list(label=1)
    txt_path      = os.path.join(save_dir, "lime_contributions.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("LIME Local Explanation - Patient #0\n")
        f.write("=" * 45 + "\n")
        prob = predict_proba_flat(X_flat[[0]], model)[0, 1]
        f.write("Predicted Probability (Diabetic): {:.4f}\n\n".format(prob))
        f.write("Feature Contributions:\n")
        for feat, contrib in contributions:
            direction = "increases risk" if contrib > 0 else "decreases risk"
            f.write("  {:<40s}: {:+.4f}  ({})\n".format(feat, contrib, direction))
    print("OK  LIME text contributions ->", txt_path)

    return explanation


def explain_single_patient(model, patient_features):
    """
    API-friendly explanation for one patient snapshot.
    patient_features: flat numpy array of shape (n_features,)
    Returns dict of top 5 LIME feature contributions.
    """
    X_train      = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_train_flat = X_train.mean(axis=1)

    explainer = lime_tabular.LimeTabularExplainer(
        training_data = X_train_flat,
        feature_names = FEATURE_COLS,
        class_names   = ["Non-Diabetic", "Diabetic"],
        mode          = "classification",
        random_state  = 42,
    )

    explain_fn  = lambda X: predict_proba_flat(X, model)
    explanation = explainer.explain_instance(
        data_row     = patient_features,
        predict_fn   = explain_fn,
        num_features = 5,
        num_samples  = 300,
    )

    contributions = explanation.as_list(label=1)
    return {feat: round(float(val), 4) for feat, val in contributions}


def run_all_explanations():
    os.makedirs(EXPL_DIR, exist_ok=True)
    model, X_test, y_test = load_model_and_data()
    run_shap(model, X_test, EXPL_DIR)
    run_lime(model, X_test, EXPL_DIR)
    print("\nAll explanations generated in:", EXPL_DIR)


if __name__ == "__main__":
    print("=" * 60)
    print("  DMNet - Explainability Module (SHAP + LIME)")
    print("=" * 60)
    run_all_explanations()