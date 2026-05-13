"""
Phase 4: Training Pipeline
===========================
Orchestrates:
  1. Dataset generation (if not already done)
  2. Preprocessing
  3. Model build
  4. Model training with callbacks
  5. Saving trained model + history
  6. Plotting learning curves

Run: python train.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt

# ── Allow imports from project root ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.generate_dataset  import generate_dataset
from utils.preprocessing    import run_preprocessing
from models.dmnet_model     import build_dmnet, get_callbacks

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_CSV    = "data/diabetes_longitudinal.csv"
SCALER_PATH = "models/scaler.pkl"
MODEL_PATH  = "models/dmnet_best.h5"
HISTORY_PATH= "models/training_history.json"
PLOTS_DIR   = "evaluation"

EPOCHS      = 50
BATCH_SIZE  = 32
VAL_SPLIT   = 0.15   # fraction of training data used for validation


def ensure_dataset():
    """Generate dataset CSV if it doesn't exist yet."""
    if not os.path.exists(DATA_CSV):
        print("📊 Dataset not found — generating...")
        df = generate_dataset()
        os.makedirs("data", exist_ok=True)
        df.to_csv(DATA_CSV, index=False)
        print(f"✅ Dataset saved → {DATA_CSV}")
    else:
        print(f"✅ Dataset found → {DATA_CSV}")


def plot_learning_curves(history: dict, save_dir: str):
    """
    Plot loss and AUC curves for training and validation sets.
    Saved to evaluation/ as PNG files.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DMNet Training Learning Curves", fontsize=14, fontweight="bold")

    # Loss curve
    axes[0].plot(history["loss"],     label="Train Loss",  color="#2196F3")
    axes[0].plot(history["val_loss"], label="Val Loss",    color="#FF5722", linestyle="--")
    axes[0].set_title("Binary Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # AUC curve
    axes[1].plot(history["auc"],     label="Train AUC",  color="#4CAF50")
    axes[1].plot(history["val_auc"], label="Val AUC",    color="#9C27B0", linestyle="--")
    axes[1].set_title("ROC-AUC Score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, "learning_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Learning curves saved → {out}")


def train():
    print("\n" + "=" * 60)
    print("  DMNet — Diabetes Prediction Model Training")
    print("=" * 60)

    # ── Step 1: Data ────────────────────────────────────────────────────────
    ensure_dataset()
    X_train, X_test, y_train, y_test, _ = run_preprocessing(
        csv_path=DATA_CSV, scaler_path=SCALER_PATH
    )

    # ── Step 2: Build Model ─────────────────────────────────────────────────
    n_timesteps = X_train.shape[1]   # 12
    n_features  = X_train.shape[2]   # 11
    model = build_dmnet(n_timesteps=n_timesteps, n_features=n_features)
    model.summary()

    # ── Step 3: Train ───────────────────────────────────────────────────────
    print(f"\n🚀 Training for up to {EPOCHS} epochs (batch={BATCH_SIZE})...")
    cbs     = get_callbacks(model_save_path=MODEL_PATH)
    history = model.fit(
        X_train, y_train,
        epochs          = EPOCHS,
        batch_size      = BATCH_SIZE,
        validation_split= VAL_SPLIT,
        callbacks       = cbs,
        verbose         = 1,
    )

    # ── Step 4: Save History ────────────────────────────────────────────────
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(HISTORY_PATH, "w") as f:
        json.dump(hist_dict, f, indent=2)
    print(f"✅ Training history saved → {HISTORY_PATH}")

    # ── Step 5: Plot ────────────────────────────────────────────────────────
    plot_learning_curves(history.history, PLOTS_DIR)

    # ── Step 6: Quick evaluation on test set ────────────────────────────────
    print("\n📊 Test Set Evaluation:")
    results = model.evaluate(X_test, y_test, verbose=0)
    metrics = dict(zip(model.metrics_names, results))
    for k, v in metrics.items():
        print(f"   {k:12s}: {v:.4f}")

    print("\n✅ Training complete!")
    print(f"   Best model saved → {MODEL_PATH}")


if __name__ == "__main__":
    train()
