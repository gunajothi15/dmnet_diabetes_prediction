"""
Phase 5: Model Evaluation
==========================
Evaluates DMNet and compares with baseline models:
  - Logistic Regression
  - Random Forest
  - XGBoost

Metrics computed:
  - Accuracy    : (TP+TN) / (TP+TN+FP+FN)
  - Precision   : TP / (TP+FP)  — how many predicted positives are real
  - Recall      : TP / (TP+FN)  — how many actual positives were caught
  - F1-Score    : 2*(Precision*Recall)/(Precision+Recall)
  - ROC-AUC     : area under the Receiver Operating Characteristic curve

Outputs:
  - evaluation/confusion_matrix.png
  - evaluation/roc_curves.png
  - evaluation/model_comparison.png
  - evaluation/metrics_report.txt

Run: python evaluation/evaluate.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.metrics       import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    classification_report,
)
import tensorflow as tf

EVAL_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(EVAL_DIR, "..", "models", "dmnet_best.h5")
DATA_DIR    = os.path.join(EVAL_DIR, "..", "data")


def load_test_data():
    """Load preprocessed test arrays saved by train.py."""
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    return X_train, X_test, y_train, y_test


def evaluate_dmnet(X_test, y_test):
    """Load saved DMNet and compute predictions + probabilities."""
    model  = tf.keras.models.load_model(MODEL_PATH)
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    return y_pred, y_prob


def train_baselines(X_train, y_train, X_test):
    """
    Train and predict with 3 baseline classifiers.
    We flatten sequences to 2D for sklearn models: (n, timesteps*features).
    """
    Xtr_2d = X_train.reshape(len(X_train), -1)
    Xte_2d = X_test.reshape(len(X_test),   -1)

    baselines = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
    }

    # Optional XGBoost
    try:
        from xgboost import XGBClassifier
        baselines["XGBoost"] = XGBClassifier(
            n_estimators=100, use_label_encoder=False,
            eval_metric="logloss", random_state=42
        )
    except ImportError:
        print("⚠️  XGBoost not installed — skipping.")

    results = {}
    for name, clf in baselines.items():
        clf.fit(Xtr_2d, y_train)
        y_pred = clf.predict(Xte_2d)
        y_prob = clf.predict_proba(Xte_2d)[:, 1]
        results[name] = (y_pred, y_prob)
        print(f"✅ {name} trained")

    return results


def compute_metrics(y_true, y_pred, y_prob, name="Model") -> dict:
    """Compute and print all evaluation metrics."""
    m = {
        "Accuracy" : accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall"   : recall_score(y_true, y_pred, zero_division=0),
        "F1-Score" : f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC"  : roc_auc_score(y_true, y_prob),
    }
    print(f"\n── {name} Metrics ──")
    for k, v in m.items():
        print(f"   {k:12s}: {v:.4f}")
    return m


def plot_confusion_matrix(y_true, y_pred, title: str, save_path: str):
    """Plot a styled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non-Diabetic", "Diabetic"],
        yticklabels=["Non-Diabetic", "Diabetic"],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title(title,        fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Confusion matrix → {save_path}")


def plot_roc_curves(y_test, all_probs: dict, save_path: str):
    """Plot overlaid ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2"]

    for (name, y_prob), color in zip(all_probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc          = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ ROC curves → {save_path}")


def plot_model_comparison(metrics_dict: dict, save_path: str):
    """Grouped bar chart comparing all models across metrics."""
    model_names = list(metrics_dict.keys())
    metric_names= ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    x           = np.arange(len(metric_names))
    width       = 0.2
    colors      = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (model, color) in enumerate(zip(model_names, colors)):
        vals = [metrics_dict[model][m] for m in metric_names]
        ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Model comparison chart → {save_path}")


def run_evaluation():
    print("=" * 60)
    print("  DMNet Evaluation Pipeline")
    print("=" * 60)
    os.makedirs(EVAL_DIR, exist_ok=True)

    # Load data
    X_train, X_test, y_train, y_test = load_test_data()

    # DMNet predictions
    y_pred_dm, y_prob_dm = evaluate_dmnet(X_test, y_test)
    dm_metrics           = compute_metrics(y_test, y_pred_dm, y_prob_dm, "DMNet")

    # Baselines
    baseline_preds = train_baselines(X_train, y_train, X_test)
    all_metrics    = {"DMNet": dm_metrics}
    all_probs      = {"DMNet": y_prob_dm}

    for name, (y_pred, y_prob) in baseline_preds.items():
        m = compute_metrics(y_test, y_pred, y_prob, name)
        all_metrics[name] = m
        all_probs[name]   = y_prob

    # Plots
    plot_confusion_matrix(
        y_test, y_pred_dm, "DMNet Confusion Matrix",
        os.path.join(EVAL_DIR, "confusion_matrix.png")
    )
    plot_roc_curves(y_test, all_probs, os.path.join(EVAL_DIR, "roc_curves.png"))
    plot_model_comparison(all_metrics,  os.path.join(EVAL_DIR, "model_comparison.png"))

    # Text report
    report_path = os.path.join(EVAL_DIR, "metrics_report.txt")
    with open(report_path, "w") as f:
        f.write("DMNet Classification Report\n")
        f.write("=" * 40 + "\n")
        f.write(classification_report(y_test, y_pred_dm,
                target_names=["Non-Diabetic", "Diabetic"]))
        f.write("\nAll Models Summary\n" + "=" * 40 + "\n")
        for model, m in all_metrics.items():
            f.write(f"\n{model}:\n")
            for k, v in m.items():
                f.write(f"  {k}: {v:.4f}\n")
    print(f"✅ Metrics report → {report_path}")

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    run_evaluation()
