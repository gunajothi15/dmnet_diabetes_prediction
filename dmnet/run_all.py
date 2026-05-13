"""
run_all.py — Full Pipeline Runner
===================================
Runs all phases sequentially:
  Phase 1: Generate dataset
  Phase 2: Preprocess
  Phase 3: Build model (verified via import)
  Phase 4: Train
  Phase 5: Evaluate
  Phase 6: SHAP + LIME explanations

Usage:
  python run_all.py              # full pipeline
  python run_all.py --skip-xai  # skip SHAP/LIME (faster demo)
"""

import os
import sys
import time
import argparse

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def banner(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def phase1_generate():
    banner("Phase 1: Dataset Generation")
    from data.generate_dataset import generate_dataset
    import pandas as pd

    path = "data/diabetes_longitudinal.csv"
    if not os.path.exists(path):
        df = generate_dataset()
        os.makedirs("data", exist_ok=True)
        df.to_csv(path, index=False)
        print(f"✅ Dataset generated → {path}")
    else:
        print(f"✅ Dataset already exists → {path}")


def phase4_train():
    banner("Phase 4: Model Training")
    from train import train
    train()


def phase5_evaluate():
    banner("Phase 5: Model Evaluation")
    # Import with path adjustment
    sys.path.insert(0, "evaluation")
    from evaluation.evaluate import run_evaluation
    run_evaluation()


def phase6_explain():
    banner("Phase 6: SHAP + LIME Explainability")
    from explainability.explain import run_all_explanations
    run_all_explanations()


def main():
    parser = argparse.ArgumentParser(description="DMNet Full Pipeline")
    parser.add_argument("--skip-xai", action="store_true",
                        help="Skip SHAP/LIME (faster)")
    args = parser.parse_args()

    start = time.time()

    phase1_generate()
    phase4_train()      # includes preprocessing internally
    phase5_evaluate()
    if not args.skip_xai:
        phase6_explain()
    else:
        print("\n⏭  Skipping XAI (--skip-xai flag set)")

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  ✅ All phases complete in {elapsed:.1f}s")
    print(f"  📊 Evaluation plots  → evaluation/")
    print(f"  🔍 Explanation plots → explainability/")
    print(f"  🤖 Trained model     → models/dmnet_best.h5")
    print(f"\n  To start the backend:")
    print(f"    uvicorn backend.app:app --port 8000 --reload")
    print(f"\n  To launch the dashboard:")
    print(f"    streamlit run frontend/streamlit_app.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
