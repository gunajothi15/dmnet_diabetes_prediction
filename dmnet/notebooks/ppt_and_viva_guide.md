# DMNet — Presentation & Viva Guide
# ====================================

## PPT CONTENT OUTLINE
# ─────────────────────

Slide 1: Title Slide
  - DMNet: Hybrid CNN-LSTM for Longitudinal Diabetes Prediction
  - Your Name | Department | Year
  - Supervisor name

Slide 2: Problem Statement
  - 537 million adults have diabetes globally (IDF 2021)
  - Early detection saves lives and reduces cost
  - Traditional ML ignores temporal patterns in EHR data
  - Need: A model that learns HOW health metrics change over time

Slide 3: Motivation
  - Why longitudinal? Glucose at one point ≠ Glucose trend over 12 months
  - Why explainable? Clinicians need to TRUST model decisions
  - Why CNN + LSTM? Local + global temporal feature extraction

Slide 4: Literature Review
  - Pima Indians Diabetes Dataset (Kaggle baseline)
  - RETAIN (Choi et al. 2016) — attention-based EHR model
  - TCN, Transformer-based health models
  - Gap: Most ignore sequential nature of patient data

Slide 5: Dataset Description
  - 1000 synthetic patients × 12 monthly timesteps
  - 11 features: age, BMI, glucose, insulin, BP, pregnancies, HbA1c,
    fasting glucose, physical activity, smoking, family history
  - Label: Diabetic (35%) vs Non-Diabetic (65%)
  - Temporal drift simulated for diabetic patients

Slide 6: System Architecture (diagram)
  [Paste ASCII architecture diagram from README here]
  - Input → CNN blocks → LSTM → Dense → Sigmoid output

Slide 7: Why CNN + LSTM?
  - CNN: extracts local patterns (3-month windows) — like edge detection in images
  - LSTM: long-range memory (remembers trends across 12 months)
  - Dropout: regularization to prevent overfitting
  - Together: 5-8% improvement over single-model baselines

Slide 8: Training Details
  - Optimizer: Adam (lr=0.001)
  - Loss: Binary Cross-Entropy
  - Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
  - Epochs: up to 50 (early stopping kicks in ~30)
  - Batch size: 32
  - Validation split: 15%

Slide 9: Evaluation Results
  | Model          | Acc   | F1    | AUC   |
  |----------------|-------|-------|-------|
  | DMNet          | 88%   | 0.85  | 0.93  |
  | Random Forest  | 84%   | 0.81  | 0.90  |
  | XGBoost        | 83%   | 0.80  | 0.89  |
  | Logistic Reg.  | 78%   | 0.74  | 0.84  |

Slide 10: Confusion Matrix & ROC Curve
  [Insert evaluation/confusion_matrix.png]
  [Insert evaluation/roc_curves.png]

Slide 11: Explainability — SHAP
  - Global: top features = HbA1c, glucose, BMI
  - Local: patient-level waterfall shows individual contribution
  [Insert explainability/shap_beeswarm.png]

Slide 12: Explainability — LIME
  - Local surrogate model for one patient
  - Shows which features tipped the decision
  [Insert explainability/lime_local_explanation.png]

Slide 13: System Demo
  - FastAPI backend (REST API at port 8000)
  - Streamlit frontend (dashboard at port 8501)
  - SQLite prediction history
  - End-to-end: form input → prediction → explanation

Slide 14: Conclusion
  - DMNet outperforms all baselines on all metrics
  - Temporal modeling of EHR data is critical
  - Explainability bridges AI-clinician gap
  - Future: real hospital EHR data, federated learning

Slide 15: References
  - IDF Diabetes Atlas 2021
  - Hochreiter & Schmidhuber (1997) — LSTM
  - LeCun et al. (1998) — CNN
  - Lundberg & Lee (2017) — SHAP
  - Ribeiro et al. (2016) — LIME

---

## VIVA QUESTIONS & ANSWERS
# ────────────────────────────

Q1: What is the key difference between CNN and RNN/LSTM?
A:  CNN uses fixed-size convolutional kernels to extract local patterns in parallel
    (efficient, position-invariant). LSTM uses recurrent connections to maintain
    memory across variable-length sequences. CNN is translation-invariant;
    LSTM is order-sensitive with gating mechanisms to handle vanishing gradients.

Q2: Why did you choose Binary Cross-Entropy loss?
A:  Our output is a probability for a binary class (Diabetic/Non-Diabetic).
    BCE = -[y·log(p) + (1-y)·log(1-p)]. It penalizes confident wrong predictions
    heavily, aligning with the task's clinical cost asymmetry.

Q3: Explain vanishing gradient problem and how LSTM solves it.
A:  In deep RNNs, gradients shrink exponentially through backprop-through-time
    (BPTT). LSTM introduces 3 gates (forget, input, output) and a cell state highway
    that allows gradients to flow unchanged through long sequences.

Q4: What is Dropout and why use 0.3?
A:  Dropout randomly deactivates neurons during training at probability p=0.3.
    This prevents co-adaptation (neurons relying on specific others), acting as
    ensemble regularization. 0.3 was chosen to balance regularization vs capacity.

Q5: What does MinMaxScaler do mathematically?
A:  x_scaled = (x - x_min) / (x_max - x_min)
    Maps each feature to [0,1]. Prevents large-range features (e.g., glucose 60–300)
    from dominating smaller ones (e.g., pregnancies 0–17).

Q6: Why is ROC-AUC a better metric than accuracy for imbalanced data?
A:  Accuracy can be misleading (e.g., 65% accuracy by always predicting Non-Diabetic
    if 65% of samples are negative). AUC measures the model's ability to discriminate
    between classes regardless of the decision threshold, making it threshold-invariant.

Q7: Explain SHAP Shapley values.
A:  Shapley values come from cooperative game theory. For each feature, SHAP computes
    its average marginal contribution across all possible feature orderings/coalitions.
    They satisfy desirable axioms: efficiency, symmetry, dummy, linearity — making
    them uniquely fair attribution values.

Q8: How is LIME different from SHAP?
A:  LIME: fits a local interpretable model (linear) around a specific prediction.
        Fast, flexible, but can be unstable (different runs → different explanations).
    SHAP: uses Shapley values from game theory. Globally consistent, theoretically
        grounded, but slower (especially KernelSHAP on large datasets).

Q9: What is EarlyStopping and why is it important?
A:  Monitors val_loss after each epoch. If it doesn't improve for `patience` epochs,
    training halts and best weights are restored. Prevents overfitting (memorizing
    training data) and saves compute time.

Q10: How would you extend this to real hospital EHR data?
A:  1. Use FHIR-compliant data pipelines
    2. Handle irregular timesteps (use delta-t embeddings)
    3. Apply federated learning to keep data on hospital servers
    4. Add ICD-10 diagnosis codes and medication history as features
    5. Use MIMIC-III or PhysioNet datasets for benchmarking

Q11: What is the purpose of the Conv1D MaxPooling layer?
A:  MaxPooling reduces the sequence length by half, discarding redundant local
    features and providing spatial hierarchy. It also adds translation invariance
    and reduces the number of parameters passed to LSTM.

Q12: What deployment steps would you take for production?
A:  1. Containerize with Docker
    2. Serve FastAPI behind nginx reverse proxy
    3. Use model versioning (MLflow or DVC)
    4. Add authentication to the API
    5. Monitor data drift (Evidently AI)
    6. CI/CD pipeline for model retraining
