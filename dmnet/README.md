# 🩺 DMNet — Longitudinal Diabetes Prediction System

> Hybrid CNN-LSTM deep learning model for high-fidelity diabetes risk prediction  
> Based on: *"DMNet: Leveraging a Hybrid CNN-LSTM Architecture for High-Fidelity Longitudinal Diabetes Prediction"*

---

## 🏗 Architecture

```
Input (batch, 12 timesteps, 11 features)
    │
    ▼
┌─────────────────────────────────┐
│  Conv1D(64, kernel=3, ReLU)     │  ← Local temporal pattern extractor
│  MaxPooling1D(pool=2)           │
│  Conv1D(128, kernel=3, ReLU)    │
└─────────────────┬───────────────┘
                  │
                  ▼
┌─────────────────────────────────┐
│  LSTM(64)                       │  ← Long-range temporal dependency
└─────────────────┬───────────────┘
                  │
                  ▼
           Dropout(0.3)
                  │
                  ▼
           Dense(32, ReLU)
                  │
                  ▼
           Dense(1, Sigmoid)
                  │
                  ▼
        Diabetes Probability (0–1)
```

---

## 📁 Project Structure

```
dmnet/
│
├── data/
│   ├── generate_dataset.py      # Phase 1 — synthetic EHR data generation
│   ├── diabetes_longitudinal.csv # generated dataset (after running)
│   ├── X_train.npy / X_test.npy  # preprocessed arrays
│   └── y_train.npy / y_test.npy
│
├── models/
│   ├── dmnet_model.py           # Phase 3 — CNN-LSTM architecture
│   ├── dmnet_best.h5            # saved trained model (after training)
│   ├── scaler.pkl               # fitted MinMaxScaler
│   └── training_history.json   # loss/AUC curves data
│
├── utils/
│   ├── preprocessing.py         # Phase 2 — scaling, sequencing, splitting
│   └── inference.py             # shared prediction + LIME logic
│
├── evaluation/
│   ├── evaluate.py              # Phase 5 — metrics + baseline comparison
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── model_comparison.png
│   └── metrics_report.txt
│
├── explainability/
│   └── explain.py               # Phase 6 — SHAP + LIME visualizations
│
├── backend/
│   ├── app.py                   # Phase 7 — FastAPI REST API
│   └── predictions.db           # SQLite database (auto-created)
│
├── frontend/
│   └── streamlit_app.py         # Phase 8 — Streamlit dashboard
│
├── train.py                     # Phase 4 — full training pipeline
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone / Extract the project
```bash
cd dmnet
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Step 1 — Generate Dataset
```bash
python data/generate_dataset.py
```
Generates `data/diabetes_longitudinal.csv` with 1000 patients × 12 months.

### Step 2 — Train the Model
```bash
python train.py
```
Runs preprocessing, builds DMNet, trains with early stopping.  
Saves: `models/dmnet_best.h5`, `models/scaler.pkl`, `evaluation/learning_curves.png`

### Step 3 — Evaluate
```bash
python evaluation/evaluate.py
```
Computes accuracy, F1, ROC-AUC. Compares DMNet vs Logistic Regression, Random Forest, XGBoost.

### Step 4 — SHAP + LIME Explanations
```bash
python explainability/explain.py
```
Generates SHAP beeswarm, SHAP waterfall, LIME bar chart in `explainability/`.

### Step 5 — Start Backend API
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```
API docs: http://localhost:8000/docs

### Step 6 — Launch Streamlit App
```bash
streamlit run frontend/streamlit_app.py
```
Opens at: http://localhost:8501

---

## 📊 API Endpoints

| Method | Endpoint    | Description                          |
|--------|-------------|--------------------------------------|
| GET    | `/`         | Health check                         |
| POST   | `/predict`  | Get diabetes risk prediction         |
| POST   | `/explain`  | Get prediction + LIME explanation    |
| GET    | `/history`  | Retrieve past predictions (SQLite)   |

### Example `/predict` request:
```json
POST http://localhost:8000/predict
{
  "age": 52,
  "bmi": 30.5,
  "glucose": 145,
  "insulin": 180,
  "blood_pressure": 82,
  "pregnancies": 2,
  "hba1c": 7.2,
  "fasting_glucose": 128,
  "physical_activity": 0,
  "smoking_history": 1,
  "family_history": 1
}
```

### Response:
```json
{
  "probability": 0.8132,
  "prediction": 1,
  "label": "Diabetic",
  "risk_category": "High"
}
```

---

## 🧪 Features

- ✅ Synthetic longitudinal EHR data (1000 patients × 12 months)
- ✅ Hybrid CNN-LSTM model (DMNet architecture)
- ✅ SHAP global + local feature importance
- ✅ LIME local explanations
- ✅ Baseline comparison (LR, RF, XGBoost)
- ✅ FastAPI REST backend with SQLite history
- ✅ Professional Streamlit dashboard
- ✅ Risk categorization (Low / Medium / High)

---

## 🎓 Viva Q&A

**Q: Why use CNN before LSTM?**  
A: Conv1D acts as a local feature extractor — it identifies short-term patterns (e.g., 3-month glucose spikes) efficiently. LSTM then models the temporal progression of these patterns over the full 12-month window.

**Q: What is the significance of HbA1c in this model?**  
A: HbA1c reflects average blood glucose over 2–3 months — a key clinical diabetes marker. The LSTM layer specifically captures trends in HbA1c over time, which is a strong predictor.

**Q: Why MinMaxScaler instead of StandardScaler?**  
A: Neural networks (especially with sigmoid activations) are sensitive to input scale. MinMaxScaler bounds all features to [0,1], preventing any single feature from dominating gradient updates.

**Q: How does SHAP differ from LIME?**  
A: SHAP is globally consistent and theoretically grounded in Shapley values (game theory). LIME fits a local linear surrogate model — faster but less stable across similar inputs.

**Q: What is the ROC-AUC score measuring?**  
A: AUC = probability that the model ranks a randomly chosen diabetic patient higher than a randomly chosen non-diabetic patient. AUC > 0.85 indicates strong discrimination.

---

## 📈 Expected Results

| Model               | Accuracy | F1-Score | ROC-AUC |
|--------------------|----------|----------|---------|
| DMNet (CNN-LSTM)   | ~88%     | ~0.85    | ~0.93   |
| Random Forest      | ~84%     | ~0.81    | ~0.90   |
| XGBoost            | ~83%     | ~0.80    | ~0.89   |
| Logistic Regression| ~78%     | ~0.74    | ~0.84   |

---

## 📄 License

MIT License — Free to use for academic and educational purposes.
