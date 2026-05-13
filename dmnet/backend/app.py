"""
Phase 7: FastAPI Backend
========================
REST API with these endpoints:

  GET  /           → health check
  POST /predict    → diabetes risk prediction
  POST /explain    → LIME feature explanation
  GET  /history    → patient prediction history (SQLite)

Run: uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import json
import sqlite3
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Project root on path ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.inference import predict, explain_prediction, FEATURE_COLS

# ── FastAPI app ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "DMNet Diabetes Prediction API",
    description = "Hybrid CNN-LSTM model for longitudinal diabetes risk prediction",
    version     = "1.0.0",
)

# Allow Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── SQLite setup ────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "predictions.db")


def init_db():
    """Create predictions table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    NOT NULL,
            features      TEXT    NOT NULL,
            probability   REAL    NOT NULL,
            prediction    INTEGER NOT NULL,
            label         TEXT    NOT NULL,
            risk_category TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()


init_db()


# ── Request / Response Schemas ──────────────────────────────────────────────────

class PatientFeatures(BaseModel):
    """Input schema — all features a clinician would enter for a patient."""
    age              : float = Field(..., ge=0,   le=120, example=52)
    bmi              : float = Field(..., ge=10,  le=70,  example=30.5)
    glucose          : float = Field(..., ge=40,  le=400, example=145)
    insulin          : float = Field(..., ge=0,   le=900, example=180)
    blood_pressure   : float = Field(..., ge=30,  le=200, example=82)
    pregnancies      : float = Field(..., ge=0,   le=20,  example=2)
    hba1c            : float = Field(..., ge=3,   le=15,  example=7.2)
    fasting_glucose  : float = Field(..., ge=40,  le=300, example=128)
    physical_activity: float = Field(..., ge=0,   le=1,   example=0)
    smoking_history  : float = Field(..., ge=0,   le=1,   example=1)
    family_history   : float = Field(..., ge=0,   le=1,   example=1)


class PredictionResponse(BaseModel):
    probability  : float
    prediction   : int
    label        : str
    risk_category: str


class ExplanationResponse(BaseModel):
    prediction  : PredictionResponse
    explanation : Dict[str, float]


# ── Endpoints ───────────────────────────────────────────────────────────────────

@app.get("/", summary="Health Check")
def root():
    return {"status": "ok", "model": "DMNet v1.0", "timestamp": datetime.utcnow().isoformat()}


@app.post("/predict", response_model=PredictionResponse, summary="Predict Diabetes Risk")
def predict_endpoint(patient: PatientFeatures):
    """
    Accept patient vitals and return:
      - probability of diabetes (0–1)
      - binary prediction (0/1)
      - human-readable label
      - risk category (Low/Medium/High)
    """
    try:
        features = patient.dict()
        result   = predict(features)

        # Persist to SQLite
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO predictions (timestamp, features, probability, prediction, label, risk_category) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                datetime.utcnow().isoformat(),
                json.dumps(features),
                result["probability"],
                result["prediction"],
                result["label"],
                result["risk_category"],
            ),
        )
        conn.commit()
        conn.close()

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplanationResponse, summary="Predict + LIME Explanation")
def explain_endpoint(patient: PatientFeatures):
    """
    Returns both the prediction AND a LIME feature contribution dict.
    Use this for the explainable AI dashboard.
    """
    try:
        features    = patient.dict()
        pred_result = predict(features)
        contribs    = explain_prediction(features)

        return {
            "prediction" : pred_result,
            "explanation": contribs,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history", summary="Prediction History")
def history_endpoint(limit: int = 20):
    """Return the last N predictions stored in SQLite."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT timestamp, probability, label, risk_category FROM predictions "
        "ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()

    return [
        {"timestamp": r[0], "probability": r[1], "label": r[2], "risk_category": r[3]}
        for r in rows
    ]
