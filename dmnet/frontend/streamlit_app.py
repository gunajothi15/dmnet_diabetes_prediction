"""
DMNet — Diabetes Risk Prediction Dashboard (Production Level)
=============================================================
Features:
  - Centered layout, dark high-contrast theme
  - Patient input form (centered, no sidebar)
  - Real-time prediction + risk gauge (HTML)
  - LIME explanation chart + table
  - SQLite-backed persistent prediction history
  - History dashboard: stats, risk distribution chart, export to CSV
  - Patient ID tracking, timestamps, notes
"""

import os
import sys
import sqlite3
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DMNet — Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
.stApp { background-color: #0f1117; color: #ffffff; }
[data-testid="collapsedControl"] { display: none; }
section[data-testid="stSidebar"]  { display: none; }

label, .stSlider label, .stSelectbox label {
    color: #ffffff !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}
.stSlider span { color: #cccccc !important; }
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #1e2130 !important;
    color: #ffffff !important;
    border-color: #4f8ef7 !important;
}
.stTextInput input {
    background: #1e2130 !important;
    color: #ffffff !important;
    border-color: #4f8ef7 !important;
}
.stTextArea textarea {
    background: #1e2130 !important;
    color: #ffffff !important;
    border-color: #4f8ef7 !important;
}
.stButton > button {
    width: 100%; padding: 12px;
    font-size: 0.95rem; font-weight: 700;
    border-radius: 8px; border: none;
}

/* Cards */
.res-card  { border-radius:12px; padding:24px; text-align:center; margin:12px 0; }
.res-low   { background:#0d2b1a; border:2px solid #22c55e; }
.res-med   { background:#2b1f0a; border:2px solid #f59e0b; }
.res-high  { background:#2b0d0d; border:2px solid #ef4444; }
.res-label { font-size:0.85rem; color:#cccccc; margin-bottom:4px; }
.res-value { font-size:2.2rem; font-weight:800; margin:0; }
.res-sub   { font-size:0.9rem; color:#dddddd; margin-top:6px; }
.c-low  { color:#22c55e; }
.c-med  { color:#f59e0b; }
.c-high { color:#ef4444; }

/* Stat tiles */
.stat-tile {
    background:#1e2130; border:1px solid #2e3250;
    border-radius:10px; padding:18px; text-align:center;
}
.stat-tile .st-num { font-size:2rem; font-weight:800; color:#4f8ef7; }
.stat-tile .st-lbl { font-size:0.8rem; color:#aaaaaa; margin-top:4px; }

/* Section title */
.sec { font-size:1.05rem; font-weight:700; color:#ffffff;
       border-left:4px solid #4f8ef7;
       padding-left:10px; margin:24px 0 12px 0; }

/* Badge */
.badge {
    display:inline-block; padding:3px 10px;
    border-radius:999px; font-size:0.78rem; font-weight:700;
}
.badge-low  { background:#14532d; color:#22c55e; }
.badge-med  { background:#451a03; color:#f59e0b; }
.badge-high { background:#450a0a; color:#ef4444; }

hr { border-color:#2e3250; }
.stDataFrame th { background:#1e2130 !important; color:#ffffff !important; }
.stDataFrame td { color:#ffffff !important; }
.streamlit-expanderHeader { color:#ffffff !important; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ── Database ───────────────────────────────────────────────────────────────────
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH      = os.path.join(project_root, "backend", "predictions.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id    TEXT    NOT NULL,
            patient_name  TEXT    DEFAULT '',
            timestamp     TEXT    NOT NULL,
            age           REAL, bmi REAL, glucose REAL, insulin REAL,
            blood_pressure REAL, pregnancies REAL, hba1c REAL,
            fasting_glucose REAL, physical_activity REAL,
            smoking_history REAL, family_history REAL,
            probability   REAL    NOT NULL,
            prediction    INTEGER NOT NULL,
            label         TEXT    NOT NULL,
            risk_category TEXT    NOT NULL,
            notes         TEXT    DEFAULT ''
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_prediction(pid, pname, features, result, notes=""):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO predictions
        (patient_id, patient_name, timestamp,
         age, bmi, glucose, insulin, blood_pressure, pregnancies,
         hba1c, fasting_glucose, physical_activity, smoking_history,
         family_history, probability, prediction, label, risk_category, notes)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        pid, pname, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        features["age"], features["bmi"], features["glucose"],
        features["insulin"], features["blood_pressure"], features["pregnancies"],
        features["hba1c"], features["fasting_glucose"],
        features["physical_activity"], features["smoking_history"],
        features["family_history"],
        result["probability"], result["prediction"],
        result["label"], result["risk_category"], notes
    ))
    conn.commit()
    conn.close()

def load_history(limit=200):
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
        conn, params=(limit,)
    )
    conn.close()
    return df

def delete_record(record_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM predictions WHERE id=?", (record_id,))
    conn.commit()
    conn.close()

def get_stats():
    conn  = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    diab  = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction=1").fetchone()[0]
    high  = conn.execute("SELECT COUNT(*) FROM predictions WHERE risk_category='High'").fetchone()[0]
    avg_p = conn.execute("SELECT AVG(probability) FROM predictions").fetchone()[0]
    conn.close()
    return total, diab, high, avg_p or 0.0


# ── Helpers ────────────────────────────────────────────────────────────────────
def risk_hex(r):  return {"Low":"#22c55e","Medium":"#f59e0b","High":"#ef4444"}.get(r,"#aaa")
def risk_cls(r):  return {"Low":"c-low","Medium":"c-med","High":"c-high"}.get(r,"")
def risk_card(r): return {"Low":"res-low","Medium":"res-med","High":"res-high"}.get(r,"")
def risk_emoji(r):return {"Low":"✅","Medium":"⚠️","High":"🚨"}.get(r,"❓")
def badge(r):     return f'<span class="badge badge-{r.lower()[:3]}">{r}</span>'

def gauge_html(prob, risk):
    pct   = int(prob * 100)
    color = risk_hex(risk)
    # Segmented bar: Low / Medium / High zones
    return f"""
    <div style="padding:16px 8px;">
        <div style="text-align:center;font-size:2.4rem;font-weight:800;color:{color};
                    line-height:1;">{pct}%</div>
        <div style="text-align:center;color:#aaaaaa;font-size:0.8rem;margin-bottom:8px;">
            Diabetes Probability</div>
        <div style="position:relative;background:#2e3250;border-radius:999px;
                    height:20px;overflow:hidden;">
            <div style="position:absolute;left:0;top:0;width:33%;height:100%;
                        background:#22c55e22;border-right:1px solid #2e3250;"></div>
            <div style="position:absolute;left:33%;top:0;width:34%;height:100%;
                        background:#f59e0b22;border-right:1px solid #2e3250;"></div>
            <div style="position:absolute;left:67%;top:0;width:33%;height:100%;
                        background:#ef444422;"></div>
            <div style="position:absolute;left:0;top:0;height:100%;
                        width:{pct}%;background:{color};border-radius:999px;
                        transition:width 0.5s;opacity:0.9;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;
                    font-size:0.72rem;color:#888;margin-top:4px;">
            <span>Low</span><span>Medium</span><span>High</span>
        </div>
        <div style="text-align:center;margin-top:10px;">
            <span style="background:{color}22;color:{color};font-weight:700;
                         padding:4px 16px;border-radius:999px;font-size:0.9rem;">
                {risk_emoji(risk)} {risk} Risk
            </span>
        </div>
    </div>
    """


def plot_lime_bar(contribs):
    items  = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)
    feats  = [i[0].split("<=")[0].split(">")[0].strip() for i in items]
    vals   = [i[1] for i in items]
    colors = ["#ef4444" if v > 0 else "#22c55e" for v in vals]
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#1e2130")
    ax.set_facecolor("#1e2130")
    ax.barh(feats[::-1], vals[::-1], color=colors[::-1], height=0.6)
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel("Contribution  (red = increases risk | green = decreases risk)",
                  fontsize=8, color="#cccccc")
    ax.set_title("LIME Feature Contributions", fontsize=11,
                 fontweight="bold", color="#ffffff", pad=10)
    ax.tick_params(colors="#cccccc", labelsize=8)
    for sp in ax.spines.values(): sp.set_color("#2e3250")
    ax.grid(axis="x", alpha=0.15, color="#aaaaaa")
    fig.tight_layout(pad=1.5)
    return fig


def plot_risk_distribution(df):
    counts = df["risk_category"].value_counts().reindex(
        ["Low","Medium","High"], fill_value=0)
    colors = ["#22c55e","#f59e0b","#ef4444"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    fig.patch.set_facecolor("#1e2130")

    # Bar chart
    ax = axes[0]
    ax.set_facecolor("#1e2130")
    ax.bar(counts.index, counts.values, color=colors, width=0.5)
    ax.set_title("Risk Distribution", color="#ffffff", fontweight="bold")
    ax.tick_params(colors="#cccccc")
    for sp in ax.spines.values(): sp.set_color("#2e3250")
    ax.set_ylabel("Patients", color="#cccccc")
    ax.grid(axis="y", alpha=0.15, color="#aaaaaa")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.3, str(v), ha="center", color="#ffffff", fontweight="bold")

    # Pie chart
    ax2 = axes[1]
    ax2.set_facecolor("#1e2130")
    non_zero = [(c, v, col) for c, v, col in
                zip(counts.index, counts.values, colors) if v > 0]
    if non_zero:
        labels, vals, cols = zip(*non_zero)
        wedges, texts, autotexts = ax2.pie(
            vals, labels=labels, colors=cols,
            autopct="%1.0f%%", startangle=90,
            textprops={"color":"#ffffff","fontsize":9},
        )
        for at in autotexts: at.set_color("#ffffff")
    ax2.set_title("Risk Share", color="#ffffff", fontweight="bold")
    fig.tight_layout(pad=1.5)
    return fig


def plot_probability_trend(df):
    if len(df) < 2:
        return None
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#1e2130")
    ax.set_facecolor("#1e2130")
    x   = range(len(df))
    ax.fill_between(x, df["probability"].values * 100,
                    alpha=0.25, color="#4f8ef7")
    ax.plot(x, df["probability"].values * 100,
            color="#4f8ef7", linewidth=2, marker="o", markersize=4)
    ax.axhline(50, color="#f59e0b", linewidth=0.8, linestyle="--", label="50% threshold")
    ax.set_title("Prediction Probability Over Time", color="#ffffff", fontweight="bold")
    ax.set_ylabel("Probability (%)", color="#cccccc")
    ax.set_xlabel("Record (newest → oldest)", color="#cccccc")
    ax.tick_params(colors="#cccccc")
    for sp in ax.spines.values(): sp.set_color("#2e3250")
    ax.grid(alpha=0.15, color="#aaaaaa")
    ax.legend(fontsize=8, labelcolor="#cccccc",
              facecolor="#1e2130", edgecolor="#2e3250")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    return fig


@st.cache_resource(show_spinner="Loading DMNet model...")
def load_inference():
    from utils.inference import predict, explain_prediction
    return predict, explain_prediction


# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [("result", None), ("contribs", None), ("tab", "predict")]:
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER + TABS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:24px 0 4px 0;">
    <span style="font-size:2rem;font-weight:800;color:#ffffff;">
        🩺 DMNet — Diabetes Risk Prediction System
    </span><br>
    <span style="color:#aaaaaa;font-size:0.9rem;">
        Hybrid CNN-LSTM · Longitudinal EHR Analysis · Explainable AI
    </span>
</div>
<hr>
""", unsafe_allow_html=True)

# Model check
model_path = os.path.join(project_root, "models", "dmnet_best.h5")
if not os.path.exists(model_path):
    st.error("Model not trained yet. Run `python train.py` first.")
    st.stop()

tab1, tab2 = st.tabs(["🔬  New Prediction", "📋  Patient History & Analytics"])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # Patient meta
    st.markdown('<div class="sec">Patient Information</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        patient_id   = st.text_input("Patient ID", value="P001",
                                      placeholder="e.g. P001")
    with m2:
        patient_name = st.text_input("Patient Name", value="",
                                      placeholder="Optional")
    with m3:
        notes        = st.text_input("Clinical Notes", value="",
                                      placeholder="Optional notes")

    # Clinical inputs
    st.markdown('<div class="sec">Clinical Measurements</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    # REPLACE WITH this:
    with c1:
        st.markdown("**Vitals**")
        age     = st.number_input("Age (years)",             min_value=20,   max_value=90,   value=50,  step=1)
        bmi     = st.number_input("BMI",                     min_value=15.0, max_value=50.0, value=27.0, step=0.1, format="%.1f")
        glucose = st.number_input("Glucose (mg/dL)",         min_value=60,   max_value=300,  value=140, step=1)
        insulin = st.number_input("Insulin (uU/mL)",         min_value=10,   max_value=400,  value=120, step=1)
    with c2:
        st.markdown("**Measurements**")
        bp      = st.number_input("Blood Pressure (mmHg)",   min_value=50,   max_value=140,  value=80,  step=1)
        preg    = st.number_input("Pregnancies",             min_value=0,    max_value=17,   value=2,   step=1)
        hba1c   = st.number_input("HbA1c (%)",              min_value=4.0,  max_value=14.0, value=6.5, step=0.1, format="%.1f")
        f_gluc  = st.number_input("Fasting Glucose (mg/dL)", min_value=60,   max_value=300,  value=110, step=1)
    with c3:
        st.markdown("**History**")
        phys   = st.selectbox("Physical Activity",
                              ["Active (1)", "Sedentary (0)"])
        smoke  = st.selectbox("Smoking History",
                              ["Never (0)", "Current / Past (1)"])
        family = st.selectbox("Family History of Diabetes",
                              ["No (0)", "Yes (1)"])

    phys_val   = 1 if "1" in phys   else 0
    smoke_val  = 1 if "1" in smoke  else 0
    family_val = 1 if "1" in family else 0

    patient_features = {
        "age": float(age), "bmi": float(bmi),
        "glucose": float(glucose), "insulin": float(insulin),
        "blood_pressure": float(bp), "pregnancies": float(preg),
        "hba1c": float(hba1c), "fasting_glucose": float(f_gluc),
        "physical_activity": float(phys_val),
        "smoking_history": float(smoke_val),
        "family_history": float(family_val),
    }

    st.markdown("")
    b1, b2, b3 = st.columns([2, 2, 1])
    with b1: predict_btn = st.button("🔬  Run Prediction", type="primary")
    with b2: explain_btn = st.button("💡  Run + Explain (LIME)")

    # ── Inference ──────────────────────────────────────────────────────────────
    if predict_btn or explain_btn:
        try:
            predict_fn, explain_fn = load_inference()
            with st.spinner("Running DMNet..."):
                result   = predict_fn(patient_features)
                contribs = explain_fn(patient_features) if explain_btn else None
            st.session_state.result   = result
            st.session_state.contribs = contribs
            save_prediction(patient_id, patient_name,
                            patient_features, result, notes)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    # ── Results ────────────────────────────────────────────────────────────────
    if st.session_state.result:
        result = st.session_state.result
        risk   = result["risk_category"]
        prob   = result["probability"]
        label  = result["label"]

        st.markdown('<hr><div class="sec">Prediction Result</div>',
                    unsafe_allow_html=True)
        ra, rb, rc = st.columns([1.2, 1.1, 1.4])

        with ra:
            st.markdown(f"""
            <div class="res-card {risk_card(risk)}">
                <div class="res-label">Diagnosis</div>
                <p class="res-value {risk_cls(risk)}">{risk_emoji(risk)} {label}</p>
                <div class="res-sub">Probability : <b>{prob*100:.1f}%</b></div>
                <div class="res-sub">Risk Level  : <b>{risk}</b></div>
                <div class="res-sub" style="margin-top:10px;font-size:0.8rem;color:#888;">
                    Patient: {patient_id}
                    {" · " + patient_name if patient_name else ""}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with rb:
            st.markdown(gauge_html(prob, risk), unsafe_allow_html=True)

        with rc:
            st.markdown('<div class="sec">Input Summary</div>',
                        unsafe_allow_html=True)
            summary_df = pd.DataFrame({
                "Feature": list(patient_features.keys()),
                "Value"  : [str(v) for v in patient_features.values()],
            })
            st.dataframe(summary_df, hide_index=True,
                         height=290, use_container_width=True)

        # Clinical recommendation
        if risk == "High":
            st.error("⚠️ **High Risk** — Immediate clinical review recommended. "
                     "Consider HbA1c confirmation, fasting glucose test, and specialist referral.")
        elif risk == "Medium":
            st.warning("🔶 **Medium Risk** — Lifestyle intervention advised. "
                       "Schedule follow-up in 3 months. Monitor glucose and BMI.")
        else:
            st.success("✅ **Low Risk** — Continue healthy lifestyle. "
                       "Routine annual screening recommended.")

        # ── LIME ──────────────────────────────────────────────────────────────
        if st.session_state.contribs:
            contribs = st.session_state.contribs
            st.markdown('<hr><div class="sec">Why did the model predict this? (LIME Explanation)</div>',
                        unsafe_allow_html=True)
            la, lb = st.columns([1.6, 1])
            with la:
                try:
                    lime_fig = plot_lime_bar(contribs)
                    st.pyplot(lime_fig)
                    plt.close("all")
                except Exception as e:
                    st.error(f"Chart error: {e}")
            with lb:
                st.markdown("**Feature Contributions**")
                rows = [{"Feature": k,
                         "Score"  : f"{v:+.4f}",
                         "Effect" : "Increases Risk" if v > 0 else "Decreases Risk"}
                        for k, v in sorted(contribs.items(),
                                           key=lambda x: -abs(x[1]))]
                st.dataframe(pd.DataFrame(rows), hide_index=True,
                             height=320, use_container_width=True)

            top3 = [k.split("<=")[0].strip()
                    for k, v in sorted(contribs.items(),
                                       key=lambda x: -x[1]) if v > 0][:3]
            if top3:
                st.info(f"**Top risk drivers:** {' · '.join(top3)}")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — HISTORY & ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    hist_df = load_history()

    if hist_df.empty:
        st.info("No predictions recorded yet. Run a prediction in the **New Prediction** tab.")
        st.stop()

    # ── Stats row ─────────────────────────────────────────────────────────────
    total, diab, high, avg_p = get_stats()
    non_diab  = total - diab
    diab_rate = (diab / total * 100) if total else 0

    st.markdown('<div class="sec">Overall Statistics</div>', unsafe_allow_html=True)
    s1, s2, s3, s4, s5 = st.columns(5)
    tiles = [
        (str(total),          "Total Patients"),
        (str(diab),           "Diabetic"),
        (str(non_diab),       "Non-Diabetic"),
        (f"{diab_rate:.1f}%", "Diabetic Rate"),
        (f"{avg_p*100:.1f}%", "Avg Probability"),
    ]
    for col, (num, lbl) in zip([s1,s2,s3,s4,s5], tiles):
        with col:
            st.markdown(f"""
            <div class="stat-tile">
                <div class="st-num">{num}</div>
                <div class="st-lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Analytics</div>', unsafe_allow_html=True)
    ch1, ch2 = st.columns([1, 1.6])

    with ch1:
        try:
            dist_fig = plot_risk_distribution(hist_df)
            st.pyplot(dist_fig)
            plt.close("all")
        except Exception as e:
            st.error(f"Chart error: {e}")

    with ch2:
        try:
            trend_fig = plot_probability_trend(hist_df)
            if trend_fig:
                st.pyplot(trend_fig)
                plt.close("all")
            else:
                st.info("Need at least 2 records for trend chart.")
        except Exception as e:
            st.error(f"Chart error: {e}")

    # ── Filter & Table ────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Patient Records</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        risk_filter = st.multiselect("Filter by Risk",
                                      ["Low","Medium","High"],
                                      default=["Low","Medium","High"])
    with f2:
        label_filter = st.multiselect("Filter by Label",
                                       ["Diabetic","Non-Diabetic"],
                                       default=["Diabetic","Non-Diabetic"])
    with f3:
        search = st.text_input("Search Patient ID / Name", "")

    filtered = hist_df[
        hist_df["risk_category"].isin(risk_filter) &
        hist_df["label"].isin(label_filter)
    ]
    if search:
        filtered = filtered[
            filtered["patient_id"].str.contains(search, case=False, na=False) |
            filtered["patient_name"].str.contains(search, case=False, na=False)
        ]

    # Display columns
    display_cols = ["id","patient_id","patient_name","timestamp",
                    "label","probability","risk_category",
                    "age","bmi","glucose","hba1c","notes"]
    display_cols = [c for c in display_cols if c in filtered.columns]
    show_df = filtered[display_cols].copy()
    show_df["probability"] = (show_df["probability"] * 100).round(1).astype(str) + "%"

    st.dataframe(show_df, hide_index=True, use_container_width=True, height=320)
    st.caption(f"Showing {len(filtered)} of {total} records")

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Export</div>', unsafe_allow_html=True)
    e1, e2, e3 = st.columns([1, 1, 2])

    with e1:
        csv_data = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label    = "⬇️  Export Filtered CSV",
            data     = csv_data,
            file_name= f"dmnet_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime     = "text/csv",
        )

    with e2:
        full_csv = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label    = "⬇️  Export All Records",
            data     = full_csv,
            file_name= f"dmnet_all_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime     = "text/csv",
        )

    with e3:
        del_id = st.number_input("Delete Record by ID", min_value=1,
                                  step=1, value=1)
        if st.button("🗑️  Delete Record", type="secondary"):
            delete_record(int(del_id))
            st.success(f"Record #{del_id} deleted.")
            st.rerun()

    # ── Recent 5 ──────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Recent Predictions</div>', unsafe_allow_html=True)
    recent = hist_df.head(5)[["patient_id","patient_name","timestamp",
                               "label","probability","risk_category"]].copy()
    recent["probability"] = (recent["probability"]*100).round(1).astype(str) + "%"
    st.dataframe(recent, hide_index=True, use_container_width=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<div style="text-align:center;color:#555;font-size:0.78rem;padding:8px 0 16px 0;">
    DMNet · Hybrid CNN-LSTM Diabetes Prediction ·
    Data stored locally in SQLite · For research use only
</div>
""", unsafe_allow_html=True)
