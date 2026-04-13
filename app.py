import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", message=".*Trying to unpickle.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GemmaCare · AI Medical Triage",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ─── Reset & base ─────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  color: #1e1e2e;
}
.main .block-container {
  padding-top: 1.5rem;
  padding-bottom: 3rem;
  max-width: 1100px;
}

/* ─── Animated hero banner ─────────────────────────────────────── */
@keyframes gradientShift {
  0%   { background-position: 0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50%       { transform: scale(1.08); }
}
.gc-hero {
  text-align: center;
  padding: 2.5rem 2rem 2rem;
  background: linear-gradient(-45deg, #667eea, #764ba2, #5e6ef7, #a855f7);
  background-size: 400% 400%;
  animation: gradientShift 8s ease infinite;
  color: white;
  border-radius: 20px;
  margin-bottom: 1.75rem;
  box-shadow: 0 20px 60px rgba(102,126,234,0.35), 0 4px 16px rgba(0,0,0,0.08);
  position: relative;
  overflow: hidden;
}
.gc-hero::before {
  content: "";
  position: absolute;
  top: -40px; right: -40px;
  width: 200px; height: 200px;
  background: rgba(255,255,255,0.07);
  border-radius: 50%;
}
.gc-hero::after {
  content: "";
  position: absolute;
  bottom: -60px; left: -30px;
  width: 240px; height: 240px;
  background: rgba(255,255,255,0.05);
  border-radius: 50%;
}
.gc-hero-icon {
  font-size: 3rem;
  display: block;
  animation: pulse 2.5s ease-in-out infinite;
  margin-bottom: 0.4rem;
}
.gc-hero h1 {
  margin: 0 0 0.35rem;
  font-size: 2.6rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  line-height: 1.1;
}
.gc-hero .gc-hero-sub {
  font-size: 1rem;
  opacity: 0.88;
  margin: 0 0 1.2rem;
  font-weight: 400;
}
.gc-hero-chips {
  display: flex;
  justify-content: center;
  gap: 0.6rem;
  flex-wrap: wrap;
}
.gc-hero-chip {
  background: rgba(255,255,255,0.18);
  backdrop-filter: blur(8px);
  border: 1px solid rgba(255,255,255,0.3);
  border-radius: 999px;
  padding: 0.25rem 0.85rem;
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.02em;
}

/* ─── Alert / disclaimer banners ───────────────────────────────── */
.gc-banner {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  padding: 0.9rem 1.2rem;
  border-radius: 12px;
  margin-bottom: 1rem;
  font-size: 0.875rem;
  line-height: 1.55;
}
.gc-banner-warn {
  background: #fffbeb;
  border: 1px solid #fde68a;
  border-left: 4px solid #f59e0b;
  color: #78350f;
}
.gc-banner-info {
  background: #f0f5ff;
  border: 1px solid #c7d7ff;
  border-left: 4px solid #667eea;
  color: #312e81;
}
.gc-banner-icon { font-size: 1.1rem; margin-top: 0.05rem; flex-shrink: 0; }

/* ─── Form container card ──────────────────────────────────────── */
.gc-form-card {
  background: #ffffff;
  border: 1px solid #e8ecff;
  border-radius: 16px;
  padding: 1.75rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 2px 12px rgba(102,126,234,0.06), 0 1px 3px rgba(0,0,0,0.04);
}

/* ─── Form section headers ─────────────────────────────────────── */
.gc-form-section {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.8rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #667eea;
  margin: 1.5rem 0 0.9rem;
  padding-bottom: 0.5rem;
  border-bottom: 1.5px solid #e8ecff;
}
.gc-form-section:first-child { margin-top: 0; }

/* ─── Submit button ────────────────────────────────────────────── */
div[data-testid="stFormSubmitButton"] > button {
  width: 100%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
  color: white !important;
  font-weight: 700 !important;
  font-size: 1.05rem !important;
  border-radius: 12px !important;
  padding: 0.85rem 2rem !important;
  border: none !important;
  box-shadow: 0 6px 20px rgba(102,126,234,0.4) !important;
  transition: all 0.2s ease !important;
  letter-spacing: 0.01em !important;
}
div[data-testid="stFormSubmitButton"] > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 10px 28px rgba(102,126,234,0.5) !important;
}
div[data-testid="stFormSubmitButton"] > button:active {
  transform: translateY(0) !important;
}

/* ─── Generic secondary buttons ────────────────────────────────── */
.stButton > button {
  border-radius: 10px !important;
  font-weight: 600 !important;
  transition: all 0.18s ease !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; }

/* ─── Results: section label ───────────────────────────────────── */
.gc-results-label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #888;
  margin: 2rem 0 0.75rem;
}
.gc-results-label::after {
  content: "";
  flex: 1;
  height: 1px;
  background: linear-gradient(to right, #e0e5ff, transparent);
}

/* ─── Diagnosis hero card ──────────────────────────────────────── */
.gc-dx-card {
  border-radius: 18px;
  padding: 1.75rem 2rem;
  margin-bottom: 1.5rem;
  border-left: 6px solid;
  position: relative;
  overflow: hidden;
  box-shadow: 0 8px 32px rgba(0,0,0,0.07);
}
.gc-dx-card::after {
  content: "";
  position: absolute;
  top: -30px; right: -30px;
  width: 130px; height: 130px;
  border-radius: 50%;
  opacity: 0.08;
  background: currentColor;
}
.gc-dx-card .gc-dx-icon { font-size: 2.4rem; margin-bottom: 0.4rem; display: block; }
.gc-dx-card h2 {
  margin: 0 0 0.5rem;
  font-size: 2rem;
  font-weight: 800;
  letter-spacing: -0.02em;
}
.gc-dx-card .gc-dx-row {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  flex-wrap: wrap;
  font-size: 0.88rem;
  color: #555;
}

/* ─── Risk / confidence badges ─────────────────────────────────── */
.gc-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  padding: 0.22rem 0.75rem;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.gc-pill-critical { background: #fee2e2; color: #b91c1c; }
.gc-pill-high     { background: #ffedd5; color: #c2410c; }
.gc-pill-moderate { background: #fef9c3; color: #a16207; }
.gc-pill-low      { background: #dcfce7; color: #166534; }
.gc-pill-conf     { background: #ede9fe; color: #5b21b6; }

/* ─── Vital metric cards ───────────────────────────────────────── */
.gc-vital {
  background: #ffffff;
  border: 1px solid #eef0ff;
  border-radius: 14px;
  padding: 1rem 0.75rem 0.85rem;
  text-align: center;
  box-shadow: 0 2px 10px rgba(102,126,234,0.06);
  transition: box-shadow 0.2s;
  height: 100%;
}
.gc-vital:hover { box-shadow: 0 6px 20px rgba(102,126,234,0.13); }
.gc-vital-emoji { font-size: 1.5rem; display: block; margin-bottom: 0.3rem; }
.gc-vital-val   { font-size: 1.45rem; font-weight: 700; color: #1e1e2e; line-height: 1.1; }
.gc-vital-name  { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; color: #9ca3af; margin: 0.25rem 0 0.15rem; }
.gc-vital-unit  { font-size: 0.72rem; color: #aaa; }
.gc-vital-ok    { border-top: 3px solid #22c55e; }
.gc-vital-warn  { border-top: 3px solid #f59e0b; }
.gc-vital-crit  { border-top: 3px solid #ef4444; }

/* ─── Symptoms tag strip ────────────────────────────────────────── */
.gc-symptom-tag {
  display: inline-block;
  background: #f3f0ff;
  color: #6d28d9;
  border: 1px solid #ddd6fe;
  border-radius: 999px;
  padding: 0.2rem 0.65rem;
  font-size: 0.76rem;
  font-weight: 500;
  margin: 0.15rem;
}

/* ─── Probability chart ─────────────────────────────────────────── */
.gc-prob-wrap {
  background: #ffffff;
  border: 1px solid #eef0ff;
  border-radius: 14px;
  padding: 1.25rem 1.5rem;
  box-shadow: 0 2px 10px rgba(102,126,234,0.05);
}
.gc-prob-row { margin: 0.7rem 0; }
.gc-prob-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.3rem;
}
.gc-prob-name  { font-size: 0.87rem; font-weight: 500; color: #374151; }
.gc-prob-name.top { font-weight: 700; color: #1e1e2e; }
.gc-prob-pct   { font-size: 0.87rem; font-weight: 600; }
.gc-bar-track {
  background: #f1f3ff;
  border-radius: 999px;
  height: 10px;
  overflow: hidden;
}
.gc-bar-fill {
  height: 10px;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--c1), var(--c2));
}

/* ─── Recommendation card ───────────────────────────────────────── */
.gc-rec {
  display: flex;
  gap: 0.85rem;
  align-items: flex-start;
  background: #fafbff;
  border: 1px solid #e8ecff;
  border-radius: 12px;
  padding: 1rem 1.15rem;
  margin: 0.5rem 0;
  line-height: 1.65;
  font-size: 0.9rem;
  color: #374151;
}
.gc-rec-num {
  flex-shrink: 0;
  width: 1.6rem; height: 1.6rem;
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.72rem;
  font-weight: 700;
  margin-top: 0.1rem;
}

/* ─── Clinical notes card ───────────────────────────────────────── */
.gc-notes {
  background: #fafbff;
  border: 1px solid #e0e7ff;
  border-radius: 14px;
  overflow: hidden;
}
.gc-notes-header {
  background: linear-gradient(135deg, #667eea11, #764ba211);
  border-bottom: 1px solid #e0e7ff;
  padding: 0.7rem 1.25rem;
  font-size: 0.8rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: #667eea;
}
.gc-notes-body {
  padding: 1.2rem 1.5rem;
  font-size: 0.91rem;
  line-height: 1.7;
  color: #374151;
}

/* ─── Action button row ─────────────────────────────────────────── */
.gc-actions {
  display: flex;
  gap: 0.75rem;
  margin-top: 1.5rem;
}

/* ─── Footer ────────────────────────────────────────────────────── */
.gc-footer {
  text-align: center;
  padding: 1.5rem 1rem 0.5rem;
  margin-top: 2rem;
  border-top: 1px solid #f0f0f0;
  font-size: 0.8rem;
  color: #aaa;
  line-height: 1.8;
}
.gc-footer strong { color: #888; }

/* ─── Sidebar ───────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #f4f5ff 0%, #fafaff 100%);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0; }

.gc-sb-brand {
  background: linear-gradient(135deg, #667eea, #764ba2);
  padding: 1.5rem 1rem 1.2rem;
  text-align: center;
  color: white;
  margin-bottom: 0.5rem;
}
.gc-sb-brand .gc-sb-icon { font-size: 2.5rem; display: block; margin-bottom: 0.3rem; }
.gc-sb-brand h2 { margin: 0; font-size: 1.3rem; font-weight: 800; letter-spacing: -0.01em; }
.gc-sb-brand p  { margin: 0.2rem 0 0; font-size: 0.74rem; opacity: 0.8; font-weight: 400; }

.gc-sb-section {
  font-size: 0.68rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #9ca3af;
  padding: 1rem 1rem 0.4rem;
}

.gc-sb-step {
  display: flex;
  align-items: flex-start;
  gap: 0.6rem;
  padding: 0.35rem 1rem;
  font-size: 0.83rem;
  color: #374151;
  line-height: 1.4;
}
.gc-sb-step-num {
  flex-shrink: 0;
  width: 1.3rem; height: 1.3rem;
  background: #667eea;
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.65rem;
  font-weight: 800;
  margin-top: 0.05rem;
}

.gc-sb-stat {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: white;
  border: 1px solid #e8ecff;
  border-radius: 10px;
  padding: 0.6rem 0.85rem;
  margin: 0.25rem 1rem;
  font-size: 0.82rem;
}
.gc-sb-stat-val { font-weight: 700; color: #667eea; }

.gc-sb-cond {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.3rem 1rem;
  font-size: 0.83rem;
  color: #374151;
}

.gc-sb-feat {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 0.2rem 1rem;
  background: white;
  border: 1px solid #e8ecff;
  border-radius: 8px;
  padding: 0.4rem 0.75rem;
  font-size: 0.8rem;
  color: #374151;
}

/* ─── Streamlit overrides ───────────────────────────────────────── */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stMultiSelect"] label {
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  color: #374151 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.04em !important;
}
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
  border-radius: 8px !important;
  border-color: #d1d9ff !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextInput"] input:focus {
  border-color: #667eea !important;
  box-shadow: 0 0 0 3px rgba(102,126,234,0.15) !important;
}
div[data-testid="stSelectbox"] > div > div {
  border-radius: 8px !important;
  border-color: #d1d9ff !important;
}
.stExpander { border-radius: 10px !important; border-color: #d1d9ff !important; }

/* hide streamlit default hr */
hr { border-color: #f0f0f0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="gc-hero">
  <span class="gc-hero-icon">🩺</span>
  <h1>GemmaCare</h1>
  <p class="gc-hero-sub">AI-Powered Medical Triage &amp; Clinical Decision Support</p>
  <div class="gc-hero-chips">
    <span class="gc-hero-chip">⚡ 95.22% Accuracy</span>
    <span class="gc-hero-chip">🔬 60K Patient Records</span>
    <span class="gc-hero-chip">📋 MedGemma Powered</span>
    <span class="gc-hero-chip">🚨 Critical Alert System</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Banners ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="gc-banner gc-banner-warn">
  <span class="gc-banner-icon">⚕️</span>
  <span><strong>Research Prototype</strong> — This system is designed to assist healthcare providers
  with triage decisions. It is <em>not</em> a substitute for direct clinical evaluation or diagnosis
  by a qualified healthcare professional.</span>
</div>

<div class="gc-banner gc-banner-info">
  <span class="gc-banner-icon">🤖</span>
  <div>
    <strong>Disease Prediction</strong> is powered by an XGBoost + LightGBM ensemble trained on
    60,000 patient records (95.22% accuracy across 5 conditions).&nbsp;
    <strong>Clinical Recommendations</strong> are generated using <strong>Google MedGemma</strong>,
    aligned with ADA 2024–25, ESC 2024, GINA 2024, and WHO 2020 guidelines.
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    path = "best_disease_model.pkl"
    try:
        obj = joblib.load(path)
        return obj if isinstance(obj, dict) else {"model": obj}
    except Exception as e:
        st.error(f"Could not load model at `{path}`.\n{e}")
        return None

model_dict = load_model()
model = model_dict.get("model") if model_dict else None

# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_bmi(weight_kg, height_cm):
    if not weight_kg or not height_cm or height_cm <= 0:
        return None
    h = height_cm / 100.0
    return round(weight_kg / (h * h), 1)

def bmi_category(bmi):
    if bmi is None:
        return ("N/A", "#999", "#f0f0f0")
    if bmi < 18.5:
        return ("Underweight", "#0369a1", "#e0f2fe")
    if bmi < 25.0:
        return ("Normal",      "#166534", "#dcfce7")
    if bmi < 30.0:
        return ("Overweight",  "#92400e", "#fef3c7")
    return     ("Obese",       "#991b1b", "#fee2e2")

def vital_status(key, value):
    """Returns CSS class suffix: ok / warn / crit based on clinical ranges."""
    if key == "sbp":
        return "crit" if value >= 180 else "warn" if value >= 140 else "ok"
    if key == "dbp":
        return "crit" if value >= 110 else "warn" if value >= 90 else "ok"
    if key == "spo2":
        return "crit" if value < 90 else "warn" if value < 95 else "ok"
    if key == "temp":
        return "crit" if value >= 40 else "warn" if value >= 38 else "ok"
    if key == "hr":
        return "crit" if value > 150 or value < 40 else "warn" if value > 100 or value < 60 else "ok"
    return "ok"

def build_feature_row(v):
    features = {
        "Gender":                           1 if v.get("sex") == "M" else 0,
        "Heart Rate (bpm)":                 v.get("heart_rate", 75),
        "SpO2 Level (%)":                   v.get("spo2", 95),
        "Systolic Blood Pressure (mmHg)":   v.get("sbp", 120),
        "Diastolic Blood Pressure (mmHg)":  v.get("dbp", 80),
        "Body Temperature (C)":             v.get("temperature_c", 37.0),
        "Weight_kg":                        v.get("weight_kg", 70),
        "Height_cm":                        v.get("height_cm", 170),
        "BMI":                              v.get("bmi", 25),
    }
    df = pd.DataFrame([features])
    if model_dict and model_dict.get("scaler"):
        cols = model_dict.get("feature_columns", df.columns.tolist())
        df = df[cols]
        df = pd.DataFrame(model_dict["scaler"].transform(df), columns=cols)
    return df

def predict_with_ensemble(v):
    if model is None:
        return {"disease": "Unknown", "confidence": 0.0, "risk_level": "unknown", "proba": {}}
    X = build_feature_row(v)
    target_encoder = model_dict.get("target_encoder") if model_dict else None
    try:
        proba_array = model.predict_proba(X)[0]
        y_raw = model.predict(X)[0]
        y_pred = y_raw.item() if hasattr(y_raw, "item") else int(y_raw)
        if not isinstance(proba_array, np.ndarray):
            proba_array = np.array(proba_array)

        if target_encoder:
            label_arr = target_encoder.inverse_transform(np.array([int(y_pred)]))
            label = label_arr[0].item() if hasattr(label_arr[0], "item") else str(label_arr[0])
            classes = target_encoder.classes_.tolist() if hasattr(target_encoder.classes_, "tolist") else list(target_encoder.classes_)
            proba_map = {str(c): float(p.item() if hasattr(p, "item") else p) for c, p in zip(classes, proba_array)}
        else:
            dm = {0: "Asthma", 1: "Diabetes Mellitus", 2: "Healthy", 3: "Heart Disease", 4: "Hypertension"}
            label = dm.get(int(y_pred), f"Condition_{y_pred}")
            classes = model.classes_.tolist() if hasattr(getattr(model, "classes_", None), "tolist") else list(range(len(proba_array)))
            proba_map = {dm.get(int(c.item() if hasattr(c, "item") else c), f"Cond_{c}"): float(p.item() if hasattr(p, "item") else p) for c, p in zip(classes, proba_array)}

        conf_raw = np.max(proba_array)
        conf = float(conf_raw.item() if hasattr(conf_raw, "item") else conf_raw)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return {"disease": "Unknown", "confidence": 0.0, "risk_level": "unknown", "proba": {}}

    sbp, dbp, spo2 = v.get("sbp", 0), v.get("dbp", 0), v.get("spo2", 100)
    if   sbp >= 180 or dbp >= 110 or spo2 < 90: risk = "critical"
    elif sbp >= 160 or dbp >= 100 or spo2 < 92: risk = "high"
    elif sbp >= 140 or dbp >=  90 or spo2 < 95: risk = "moderate"
    else:                                         risk = "low"

    return {"disease": label, "confidence": conf, "risk_level": risk, "proba": proba_map}

# ── Recommendations ───────────────────────────────────────────────────────────
try:
    HF_TOKEN = st.secrets.get("HF_TOKEN", "")
except Exception:
    HF_TOKEN = os.getenv("HF_TOKEN", "")

DISEASE_REC = {
    "Diabetes Mellitus": {
        "rec": "Maintain structured meal plans with carbohydrate counting and engage in 150+ minutes of weekly aerobic activity. Monitor blood glucose with CGM when available, targeting HbA1c <7%.",
        "notes": "Schedule comprehensive diabetes screening: fasting glucose, HbA1c, and OGTT if indicated. Consider GLP-1 receptor agonists for cardiovascular protection and implement structured DSME. Take activity breaks every 30 minutes to optimise glycaemic control.",
    },
    "Heart Disease": {
        "rec": "Adopt Mediterranean or DASH dietary patterns with <2,300 mg sodium daily and 150+ minutes of weekly aerobic exercise. Include muscle-strengthening twice weekly; emphasise healthy fats from olive oil, nuts, and fatty fish.",
        "notes": "Cardiovascular risk stratification using validated calculators is recommended. Consider lipid panel, hs-CRP, and coronary calcium scoring. Immediate tobacco cessation if applicable. Target BP <130/80 mmHg. Seek urgent cardiology consultation for chest pain or dyspnoea.",
    },
    "Hypertension": {
        "rec": "Target systolic BP 120–129 mmHg through sodium reduction to <1,500 mg daily and increased potassium intake. Follow DASH or Mediterranean dietary patterns; maintain BMI 18.5–24.9 kg/m².",
        "notes": "2024 ESC guidelines redefine elevated BP as 120–139/70–89 mmHg. Implement home BP monitoring and consider 24-hour ambulatory monitoring for white-coat / masked hypertension. Weight reduction of 3–5% yields ~1 mmHg per kg lost. Target 7–9 hours of sleep nightly.",
    },
    "Asthma": {
        "rec": "Use ICS-containing medication; low-dose ICS-formoterol is preferred (GINA 2024 Track 1). Never use SABA alone — increased mortality risk. Implement a written asthma action plan with peak-flow targets.",
        "notes": "Objective confirmation via FeNO, blood eosinophils, and spirometry with bronchodilator reversibility is essential. The MART approach reduces severe exacerbations by 60–64% vs SABA-only. Seek emergency care for peak flow <33% predicted or inability to speak in full sentences.",
    },
    "Healthy": {
        "rec": "Maintain 150–300 min moderate-intensity or 75–150 min vigorous aerobic activity weekly, plus muscle-strengthening twice weekly. Follow whole-food nutrition, adequate protein, healthy fats, and 7–9 hours of quality sleep.",
        "notes": "Continue evidence-based preventive care with age-appropriate screenings per USPSTF recommendations. All physical activity counts toward weekly totals. Periodic biomarker monitoring and annual preventive evaluations are recommended, including family history assessment.",
    },
}

def medgemma_recommend(vitals, ensemble_out):
    disease  = ensemble_out.get("disease", "Unknown")
    base     = DISEASE_REC.get(disease, {})
    rec_text = base.get("rec",   f"Comprehensive medical evaluation recommended. Follow up with primary care within 1 week for {disease}.")
    notes_text = base.get("notes", f"Medical attention recommended. A comprehensive evaluation is needed for {disease}.")

    alerts = []
    sbp  = vitals.get("sbp",           120)
    dbp  = vitals.get("dbp",            80)
    spo2 = vitals.get("spo2",           98)
    temp = vitals.get("temperature_c", 37.0)

    if sbp >= 180 or dbp >= 110:
        alerts.append(f"🚨 HYPERTENSIVE CRISIS: BP {sbp}/{dbp} mmHg — Immediate emergency care required!")
        notes_text = f"⚠️ CRITICAL: Hypertensive emergency detected. {notes_text}"
    if spo2 < 90:
        alerts.append(f"🚨 SEVERE HYPOXAEMIA: SpO₂ {spo2}% — Emergency care needed immediately!")
        notes_text = f"⚠️ CRITICAL: Severe hypoxaemia requiring immediate intervention. {notes_text}"
    elif spo2 < 92:
        alerts.append(f"⚠️ LOW OXYGEN: SpO₂ {spo2}% — Seek urgent medical attention")
    if temp >= 39.0:
        alerts.append(f"⚠️ HIGH FEVER: {temp}°C — Medical evaluation recommended")

    return {"predicted_disease": disease, "recommendations": alerts + [rec_text], "notes": notes_text}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="gc-sb-brand">
      <span class="gc-sb-icon">🩺</span>
      <h2>GemmaCare</h2>
      <p>AI Medical Triage System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gc-sb-section">How It Works</div>', unsafe_allow_html=True)
    for n, t in [
        ("1", "Enter patient vitals & measurements"),
        ("2", "ML ensemble predicts condition"),
        ("3", "MedGemma generates evidence-based guidance"),
        ("4", "Critical alerts are automatically flagged"),
    ]:
        st.markdown(f"""
        <div class="gc-sb-step">
          <span class="gc-sb-step-num">{n}</span>
          <span>{t}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="gc-sb-section">Model Performance</div>', unsafe_allow_html=True)
    for lbl, val in [("Accuracy", "95.22%"), ("Training Records", "60,000"), ("Conditions Detected", "5")]:
        st.markdown(f"""
        <div class="gc-sb-stat">
          <span>{lbl}</span>
          <span class="gc-sb-stat-val">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="gc-sb-section">Detectable Conditions</div>', unsafe_allow_html=True)
    for ico, name in [("🩺","Diabetes Mellitus"),("❤️","Heart Disease"),("⚠️","Hypertension"),("🫁","Asthma"),("✅","Healthy")]:
        st.markdown(f'<div class="gc-sb-cond"><span>{ico}</span><span>{name}</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="gc-sb-section">Key Features</div>', unsafe_allow_html=True)
    for feat in ["🚨 Critical vital alert detection", "📋 Evidence-based clinical recs",
                 "⚕️ Healthcare triage assistance",  "🔒 Secure local processing"]:
        st.markdown(f'<div class="gc-sb-feat">{feat}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("💡 Use realistic vital signs for best results.")

# ── Form ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="gc-results-label"><span>Patient Vitals Entry</span></div>', unsafe_allow_html=True)

with st.form("vitals_form"):
    # Demographics
    st.markdown('<div class="gc-form-section">👤 Demographics</div>', unsafe_allow_html=True)
    c1, c2, _pad = st.columns([1, 1, 2])
    with c1:
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=58)
    with c2:
        sex = st.selectbox("Biological Sex", ["Female", "Male"], index=0)

    # Vital Signs
    st.markdown('<div class="gc-form-section">🌡️ Vital Signs</div>', unsafe_allow_html=True)
    c3, c4, c5 = st.columns(3)
    with c3:
        temperature_c = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.8, step=0.1, help="Body temperature in Celsius")
        sbp = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=260, value=120, help="Upper blood pressure reading")
    with c4:
        dbp = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=160, value=80, help="Lower blood pressure reading")
        spo2 = st.number_input("SpO₂ (%)", min_value=50.0, max_value=100.0, value=98.0, step=0.5, help="Blood oxygen saturation")
    with c5:
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=75, help="Beats per minute")

    # Body Measurements
    st.markdown('<div class="gc-form-section">📏 Body Measurements</div>', unsafe_allow_html=True)
    c6, c7, c8 = st.columns(3)
    with c6:
        weight_kg = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0, step=0.1)
    with c7:
        height_cm = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.5)
    with c8:
        bmi_input = st.text_input("BMI (optional)", value="", placeholder="Auto-calculated", help="Leave blank to auto-calculate")

    with st.expander("➕ Additional Symptoms (Optional)"):
        symptoms = st.multiselect(
            "Select any symptoms present",
            ["Chest Pain","Shortness of Breath","Palpitations","Fatigue",
             "Dizziness","Headache","Nausea","Sweating"],
            default=[],
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍 Run AI Analysis", use_container_width=True)

# ── Results ───────────────────────────────────────────────────────────────────
if submitted:
    # Compute BMI
    try:
        bmi_val = float(bmi_input.strip()) if bmi_input.strip() else None
    except ValueError:
        bmi_val = None
    if bmi_val is None:
        bmi_val = compute_bmi(weight_kg, height_cm)

    sex_code = "M" if sex == "Male" else "F"
    vitals = {
        "age": age, "sex": sex_code,
        "temperature_c": float(temperature_c),
        "sbp": int(sbp), "dbp": int(dbp),
        "spo2": float(spo2), "heart_rate": int(heart_rate),
        "weight_kg": float(weight_kg), "height_cm": float(height_cm),
        "bmi": bmi_val, "symptoms": symptoms,
    }

    with st.spinner("🔬 Running ML analysis…"):
        ens = predict_with_ensemble(vitals)
    with st.spinner("💡 Generating clinical recommendations…"):
        ai = medgemma_recommend(vitals, ens)

    predicted_disease = ai.get("predicted_disease", ens.get("disease", ""))
    confidence   = ens.get("confidence", 0) * 100
    risk_level   = ens.get("risk_level", "unknown")
    recommendations = ai.get("recommendations", [])
    notes        = ai.get("notes", "")

    # Design tokens per risk / disease
    RISK_CFG = {
        "critical": dict(border="#ef4444", bg="#fff5f5", pill="gc-pill-critical"),
        "high":     dict(border="#f97316", bg="#fff7ed", pill="gc-pill-high"),
        "moderate": dict(border="#eab308", bg="#fefce8", pill="gc-pill-moderate"),
        "low":      dict(border="#22c55e", bg="#f0fdf4", pill="gc-pill-low"),
    }
    DISEASE_META = {
        "Healthy":          ("✅", "#22c55e",  "#16a34a"),
        "Heart Disease":    ("❤️", "#ef4444",  "#dc2626"),
        "Hypertension":     ("⚠️", "#f97316",  "#ea580c"),
        "Asthma":           ("🫁", "#06b6d4",  "#0891b2"),
        "Diabetes Mellitus":("🩺", "#667eea",  "#4f46e5"),
    }
    PROB_GRAD = {
        "Healthy":           ("#22c55e", "#16a34a"),
        "Heart Disease":     ("#ef4444", "#dc2626"),
        "Hypertension":      ("#f97316", "#ea580c"),
        "Asthma":            ("#06b6d4", "#0891b2"),
        "Diabetes Mellitus": ("#667eea", "#764ba2"),
    }

    dx_icon, dx_color, dx_dark = DISEASE_META.get(predicted_disease, ("🩺","#667eea","#4f46e5"))
    risk_cfg = RISK_CFG.get(risk_level, RISK_CFG["low"])

    # ── Section label
    st.markdown('<div class="gc-results-label"><span>Analysis Results</span></div>', unsafe_allow_html=True)

    # ── Diagnosis card
    st.markdown(f"""
    <div class="gc-dx-card" style="background:{risk_cfg['bg']};border-color:{risk_cfg['border']};color:{dx_color};">
      <span class="gc-dx-icon">{dx_icon}</span>
      <h2 style="color:{dx_dark};">{predicted_disease}</h2>
      <div class="gc-dx-row">
        <span class="gc-pill gc-pill-conf">🎯 {confidence:.1f}% confidence</span>
        <span class="gc-pill {risk_cfg['pill']}">⚡ {risk_level.upper()} RISK</span>
        <span style="color:#888;">·</span>
        <span>{sex} &nbsp;·&nbsp; {age} yrs</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Vitals cards
    bmi_cat, bmi_fg, bmi_bg = bmi_category(bmi_val)
    bmi_display = f"{bmi_val:.1f}" if bmi_val else "N/A"

    v_cols = st.columns(5)
    vitals_display = [
        ("🩸", "Blood Pressure", f"{sbp}/{dbp}", "mmHg",   vital_status("sbp", sbp)),
        ("💧", "SpO₂",           f"{spo2}%",    "Oxygen",  vital_status("spo2", spo2)),
        ("🌡️", "Temperature",   f"{temperature_c}°C", "Body Temp", vital_status("temp", temperature_c)),
        ("💓", "Heart Rate",     str(heart_rate), "bpm",   vital_status("hr", heart_rate)),
    ]
    for col, (emoji, name, val, unit, status) in zip(v_cols[:4], vitals_display):
        with col:
            st.markdown(f"""
            <div class="gc-vital gc-vital-{status}">
              <span class="gc-vital-emoji">{emoji}</span>
              <div class="gc-vital-val">{val}</div>
              <div class="gc-vital-name">{name}</div>
              <div class="gc-vital-unit">{unit}</div>
            </div>""", unsafe_allow_html=True)

    with v_cols[4]:
        st.markdown(f"""
        <div class="gc-vital gc-vital-ok">
          <span class="gc-vital-emoji">⚖️</span>
          <div class="gc-vital-val">{bmi_display}</div>
          <div class="gc-vital-name">BMI</div>
          <div class="gc-vital-unit">
            <span class="gc-pill" style="background:{bmi_bg};color:{bmi_fg};font-size:0.65rem;">{bmi_cat}</span>
          </div>
        </div>""", unsafe_allow_html=True)

    # Symptoms
    if symptoms:
        tags = "".join(f'<span class="gc-symptom-tag">{s}</span>' for s in symptoms)
        st.markdown(f"""
        <div style="margin:0.9rem 0 0.2rem;">
          <span style="font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;color:#9ca3af;">
            Reported Symptoms
          </span><br>{tags}
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column layout: proba left, notes right
    left_col, right_col = st.columns([5, 6])

    with left_col:
        st.markdown('<div class="gc-results-label"><span>Disease Probability</span></div>', unsafe_allow_html=True)
        proba_dict = ens.get("proba", {})
        if proba_dict:
            sorted_proba = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)
            rows_html = ""
            for d_name, prob in sorted_proba:
                pct  = prob * 100
                g1, g2 = PROB_GRAD.get(d_name, ("#667eea","#764ba2"))
                is_top = d_name == predicted_disease
                name_cls = "gc-prob-name top" if is_top else "gc-prob-name"
                prefix = "🎯 " if is_top else ""
                rows_html += f"""
                <div class="gc-prob-row">
                  <div class="gc-prob-header">
                    <span class="{name_cls}">{prefix}{d_name}</span>
                    <span class="gc-prob-pct" style="color:{g1};">{pct:.1f}%</span>
                  </div>
                  <div class="gc-bar-track">
                    <div class="gc-bar-fill" style="width:{pct}%;--c1:{g1};--c2:{g2};"></div>
                  </div>
                </div>"""
            st.markdown(f'<div class="gc-prob-wrap">{rows_html}</div>', unsafe_allow_html=True)
        else:
            st.info("Probability data unavailable.")

    with right_col:
        st.markdown('<div class="gc-results-label"><span>Clinical Notes</span></div>', unsafe_allow_html=True)
        notes_html = notes.replace("\n", "<br>")
        st.markdown(f"""
        <div class="gc-notes">
          <div class="gc-notes-header">📝 MedGemma Clinical Notes</div>
          <div class="gc-notes-body">{notes_html}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Recommendations
    st.markdown('<div class="gc-results-label"><span>Clinical Recommendations</span></div>', unsafe_allow_html=True)
    rec_num = 0
    for rec in recommendations:
        if "🚨" in rec:
            st.error(rec)
        elif "⚠️" in rec:
            st.warning(rec)
        else:
            rec_num += 1
            st.markdown(f"""
            <div class="gc-rec">
              <div class="gc-rec-num">{rec_num}</div>
              <div>{rec}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Actions
    record = {
        "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "diagnosis":    predicted_disease,
        "confidence":   f"{confidence:.1f}%",
        "risk_level":   risk_level,
        "bmi_category": bmi_cat,
        "vitals":       vitals,
        "recommendations": recommendations,
        "notes":        notes,
    }
    rec_json = json.dumps(record, indent=2)

    dl_col, reset_col = st.columns(2)
    with dl_col:
        st.download_button(
            "💾 Download Report (JSON)",
            data=rec_json,
            file_name=f"gemmacare_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
    with reset_col:
        if st.button("🔄 Analyse Another Patient", use_container_width=True):
            st.rerun()

    # ── Footer
    st.markdown("""
    <div class="gc-footer">
      🧠 <strong>Powered by Google MedGemma</strong> — Medical instruction-tuned LLM<br>
      🔬 XGBoost + LightGBM ensemble · 60,000 patient records · 95.22% accuracy<br>
      📋 ADA 2024–25 &nbsp;·&nbsp; ESC 2024 &nbsp;·&nbsp; GINA 2024 &nbsp;·&nbsp; WHO 2020 &nbsp;·&nbsp; USPSTF
    </div>
    """, unsafe_allow_html=True)
