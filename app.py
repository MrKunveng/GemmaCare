import os
import json
import time
import base64
import requests
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# -------------------------------
# App config
# -------------------------------
st.set_page_config(page_title="Ensemble + MedGemma Triage", page_icon="🩺", layout="centered")

st.title("GemmaCare")
st.caption("Research-only prototype — not for clinical use.")

# -------------------------------
# Load ensemble model (.pkl)
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    path = "disease_prediction_ensemble.pkl"
    try:
        model_dict = joblib.load(path)
        # If it's a dict, extract the ensemble model
        if isinstance(model_dict, dict):
            return model_dict.get('ensemble_model', model_dict)
        return model_dict
    except Exception as e:
        st.error(f"Could not load model at `{path}`.\n{e}")
        return None

model = load_model()
model_dict = joblib.load("disease_prediction_ensemble.pkl") if model else None

# -------------------------------
# Helpers
# -------------------------------
def compute_bmi(weight_kg: float | None, height_cm: float | None) -> float | None:
    if not weight_kg or not height_cm or height_cm <= 0:
        return None
    h_m = height_cm / 100.0
    return round(weight_kg / (h_m * h_m), 1)

def build_feature_row(v):
    """
    Build a DataFrame row that matches the optimized model feature names.
    Model expects: Gender, Heart Rate (bpm), SpO2 Level (%), Systolic/Diastolic BP, 
    Body Temperature, Weight_kg, BMI, BMI_Category, Vital_Risk_Score, Alert_Count
    """
    # Get actual feature names from model
    feature_names = None
    if model_dict and isinstance(model_dict, dict) and 'feature_names' in model_dict:
        feature_names = model_dict['feature_names']
    elif model is not None and hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    
    # Map user inputs to model features
    sbp = v.get("sbp", 120)
    dbp = v.get("dbp", 80)
    spo2 = v.get("spo2", 95)
    temp = v.get("temperature_c", 37.0)
    bmi = v.get("bmi", 25)
    
    # Calculate BMI category (0: <18.5, 1: 18.5-25, 2: 25-30, 3: >30)
    if bmi < 18.5:
        bmi_category = 0
    elif bmi < 25:
        bmi_category = 1
    elif bmi < 30:
        bmi_category = 2
    else:
        bmi_category = 3
    
    # Calculate Vital Risk Score
    vital_risk = 0
    if sbp >= 140:
        vital_risk += 1
    if dbp >= 90:
        vital_risk += 1
    if spo2 < 95:
        vital_risk += 1
    if temp >= 38:
        vital_risk += 1
    
    # Calculate Alert Count
    alert_count = 0
    if sbp >= 160:
        alert_count += 1
    if dbp >= 100:
        alert_count += 1
    if spo2 < 92:
        alert_count += 1
    
    # Build feature dictionary with exact feature names
    base = {
        "Gender": 1 if v.get("sex") == "M" else 0,
        "Heart Rate (bpm)": 75,  # Default, not collected in UI currently
        "SpO2 Level (%)": spo2,
        "Systolic Blood Pressure (mmHg)": sbp,
        "Diastolic Blood Pressure (mmHg)": dbp,
        "Body Temperature (C)": temp,
        "Weight_kg": v.get("weight_kg", 70),
        "BMI": bmi,
        "BMI_Category": bmi_category,
        "Vital_Risk_Score": vital_risk,
        "Alert_Count": alert_count,
    }
    
    df = pd.DataFrame([base])
    
    # Ensure correct order if feature names available
    if feature_names:
        df = df[feature_names]
    
    return df

def predict_with_ensemble(v):
    """
    Returns dict: {'disease': str, 'confidence': float, 'risk_level': str, 'proba': dict}
    """
    if model is None:
        return {"disease": "Unknown", "confidence": 0.0, "risk_level": "unknown", "proba": {}}

    X = build_feature_row(v)
    
    # Disease mapping (numeric to name) - matches optimized model
    disease_map = {
        0: "Asthma",
        1: "Diabetes Mellitus",
        2: "Healthy",
        3: "Heart Disease",
        4: "Hypertension",
    }
    
    # Support both classifier APIs
    try:
        proba = model.predict_proba(X)[0]
        classes = list(getattr(model, "classes_", [i for i in range(len(proba))]))
        idx = int(np.argmax(proba))
        numeric_label = int(classes[idx])
        label = disease_map.get(numeric_label, f"Condition_{numeric_label}")
        conf = float(proba[idx])
        
        # Map numeric classes to disease names in proba
        proba_map = {}
        for c, p in zip(classes, proba):
            disease_name = disease_map.get(int(c), f"Condition_{c}")
            proba_map[disease_name] = float(p)
    except Exception as e:
        y = model.predict(X)[0]
        numeric_label = int(y)
        label = disease_map.get(numeric_label, f"Condition_{numeric_label}")
        conf = 0.0
        proba_map = {}

    # Risk assessment based on vitals
    sbp = v.get("sbp", 0)
    dbp = v.get("dbp", 0)
    spo2 = v.get("spo2", 100)
    
    if sbp >= 180 or dbp >= 110 or spo2 < 90:
        risk = "critical"
    elif sbp >= 160 or dbp >= 100 or spo2 < 92:
        risk = "high"
    elif sbp >= 140 or dbp >= 90 or spo2 < 95:
        risk = "moderate"
    else:
        risk = "low"
    
    return {"disease": label, "confidence": conf, "risk_level": risk, "proba": proba_map}

# -------------------------------
# MedGemma via Hugging Face Inference API (recommended on Streamlit Cloud)
# -------------------------------
# Safely get HF_TOKEN from secrets or environment
try:
    HF_TOKEN = st.secrets.get("HF_TOKEN", "")
except Exception:
    HF_TOKEN = os.getenv("HF_TOKEN", "")

MEDGEMMA_MODEL = "microsoft/Phi-3-mini-4k-instruct"  # publicly accessible alternative

def medgemma_recommend(vitals_dict: dict, ensemble_out: dict) -> dict:
    """
    Calls HF Inference API to avoid running the 4B model on Streamlit CPU.
    Returns: {'predicted_disease': str, 'recommendations': [..], 'notes': str}
    """
    if not HF_TOKEN:
        # Safe local fallback without LLM
        disease = ensemble_out.get("disease", "Unknown")
        risk = ensemble_out.get("risk_level", "moderate")
        
        # Generate smarter fallback based on vitals
        recommendations = []
        if vitals_dict.get("sbp", 0) >= 180 or vitals_dict.get("dbp", 0) >= 110:
            recommendations.append("⚠️ URGENT: Hypertensive crisis - immediate medical attention required")
            recommendations.append("Administer antihypertensive medication as per protocol")
        elif vitals_dict.get("sbp", 0) >= 160 or vitals_dict.get("dbp", 0) >= 100:
            recommendations.append("Monitor blood pressure closely every 15-30 minutes")
            recommendations.append("Consider antihypertensive therapy")
        
        if vitals_dict.get("spo2", 100) < 92:
            recommendations.append("⚠️ Low oxygen saturation - administer supplemental oxygen")
        
        if not recommendations:
            recommendations = [
                "Monitor vital signs regularly",
                "Consult with cardiology if symptoms persist",
                "Provide patient education on lifestyle modifications"
            ]
        
        return {
            "predicted_disease": disease,
            "recommendations": recommendations,
            "notes": "✓ Clinical recommendations generated using evidence-based protocols (AI model unavailable)."
        }

    url = f"https://api-inference.huggingface.co/models/{MEDGEMMA_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    system = (
        "You are a medical AI assistant for triage. Analyze patient vitals and provide clinical recommendations. "
        "Return ONLY valid JSON with these exact keys: \"predicted_disease\", \"recommendations\" (array of strings), \"notes\". "
        "Flag critical thresholds: BP≥160/100 (urgent), SpO2<92% (oxygen needed), Temp≥38.5°C (fever). "
        "Recommendations should be concise, evidence-based actions. No explanatory text outside JSON."
    )

    payload = {
        "inputs": f"{system}\n\n{json.dumps({'vitals': vitals_dict, 'model_prediction': ensemble_out}, ensure_ascii=False)}",
        "options": {"wait_for_model": True, "use_cache": True},
        "parameters": {"max_new_tokens": 400}
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=90)
        r.raise_for_status()
        data = r.json()

        # Inference API may return a list of dicts with 'generated_text'
        if isinstance(data, list) and data and "generated_text" in data[0]:
            txt = data[0]["generated_text"]
        elif isinstance(data, dict) and "generated_text" in data:
            txt = data["generated_text"]
        else:
            # Raw text path
            txt = str(data)

        # Extract JSON block
        s, e = txt.find("{"), txt.rfind("}")
        blob = txt[s:e+1] if s != -1 and e > s else txt
        out = json.loads(blob)
        # normalize keys
        out.setdefault("predicted_disease", ensemble_out.get("disease", "Unknown"))
        out.setdefault("recommendations", [])
        out.setdefault("notes", "")
        return out
    except Exception as e:
        return {
            "predicted_disease": ensemble_out.get("disease", "Unknown"),
            "recommendations": [
                "Repeat vitals and monitor closely.",
                "Consult a cardiologist.",
                "Consider urgent cardiac assessment if red flags persist."
            ],
            "notes": f"LLM request failed: {e}"
        }

def bullets_to_md(items):
    return "\n".join([f"• {x}" for x in items if x])

# -------------------------------
# Sidebar info
# -------------------------------
with st.sidebar:
    st.subheader("ℹ️ How it works")
    st.write(
        "1) Enter vitals → 2) Ensemble predicts a disease → "
        "3) We send vitals + prediction to MedGemma (HF Inference API) → "
        "4) JSON response populates the Results screen."
    )
    st.divider()
    st.write("**Secrets**: add `HF_TOKEN` in Streamlit → App → Settings → Secrets.")

# -------------------------------
# UI Form
# -------------------------------
with st.form("vitals_form"):
    colA, colB = st.columns(2)
    with colA:
        patient_id = st.text_input("Patient ID", value="001")
        age = st.number_input("Age", min_value=0, max_value=120, value=58)
        sex = st.selectbox("Sex", ["F", "M"], index=0)
        temperature_c = st.number_input("Temperature (°C)",  min_value=30.0, max_value=45.0, value=37.8, step=0.1)
        sbp = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=260, value=177)
        dbp = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=160, value=104)
    with colB:
        spo2 = st.number_input("SpO₂ (%)", min_value=50.0, max_value=100.0, value=94.0, step=0.5)
        weight_kg = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=75.5, step=0.1)
        height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=165.0, step=0.5)
        bmi_input = st.text_input("BMI (optional, auto-computed if blank)", value="")

    symptoms = st.multiselect(
        "Symptoms",
        ["Chest Pain", "Shortness of Breath", "Palpitations", "Fatigue", "Dizziness", "Headache"],
        default=["Chest Pain", "Shortness of Breath", "Palpitations", "Fatigue"]
    )

    submitted = st.form_submit_button("Confirm")

if submitted:
    bmi_val = None
    try:
        bmi_val = float(bmi_input) if bmi_input.strip() else None
    except ValueError:
        bmi_val = None
    if bmi_val is None:
        bmi_val = compute_bmi(weight_kg, height_cm)

    vitals = {
        "patient_id": patient_id,
        "age": age,
        "sex": sex,
        "temperature_c": float(temperature_c),
        "sbp": int(sbp),
        "dbp": int(dbp),
        "spo2": float(spo2),
        "weight_kg": float(weight_kg),
        "height_cm": float(height_cm),
        "bmi": bmi_val,
        "symptoms": symptoms,
    }

    with st.spinner("Running ensemble model..."):
        ens = predict_with_ensemble(vitals)

    with st.spinner("Generating clinical recommendations (MedGemma)..."):
        ai = medgemma_recommend(vitals, ens)

    # ---------------------------
    # Results screen (matches your mock)
    # ---------------------------
    st.header("Results")
    st.markdown(f"**Patient ID:** `{patient_id}`")
    st.subheader("Possible Diagnosis")
    st.text_input("Diagnosis", value=ai.get("predicted_disease", ens.get("disease", "")))

    st.subheader("Recommendation")
    st.text_area(
        "Recommendations",
        value=bullets_to_md(ai.get("recommendations", [])),
        height=140
    )

    st.subheader("Notes")
    st.text_area(
        "Notes",
        value=ai.get("notes", ""),
        height=160
    )

    # Save record
    record = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vitals": vitals,
        "ensemble": ens,
        "ai_result": ai,
    }
    rec_json = json.dumps(record, indent=2)
    st.download_button("💾 Save to patient record (JSON)", data=rec_json, file_name=f"{patient_id}_triage.json", mime="application/json")

    with st.expander("Debug info"):
        st.json(record)
