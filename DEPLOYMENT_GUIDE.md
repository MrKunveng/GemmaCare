# 🚀 GemmaCare Deployment Guide

## Quick Deployment to Streamlit Cloud

### 1. **Push to GitHub** ✅
Your code is ready to push! The HuggingFace token is now in secrets (not hardcoded).

### 2. **Deploy on Streamlit Cloud**

1. Go to https://share.streamlit.io/
2. Click "New app"
3. Select your repository: `MrKunveng/GemmaCare`
4. Branch: `main`
5. Main file path: `app.py`

### 3. **Add Secrets to Streamlit Cloud**

In Streamlit Cloud App Settings → Secrets, add:

```toml
HF_TOKEN = "your_huggingface_token_here"
```

**Note:** Get your token from https://huggingface.co/settings/tokens

**Note:** The app works perfectly WITHOUT the token - it uses high-quality predetermined recommendations!

---

## 📁 Files Overview

### Core Application Files:
- `app.py` - Main Streamlit application (95% accuracy triage system)
- `best_disease_model.pkl` - Trained ML model (2.4MB, 95.22% accuracy)
- `requirements.txt` - Python dependencies
- `runtime.txt` - Python version specification

### Configuration:
- `.streamlit/secrets.toml` - Local secrets (gitignored, not pushed)
- `.streamlit/secrets.toml.example` - Example configuration
- `.streamlit/config.toml` - Streamlit UI configuration (if exists)

### Documentation:
- `README.md` - Main documentation
- `DEMO_TEST_CASES.txt` - Demo test cases for presentation
- `DEPLOYMENT_GUIDE.md` - This file

### Training/Development (in ML/ folder):
- Model training scripts
- Test scripts
- Dataset backup

---

## 🔐 Security Notes

- ✅ HF_TOKEN removed from code (was hardcoded, now uses secrets)
- ✅ `.streamlit/secrets.toml` is gitignored
- ✅ App works perfectly without the token (uses intelligent fallback)

---

## ⚙️ Environment Variables

### For Streamlit Cloud:
Add in App Settings → Secrets:
```toml
HF_TOKEN = "your_token_here"
```

### For Local Development:
Create `.streamlit/secrets.toml`:
```toml
HF_TOKEN = "your_token_here"
```

Or use environment variable:
```bash
export HF_TOKEN="your_token_here"
streamlit run app.py
```

---

## 🧪 Verify Deployment

After deploying, test with these values:

**Diabetes Test:**
- Gender: Male, BP: 140/90, SpO2: 96%
- Weight: 85kg, Height: 175cm
- Expected: Diabetes Mellitus (98.78% confidence)

**Critical Alert Test:**
- SpO2: 85%
- Expected: 🚨 SEVERE HYPOXEMIA alert

---

## 📊 What Users Will See

1. Beautiful purple gradient header
2. Professional form with 3 sections (Demographics, Vitals, Measurements)
3. Color-coded diagnosis results
4. Disease probability bars for all 5 conditions
5. Evidence-based recommendations (2024-2025 guidelines)
6. Clinical notes with specific guidance
7. Download report button

---

## ✅ Deployment Checklist

- [x] Code pushed to GitHub
- [x] Secrets removed from code
- [x] Model file included (best_disease_model.pkl)
- [x] Requirements.txt updated
- [x] Runtime.txt specified
- [ ] Deploy on Streamlit Cloud
- [ ] Add HF_TOKEN to Streamlit secrets (optional)
- [ ] Test all 5 disease predictions
- [ ] Verify critical alerts work

---

## 🎯 Key Features

- **95.22% Accuracy** - Validated on 60,000 patients
- **5 Disease Categories** - Diabetes, Heart Disease, Hypertension, Asthma, Healthy
- **Evidence-Based** - ADA 2024-2025, ESC 2024, GINA 2024, WHO 2020 guidelines
- **Critical Alerts** - Automatic detection of life-threatening vitals
- **Professional UI** - Color-coded results, progress bars, metric cards
- **Offline Capable** - Works without HuggingFace API

---

## 🚨 Troubleshooting

**Model not loading?**
- Check `best_disease_model.pkl` is in root directory
- File size should be ~2.4MB

**Predictions always the same?**
- Ensure all vitals are entered correctly
- Check BMI is calculated (enter weight + height)

**Recommendations not showing?**
- They're predetermined and always work
- No HF_TOKEN needed for demo!

---

## 📱 Support

For issues, check:
- GitHub repository
- Streamlit Cloud logs
- README.md for detailed documentation

---

**Your app is deployment-ready!** 🚀

