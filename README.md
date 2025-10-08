# Ensemble + MedGemma Triage Demo

Research-only prototype for disease prediction and triage recommendations.

## Features
- Ensemble ML model for disease prediction
- MedGemma AI integration for clinical recommendations
- Patient vitals input and analysis
- Risk assessment and triage guidance

## Deployment to Streamlit Cloud

### Prerequisites
1. GitHub account
2. Hugging Face account with API token (optional)
3. Model file: `disease_prediction_ensemble.pkl`

### ⚠️ Important: Model File Upload Required

The model file (157MB) is **too large for GitHub**. You'll need to upload it separately.  
**See:** [`MODEL_UPLOAD_INSTRUCTIONS.md`](MODEL_UPLOAD_INSTRUCTIONS.md) for detailed steps.

### Steps

1. **✅ Code Already Pushed to GitHub**
   - Repository: https://github.com/MrKunveng/GemmaCare
   - Branch: `main`

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select repository: `MrKunveng/GemmaCare`
   - Branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

3. **Upload Model File** ⚠️ REQUIRED
   - App will error without model file
   - In Streamlit Cloud: Settings → Upload `disease_prediction_ensemble.pkl`
   - See [`MODEL_UPLOAD_INSTRUCTIONS.md`](MODEL_UPLOAD_INSTRUCTIONS.md)

4. **Configure Secrets** (Optional - for AI recommendations)
   - In Streamlit Cloud: App Settings → Secrets
   - Add your Hugging Face token:
   ```toml
   HF_TOKEN = "your_huggingface_token_here"
   ```

5. **Monitor Deployment**
   - Check logs for any errors
   - First deployment may take 2-3 minutes
   - Test with sample patient data

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variable
export HF_TOKEN="your_token"

# Run locally
streamlit run app.py
```

## Important Notes
- **Not for clinical use** - Research prototype only
- Model file is 157MB (ensure your Streamlit plan supports this)
- Requires Python 3.11+
- HF_TOKEN must be configured for MedGemma recommendations

## File Structure
```
.
├── app.py                              # Main Streamlit app
├── disease_prediction_ensemble.pkl     # ML model (157MB)
├── requirements.txt                    # Python dependencies
├── runtime.txt                         # Python version
└── .streamlit/
    └── config.toml                     # Streamlit configuration
```

