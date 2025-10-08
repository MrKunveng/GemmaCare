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
2. Hugging Face account with API token

### Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add Streamlit triage app"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository and branch
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Configure Secrets**
   - In Streamlit Cloud, go to: App Settings → Secrets
   - Add your Hugging Face token:
   ```toml
   HF_TOKEN = "your_huggingface_token_here"
   ```

4. **Monitor Deployment**
   - Check logs for any errors
   - Model file (157MB) should load automatically
   - First deployment may take 3-5 minutes

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

