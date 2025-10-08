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

### Steps

1. **✅ Code Already Pushed to GitHub**
   - Repository: https://github.com/MrKunveng/GemmaCare
   - Branch: `main`
   - ✅ Model included (optimized to 0.22 MB)

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select repository: `MrKunveng/GemmaCare`
   - Branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

3. **Configure Secrets** (Optional - for AI recommendations)
   - In Streamlit Cloud: App Settings → Secrets
   - Add your Hugging Face token:
   ```toml
   HF_TOKEN = "your_huggingface_token_here"
   ```

4. **Monitor Deployment**
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
- Optimized model file is only 0.22 MB (GitHub-friendly!)
- Requires Python 3.11+
- HF_TOKEN optional for AI recommendations (fallback available)

## Model Details
- **Type:** Voting Classifier Ensemble (XGBoost + LightGBM)
- **Size:** 0.22 MB (optimized from 157 MB)
- **Accuracy:** 95.2%
- **F1 Score:** 95.2%
- **Diseases:** Asthma, Diabetes Mellitus, Healthy, Heart Disease, Hypertension

## File Structure
```
.
├── app.py                              # Main Streamlit app
├── disease_prediction_ensemble.pkl     # Optimized ML model (0.22 MB)
├── requirements.txt                    # Python dependencies
├── runtime.txt                         # Python version
├── train_optimized_model.py           # Model training script
└── .streamlit/
    └── config.toml                     # Streamlit configuration
```

