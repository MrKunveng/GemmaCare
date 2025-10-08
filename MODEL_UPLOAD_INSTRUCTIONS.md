# 📦 Model File Upload Instructions

## ⚠️ Important: Model File Not in GitHub

The `disease_prediction_ensemble.pkl` file (157MB) is **too large for GitHub** (100MB limit) and was excluded from the repository.

---

## Option 1: Upload via Streamlit Cloud (Recommended)

### Steps:

1. **Deploy the app on Streamlit Cloud:**
   - Go to https://share.streamlit.io
   - Connect to repository: `MrKunveng/GemmaCare`
   - Select branch: `main`
   - Main file: `app.py`
   - Click "Deploy"

2. **The app will show an error** (model file missing) - this is expected!

3. **Upload the model file:**
   - In Streamlit Cloud dashboard, go to your app
   - Click the **3 dots menu** → **Settings**
   - Go to **"Files"** or **"Manage app"** section
   - Upload `disease_prediction_ensemble.pkl` to the root directory

4. **Reboot the app** - it should now work!

---

## Option 2: Use Git LFS (Advanced)

If you have Git LFS installed:

```bash
# Install Git LFS (if not installed)
brew install git-lfs  # macOS
# or
sudo apt-get install git-lfs  # Linux

# Initialize LFS
git lfs install

# Track the model file
git lfs track "*.pkl"

# Add .gitattributes
git add .gitattributes

# Add and commit the model
git add disease_prediction_ensemble.pkl
git commit -m "Add model file via Git LFS"
git push origin main
```

---

## Option 3: Cloud Storage (Production)

For production deployments, use cloud storage:

### Using Google Drive:

1. **Upload model to Google Drive**
2. **Make it publicly accessible** (or use service account)
3. **Update `app.py`:**

```python
import gdown

@st.cache_resource(show_spinner=False)
def load_model():
    model_url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
    output = "disease_prediction_ensemble.pkl"
    
    if not os.path.exists(output):
        gdown.download(model_url, output, quiet=False)
    
    return joblib.load(output)
```

4. **Add to requirements.txt:** `gdown>=4.7.1`

### Using AWS S3:

```python
import boto3

@st.cache_resource
def load_model():
    s3 = boto3.client('s3')
    s3.download_file('your-bucket', 'model.pkl', 'disease_prediction_ensemble.pkl')
    return joblib.load('disease_prediction_ensemble.pkl')
```

---

## Option 4: Model Hosting Services

Use specialized ML model hosting:

- **Hugging Face Hub** (free for public models)
- **AWS S3** + CloudFront
- **Google Cloud Storage**
- **Azure Blob Storage**

---

## Current Model Location

📍 The model file is currently at:
```
/Users/lukman/Desktop/Ideathon/Deployment/disease_prediction_ensemble.pkl
```

**File size:** 157MB  
**Model type:** VotingClassifier ensemble (XGBoost-based)

---

## Quick Fix for Now

**Easiest approach:**
1. Deploy app to Streamlit Cloud from GitHub
2. Manually upload `disease_prediction_ensemble.pkl` through Streamlit Cloud UI
3. Done! ✅

The app will automatically detect and load the model once it's in the same directory.

---

## Need Help?

- Streamlit Cloud docs: https://docs.streamlit.io/streamlit-community-cloud
- Git LFS guide: https://git-lfs.github.com
- Contact: Check deployment logs for specific errors

