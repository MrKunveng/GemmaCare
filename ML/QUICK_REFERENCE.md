# 🚀 Quick Reference Guide

## Model Information

| Property | Value |
|----------|-------|
| **Model Type** | CatBoost Classifier |
| **Accuracy** | 95.30% |
| **F1 Score** | 95.29% |
| **File Size** | 2.4 MB |
| **Training Samples** | 48,000 |
| **Test Samples** | 12,000 |

## Required Input Features (9 total)

```python
{
    'Gender': 'Male' or 'Female',
    'Heart Rate (bpm)': float,          # e.g., 95
    'SpO2 Level (%)': float,            # e.g., 96
    'Systolic Blood Pressure (mmHg)': float,   # e.g., 140
    'Diastolic Blood Pressure (mmHg)': float,  # e.g., 90
    'Body Temperature (C)': float,      # e.g., 37.2
    'Weight_kg': float,                 # e.g., 85
    'Height_cm': float,                 # e.g., 175
    'BMI': float                        # e.g., 27.8
}
```

## Predicted Diseases (5 classes)

1. ✅ **Healthy** (100% precision)
2. 🫁 **Asthma** (93% F1)
3. 💉 **Diabetes Mellitus** (92% F1)
4. ❤️ **Heart Disease** (96% F1)
5. 🩺 **Hypertension** (96% F1)

## Quick Usage

### 1️⃣ Single Prediction

```python
from predict import DiseasePredictionSystem

predictor = DiseasePredictionSystem('best_disease_model.pkl')

patient = {
    'Gender': 'Male',
    'Heart Rate (bpm)': 95,
    'SpO2 Level (%)': 96,
    'Systolic Blood Pressure (mmHg)': 140,
    'Diastolic Blood Pressure (mmHg)': 90,
    'Body Temperature (C)': 37.2,
    'Weight_kg': 85,
    'Height_cm': 175,
    'BMI': 27.8
}

result = predictor.predict(patient)
print(result['predicted_disease'])  # e.g., "Diabetes Mellitus"
print(f"{result['confidence']:.1%}")  # e.g., "98.8%"
```

### 2️⃣ Command Line Demo

```bash
python predict.py
```

### 3️⃣ Batch Prediction

```bash
python predict.py patients.csv predictions.csv
```

## Output Format

```python
{
    'predicted_disease': 'Diabetes Mellitus',
    'confidence': 0.9878,
    'all_probabilities': {
        'Asthma': 0.0011,
        'Diabetes Mellitus': 0.9878,
        'Healthy': 0.0007,
        'Heart Disease': 0.0069,
        'Hypertension': 0.0035
    }
}
```

## Files You Need

| File | Purpose | Required |
|------|---------|----------|
| `best_disease_model.pkl` | Trained model | ✅ Yes |
| `predict.py` | Prediction script | ✅ Yes |
| `train_best_model.py` | Training script | ❌ No (for retraining only) |
| `Patient_dataset.csv` | Training data | ❌ No (for retraining only) |

## Common Issues & Solutions

### ❌ "Missing required features"
**Solution:** Ensure all 9 features are present with exact names (case-sensitive)

### ❌ "Model file not found"
**Solution:** Check the path to `best_disease_model.pkl`

### ❌ "Unexpected feature values"
**Solution:** Check data types (Gender must be string, others numeric)

## Integration Example

```python
# For Flask API
from flask import Flask, request, jsonify
from predict import DiseasePredictionSystem

app = Flask(__name__)
predictor = DiseasePredictionSystem('best_disease_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    result = predictor.predict(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
```

## Performance Characteristics

- **Prediction Time**: ~5ms per sample
- **Memory Usage**: ~50MB loaded
- **Thread Safe**: Yes
- **GPU Required**: No
- **Batch Optimized**: Yes

## Confidence Interpretation

| Confidence | Interpretation |
|------------|----------------|
| > 95% | Very High - Reliable prediction |
| 80-95% | High - Good confidence |
| 60-80% | Moderate - Consider additional tests |
| < 60% | Low - Uncertain, needs review |

