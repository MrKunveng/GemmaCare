# Disease Prediction Model

This directory contains the trained disease prediction model and associated scripts.

## 📊 Model Performance

**Best Model:** CatBoost Classifier

| Metric | Score |
|--------|-------|
| Cross-Validation F1 Score | 95.66% |
| Test Accuracy | 95.30% |
| Test F1 Score | 95.29% |

### Model Comparison

All models were trained and evaluated. Here are the results:

| Rank | Model | CV F1 | Test Accuracy | Test F1 |
|------|-------|-------|---------------|---------|
| 1 | **CatBoost** | 0.9566 | 0.9530 | 0.9529 |
| 2 | Random Forest | 0.9565 | 0.9523 | 0.9522 |
| 3 | XGBoost | 0.9555 | 0.9519 | 0.9518 |
| 4 | LightGBM | 0.9549 | 0.9513 | 0.9512 |
| 5 | Gradient Boosting | 0.9553 | 0.9508 | 0.9507 |
| 6 | SVM | 0.9426 | 0.9408 | 0.9409 |
| 7 | Logistic Regression | 0.9196 | 0.9171 | 0.9170 |

## 🎯 Disease Classes

The model predicts 5 disease categories:

1. **Asthma** (Precision: 92%, Recall: 93%, F1: 93%)
2. **Diabetes Mellitus** (Precision: 94%, Recall: 90%, F1: 92%)
3. **Healthy** (Precision: 100%, Recall: 100%, F1: 100%)
4. **Heart Disease** (Precision: 93%, Recall: 99%, F1: 96%)
5. **Hypertension** (Precision: 97%, Recall: 95%, F1: 96%)

## 📁 Files

- `train_best_model.py` - Script to train all models and save the best one
- `predict.py` - Script to make predictions using the saved model
- `best_disease_model.pkl` - Saved best model (CatBoost)
- `Patient_dataset.csv` - Training dataset (60,000 samples)
- `ensemble_disease_prediction.py` - Original ensemble approach (for reference)

## 🔧 Required Features

The model expects the following 9 features for prediction:

1. **Gender** (categorical: Male/Female)
2. **Heart Rate (bpm)** (numeric)
3. **SpO2 Level (%)** (numeric: 0-100)
4. **Systolic Blood Pressure (mmHg)** (numeric)
5. **Diastolic Blood Pressure (mmHg)** (numeric)
6. **Body Temperature (C)** (numeric)
7. **Weight_kg** (numeric)
8. **Height_cm** (numeric)
9. **BMI** (numeric)

## 🚀 Usage

### Training the Model

To retrain the model from scratch:

```bash
python train_best_model.py
```

This will:
- Load and preprocess the dataset
- Train 7 different models
- Evaluate and compare all models
- Save the best performing model to `best_disease_model.pkl`

### Making Predictions

#### Option 1: Demo Prediction (Single Patient)

```bash
python predict.py
```

This runs a demo with sample patient data.

#### Option 2: Single Patient Prediction (Python Script)

```python
from predict import DiseasePredictionSystem

# Load the model
predictor = DiseasePredictionSystem('best_disease_model.pkl')

# Patient data
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

# Make prediction
result = predictor.predict(patient)

print(f"Predicted Disease: {result['predicted_disease']}")
print(f"Confidence: {result['confidence']:.2%}")
```

#### Option 3: Batch Prediction from CSV

```bash
python predict.py input.csv output.csv
```

This will:
- Load patient data from `input.csv`
- Make predictions for all patients
- Save results to `output.csv`

### Python API Example

```python
from predict import DiseasePredictionSystem
import pandas as pd

# Initialize predictor
predictor = DiseasePredictionSystem('best_disease_model.pkl')

# Get feature information
info = predictor.get_feature_info()
print(f"Model: {info['model_name']}")
print(f"Features: {info['feature_columns']}")
print(f"Disease Classes: {info['disease_classes']}")

# Single prediction
patient_data = {
    'Gender': 'Female',
    'Heart Rate (bpm)': 88,
    'SpO2 Level (%)': 98,
    'Systolic Blood Pressure (mmHg)': 165,
    'Diastolic Blood Pressure (mmHg)': 110,
    'Body Temperature (C)': 37.0,
    'Weight_kg': 70,
    'Height_cm': 165,
    'BMI': 25.7
}

result = predictor.predict(patient_data)

print(f"\nPrediction: {result['predicted_disease']}")
print(f"Confidence: {result['confidence']:.2%}")

# Show all disease probabilities
for disease, prob in result['all_probabilities'].items():
    print(f"  {disease}: {prob:.2%}")

# Batch prediction
df = pd.read_csv('patients.csv')
results = predictor.predict(df)

for i, result in enumerate(results):
    print(f"Patient {i+1}: {result['predicted_disease']} ({result['confidence']:.1%})")
```

## 📋 Input Data Format

### CSV File Format

Your CSV file should have these columns (in any order):

```csv
Gender,Heart Rate (bpm),SpO2 Level (%),Systolic Blood Pressure (mmHg),Diastolic Blood Pressure (mmHg),Body Temperature (C),Weight_kg,Height_cm,BMI
Male,95,96,140,90,37.2,85,175,27.8
Female,88,98,165,110,37.0,70,165,25.7
```

### Dictionary Format (Python)

```python
{
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
```

## 📊 Model Output

The prediction returns:

```python
{
    'predicted_disease': 'Diabetes Mellitus',
    'confidence': 0.9878,  # 98.78%
    'all_probabilities': {
        'Asthma': 0.0011,
        'Diabetes Mellitus': 0.9878,
        'Healthy': 0.0007,
        'Heart Disease': 0.0069,
        'Hypertension': 0.0035
    }
}
```

## 🔍 Model Details

### Preprocessing Steps

1. **Feature Encoding**: Categorical features (Gender) are label-encoded
2. **Feature Scaling**: All features are standardized using StandardScaler
3. **No Missing Values**: Dataset is complete with no missing values

### Model Configuration

**CatBoost Classifier:**
- Iterations: 200
- Depth: 8
- Learning Rate: 0.1
- Random State: 42

### Training Dataset

- **Total Samples**: 60,000
- **Training Set**: 48,000 (80%)
- **Test Set**: 12,000 (20%)
- **Class Distribution**:
  - Diabetes Mellitus: 20.3%
  - Heart Disease: 20.1%
  - Healthy: 20.0%
  - Hypertension: 19.9%
  - Asthma: 19.8%

## 🛠️ Requirements

```
pandas
numpy
scikit-learn
xgboost
lightgbm
catboost
joblib
```

## 📝 Notes

- The model expects all 9 features to be present in the input data
- Feature names must match exactly (case-sensitive)
- The model handles categorical encoding automatically
- All preprocessing is included in the saved model package
- The model provides probability distributions for all disease classes

## 🎓 Model Interpretability

The CatBoost model was selected because:
- **Highest F1 Score**: Best balance between precision and recall
- **Robust**: Handles categorical features natively
- **Fast Inference**: Quick prediction time
- **Probability Calibration**: Provides reliable confidence scores
- **No Overfitting**: Consistent performance on validation and test sets

## 📈 Future Improvements

Potential enhancements:
- Feature engineering (e.g., vital signs risk scores)
- Ensemble of top 3 models for even better performance
- SHAP values for model explainability
- API endpoint for real-time predictions
- Model monitoring and retraining pipeline

