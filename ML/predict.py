#!/usr/bin/env python3
"""
Disease Prediction Using Saved Model
====================================

This script loads the best trained model and makes predictions
on new patient data.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

class DiseasePredictionSystem:
    """
    System for making disease predictions using the trained model
    """
    
    def __init__(self, model_path):
        """
        Initialize the prediction system by loading the saved model
        """
        print("Loading trained model...")
        self.model_package = joblib.load(model_path)
        
        self.model = self.model_package['model']
        self.model_name = self.model_package['model_name']
        self.label_encoders = self.model_package['label_encoders']
        self.target_encoder = self.model_package['target_encoder']
        self.scaler = self.model_package['scaler']
        self.feature_columns = self.model_package['feature_columns']
        
        print(f"✓ Model loaded: {self.model_name}")
        print(f"✓ Expected features: {len(self.feature_columns)}")
        print(f"✓ Disease classes: {len(self.target_encoder.classes_)}")
    
    def predict(self, patient_data):
        """
        Make prediction for a single patient or multiple patients
        
        Args:
            patient_data: Dictionary or DataFrame containing patient features
        
        Returns:
            Dictionary with prediction and probabilities
        """
        # Convert to DataFrame if dictionary
        if isinstance(patient_data, dict):
            df = pd.DataFrame([patient_data])
        else:
            df = patient_data.copy()
        
        # Ensure all required features are present
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select and order features
        df = df[self.feature_columns]
        
        # Encode categorical features
        df_encoded = df.copy()
        for col, encoder in self.label_encoders.items():
            if col in df_encoded.columns:
                df_encoded[col] = encoder.transform(df_encoded[col].astype(str))
        
        # Scale features
        X_scaled = self.scaler.transform(df_encoded)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)
        predicted_disease = self.target_encoder.inverse_transform(prediction)
        
        # Get probabilities if available
        results = []
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)
            
            for i in range(len(predicted_disease)):
                prob_dict = {}
                for j, disease in enumerate(self.target_encoder.classes_):
                    prob_dict[disease] = float(probabilities[i][j])
                
                results.append({
                    'predicted_disease': predicted_disease[i],
                    'confidence': float(probabilities[i].max()),
                    'all_probabilities': prob_dict
                })
        else:
            for i in range(len(predicted_disease)):
                results.append({
                    'predicted_disease': predicted_disease[i],
                    'confidence': None,
                    'all_probabilities': None
                })
        
        return results if len(results) > 1 else results[0]
    
    def get_feature_info(self):
        """
        Get information about expected features
        """
        return {
            'feature_columns': self.feature_columns,
            'categorical_features': list(self.label_encoders.keys()),
            'model_name': self.model_name,
            'disease_classes': list(self.target_encoder.classes_)
        }
    
    def predict_batch(self, csv_file_path):
        """
        Make predictions for a batch of patients from a CSV file
        
        Args:
            csv_file_path: Path to CSV file with patient data
        
        Returns:
            DataFrame with predictions
        """
        print(f"Loading patient data from: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        
        # Remove disease column if present
        if 'Predicted Disease' in df.columns:
            print("Note: 'Predicted Disease' column found and will be ignored")
            df = df.drop('Predicted Disease', axis=1)
        
        print(f"Making predictions for {len(df)} patients...")
        results = self.predict(df)
        
        # Add predictions to dataframe
        df['Predicted_Disease'] = [r['predicted_disease'] for r in results]
        if results[0]['confidence'] is not None:
            df['Confidence'] = [r['confidence'] for r in results]
        
        return df


def demo_prediction():
    """
    Demonstrate how to use the prediction system
    """
    print("\n" + "=" * 70)
    print("DISEASE PREDICTION SYSTEM - DEMO")
    print("=" * 70)
    
    # Load the model
    model_path = '/Users/lukman/Desktop/Ideathon/Deployment/ML/best_disease_model.pkl'
    predictor = DiseasePredictionSystem(model_path)
    
    # Show feature information
    info = predictor.get_feature_info()
    print(f"\nModel Information:")
    print(f"  - Model Type: {info['model_name']}")
    print(f"  - Number of Features: {len(info['feature_columns'])}")
    print(f"  - Disease Classes: {', '.join(info['disease_classes'])}")
    
    # Example patient data
    example_patient = {
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
    
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTION")
    print("=" * 70)
    print("\nPatient Data:")
    for key, value in example_patient.items():
        print(f"  - {key}: {value}")
    
    # Make prediction
    result = predictor.predict(example_patient)
    
    print(f"\nPrediction Results:")
    print(f"  - Predicted Disease: {result['predicted_disease']}")
    if result['confidence']:
        print(f"  - Confidence: {result['confidence']:.2%}")
        print(f"\n  Disease Probabilities:")
        for disease, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
            print(f"    • {disease}: {prob:.2%}")
    
    print("\n" + "=" * 70)


def predict_from_csv(csv_path, output_path=None):
    """
    Make predictions from a CSV file
    """
    print("\n" + "=" * 70)
    print("BATCH PREDICTION FROM CSV")
    print("=" * 70)
    
    # Load the model
    model_path = '/Users/lukman/Desktop/Ideathon/Deployment/ML/best_disease_model.pkl'
    predictor = DiseasePredictionSystem(model_path)
    
    # Make predictions
    results_df = predictor.predict_batch(csv_path)
    
    # Save results if output path provided
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"\n✓ Predictions saved to: {output_path}")
    
    # Show sample results
    print(f"\nSample Predictions (first 5 rows):")
    print(results_df[['Gender', 'Heart Rate (bpm)', 'Predicted_Disease']].head())
    
    # Show summary
    print(f"\nPrediction Summary:")
    print(results_df['Predicted_Disease'].value_counts())
    
    return results_df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # If CSV file path provided as argument
        csv_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        predict_from_csv(csv_path, output_path)
    else:
        # Run demo
        demo_prediction()
        
        print("\n" + "=" * 70)
        print("USAGE EXAMPLES:")
        print("=" * 70)
        print("\n1. For demo prediction:")
        print("   python predict.py")
        print("\n2. For batch prediction from CSV:")
        print("   python predict.py input.csv output.csv")
        print("\n")

