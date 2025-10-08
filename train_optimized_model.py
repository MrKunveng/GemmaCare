#!/usr/bin/env python3
"""
Optimized Lightweight Ensemble Model for Disease Prediction
===========================================================

This creates a smaller, production-ready model by:
1. Using only essential models (XGBoost + LightGBM)
2. Reducing model complexity (fewer trees)
3. Saving only the ensemble model (not individual models separately)
4. Using compression
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class OptimizedDiseasePrediction:
    """
    Lightweight ensemble model for disease prediction
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.ensemble_model = None
        self.feature_names = None
        self.target_encoder = LabelEncoder()
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the dataset"""
        print("Loading data...")
        df = pd.read_csv(file_path)
        print(f"Dataset shape: {df.shape}")
        
        # Feature engineering
        df = self._engineer_features(df)
        
        # Prepare features and target
        X, y = self._prepare_features_target(df)
        
        # Encode categorical features
        X_encoded = self._encode_categorical_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Feature selection (keep top 15 features to reduce size)
        self.feature_selector = SelectKBest(f_classif, k=min(15, X_train.shape[1]))
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_test_selected = self.feature_selector.transform(X_test)
        
        # Get selected feature names
        selected_indices = self.feature_selector.get_support(indices=True)
        self.feature_names = [X_train.columns[i] for i in selected_indices]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        self.X_train = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        self.X_test = pd.DataFrame(X_test_scaled, columns=self.feature_names)
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Selected features ({len(self.feature_names)}): {self.feature_names}")
        print(f"Training set: {self.X_train.shape}, Test set: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def _engineer_features(self, df):
        """Engineer additional features"""
        df = df.copy()
        
        # Create BMI categories
        df['BMI_Category'] = pd.cut(df['BMI'], 
                                     bins=[0, 18.5, 25, 30, float('inf')],
                                     labels=[0, 1, 2, 3])
        
        # Vital Risk Score
        df['Vital_Risk_Score'] = 0
        df.loc[df['Systolic Blood Pressure (mmHg)'] >= 140, 'Vital_Risk_Score'] += 1
        df.loc[df['Diastolic Blood Pressure (mmHg)'] >= 90, 'Vital_Risk_Score'] += 1
        df.loc[df['SpO2 Level (%)'] < 95, 'Vital_Risk_Score'] += 1
        df.loc[df['Body Temperature (C)'] >= 38, 'Vital_Risk_Score'] += 1
        
        # Alert Count
        df['Alert_Count'] = (
            (df['Systolic Blood Pressure (mmHg)'] >= 160).astype(int) +
            (df['Diastolic Blood Pressure (mmHg)'] >= 100).astype(int) +
            (df['SpO2 Level (%)'] < 92).astype(int)
        )
        
        return df
    
    def _prepare_features_target(self, df):
        """Prepare features and target"""
        # Target
        y = self.target_encoder.fit_transform(df['Predicted Disease'])
        
        # Features - select key columns
        feature_cols = [
            'Gender', 'Heart Rate (bpm)', 'SpO2 Level (%)',
            'Systolic Blood Pressure (mmHg)', 'Diastolic Blood Pressure (mmHg)',
            'Body Temperature (C)', 'Weight_kg', 'BMI', 
            'BMI_Category', 'Vital_Risk_Score', 'Alert_Count'
        ]
        
        X = df[feature_cols].copy()
        return X, y
    
    def _encode_categorical_features(self, X):
        """Encode categorical features"""
        X_encoded = X.copy()
        categorical_cols = X_encoded.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X_encoded[col] = self.label_encoders[col].fit_transform(X_encoded[col].astype(str))
            else:
                X_encoded[col] = self.label_encoders[col].transform(X_encoded[col].astype(str))
        
        return X_encoded
    
    def create_and_train_ensemble(self):
        """Create and train a lightweight ensemble"""
        print("\nCreating optimized ensemble model...")
        
        # Lightweight XGBoost (50 trees instead of 200)
        xgb_model = xgb.XGBClassifier(
            n_estimators=50,  # Reduced from 200
            max_depth=4,      # Reduced from 6
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric='mlogloss',
            tree_method='hist'  # Faster and more memory efficient
        )
        
        # Lightweight LightGBM (50 trees instead of 200)
        lgb_model = lgb.LGBMClassifier(
            n_estimators=50,  # Reduced from 200
            max_depth=4,      # Reduced from 6
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbose=-1
        )
        
        # Create voting ensemble (only 2 models instead of 4)
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('lgb', lgb_model)
            ],
            voting='soft'
        )
        
        # Train ensemble
        print("Training ensemble...")
        self.ensemble_model.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred = self.ensemble_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='macro')
        
        print(f"\n✓ Model trained successfully!")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.ensemble_model, self.X_train, self.y_train, 
                                     cv=5, scoring='f1_macro')
        print(f"  CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Detailed report
        print("\nClassification Report:")
        y_test_labels = self.target_encoder.inverse_transform(self.y_test)
        y_pred_labels = self.target_encoder.inverse_transform(y_pred)
        print(classification_report(y_test_labels, y_pred_labels))
        
        return self.ensemble_model
    
    def save_optimized_model(self, filepath):
        """Save only essential components (smaller file)"""
        print(f"\nSaving optimized model to {filepath}...")
        
        # Only save what's needed for inference - NO redundant individual models!
        model_data = {
            'ensemble_model': self.ensemble_model,  # This already contains the trained models
            'target_encoder': self.target_encoder,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'label_encoders': self.label_encoders
        }
        
        # Save with compression
        joblib.dump(model_data, filepath, compress=3)
        
        # Check file size
        import os
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"✓ Model saved! File size: {file_size_mb:.2f} MB")
        
        return file_size_mb

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("OPTIMIZED DISEASE PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Initialize
    model = OptimizedDiseasePrediction(random_state=42)
    
    # Load and preprocess
    X_train, X_test, y_train, y_test = model.load_and_preprocess_data(
        'Patient_dataset.csv'
    )
    
    # Train ensemble
    ensemble = model.create_and_train_ensemble()
    
    # Save optimized model
    file_size = model.save_optimized_model('disease_prediction_optimized.pkl')
    
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Model file: disease_prediction_optimized.pkl ({file_size:.2f} MB)")
    print("This optimized model is:")
    print("  • Smaller in size (using compression)")
    print("  • Faster for inference (fewer trees)")
    print("  • Production-ready (no redundant data)")
    print("=" * 60)

if __name__ == "__main__":
    main()

