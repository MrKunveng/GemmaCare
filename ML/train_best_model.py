#!/usr/bin/env python3
"""
Train and Save Best Disease Prediction Model
============================================

This script trains multiple models, evaluates them, and saves only
the best performing model for disease prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

class BestModelTrainer:
    """
    Train multiple models and save the best one
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_columns = None
        self.target_encoder = None
        
    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess the dataset
        """
        print("=" * 70)
        print("LOADING AND PREPROCESSING DATA")
        print("=" * 70)
        
        # Load data
        df = pd.read_csv(file_path)
        print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Define feature columns (all except the target)
        self.feature_columns = [col for col in df.columns if col != 'Predicted Disease']
        print(f"✓ Features identified: {len(self.feature_columns)} features")
        
        # Separate features and target
        X = df[self.feature_columns].copy()
        y = df['Predicted Disease'].copy()
        
        # Show disease distribution
        print(f"\n✓ Disease distribution:")
        for disease, count in y.value_counts().items():
            print(f"  - {disease}: {count} ({count/len(y)*100:.1f}%)")
        
        # Encode categorical features
        X_encoded = self._encode_features(X)
        
        # Encode target variable
        self.target_encoder = LabelEncoder()
        y_encoded = self.target_encoder.fit_transform(y)
        print(f"\n✓ Target encoded: {len(self.target_encoder.classes_)} classes")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=0.2, random_state=self.random_state, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\n✓ Data split:")
        print(f"  - Training set: {X_train_scaled.shape[0]} samples")
        print(f"  - Test set: {X_test_scaled.shape[0]} samples")
        print(f"  - Features: {X_train_scaled.shape[1]}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def _encode_features(self, X):
        """
        Encode categorical features
        """
        X_encoded = X.copy()
        
        # Identify categorical columns (non-numeric)
        categorical_columns = X_encoded.select_dtypes(include=['object']).columns.tolist()
        
        print(f"\n✓ Encoding {len(categorical_columns)} categorical features:")
        for col in categorical_columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            self.label_encoders[col] = le
            print(f"  - {col}: {len(le.classes_)} categories")
        
        return X_encoded
    
    def create_models(self):
        """
        Create multiple models for evaluation
        """
        print("\n" + "=" * 70)
        print("CREATING MODELS")
        print("=" * 70)
        
        # Random Forest
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # XGBoost
        self.models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric='mlogloss',
            n_jobs=-1
        )
        
        # LightGBM
        self.models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbose=-1,
            n_jobs=-1
        )
        
        # CatBoost
        self.models['CatBoost'] = CatBoostClassifier(
            iterations=200,
            depth=8,
            learning_rate=0.1,
            random_state=self.random_state,
            verbose=False
        )
        
        # Gradient Boosting
        self.models['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=self.random_state
        )
        
        # Logistic Regression
        self.models['Logistic Regression'] = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            multi_class='ovr',
            n_jobs=-1
        )
        
        # SVM (for smaller datasets or as comparison)
        self.models['SVM'] = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=self.random_state
        )
        
        print(f"✓ Created {len(self.models)} models:")
        for i, name in enumerate(self.models.keys(), 1):
            print(f"  {i}. {name}")
        
        return self.models
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Train all models and evaluate them
        """
        print("\n" + "=" * 70)
        print("TRAINING AND EVALUATING MODELS")
        print("=" * 70)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n{'─' * 70}")
            print(f"Training: {name}")
            print(f"{'─' * 70}")
            
            # Train model
            model.fit(X_train, y_train)
            print("✓ Training completed")
            
            # Cross-validation score
            print("  Performing 5-fold cross-validation...")
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            print(f"  ✓ CV F1 Score: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
            
            # Test set evaluation
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            
            results[name] = {
                'model': model,
                'cv_f1': cv_mean,
                'cv_std': cv_std,
                'test_accuracy': accuracy,
                'test_f1': f1
            }
            
            print(f"  ✓ Test Accuracy: {accuracy:.4f}")
            print(f"  ✓ Test F1 Score: {f1:.4f}")
        
        return results
    
    def select_best_model(self, results):
        """
        Select the best model based on test F1 score
        """
        print("\n" + "=" * 70)
        print("MODEL PERFORMANCE COMPARISON")
        print("=" * 70)
        
        # Sort by test F1 score
        sorted_results = sorted(results.items(), key=lambda x: x[1]['test_f1'], reverse=True)
        
        print(f"\n{'Rank':<6} {'Model':<25} {'CV F1':<12} {'Test Acc':<12} {'Test F1':<12}")
        print("─" * 70)
        
        for i, (name, metrics) in enumerate(sorted_results, 1):
            print(f"{i:<6} {name:<25} {metrics['cv_f1']:.4f}      {metrics['test_accuracy']:.4f}      {metrics['test_f1']:.4f}")
        
        # Select best model
        self.best_model_name = sorted_results[0][0]
        self.best_model = results[self.best_model_name]['model']
        
        print("\n" + "=" * 70)
        print(f"BEST MODEL SELECTED: {self.best_model_name}")
        print("=" * 70)
        print(f"✓ CV F1 Score: {results[self.best_model_name]['cv_f1']:.4f}")
        print(f"✓ Test Accuracy: {results[self.best_model_name]['test_accuracy']:.4f}")
        print(f"✓ Test F1 Score: {results[self.best_model_name]['test_f1']:.4f}")
        
        return self.best_model, self.best_model_name
    
    def show_detailed_report(self, X_test, y_test):
        """
        Show detailed classification report for the best model
        """
        print("\n" + "=" * 70)
        print("DETAILED CLASSIFICATION REPORT")
        print("=" * 70)
        
        y_pred = self.best_model.predict(X_test)
        
        # Convert back to original labels
        y_test_original = self.target_encoder.inverse_transform(y_test)
        y_pred_original = self.target_encoder.inverse_transform(y_pred)
        
        print("\n", classification_report(y_test_original, y_pred_original))
    
    def save_best_model(self, filepath):
        """
        Save the best model with all preprocessing components
        """
        print("\n" + "=" * 70)
        print("SAVING BEST MODEL")
        print("=" * 70)
        
        model_package = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'label_encoders': self.label_encoders,
            'target_encoder': self.target_encoder,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_package, filepath)
        
        print(f"✓ Model saved to: {filepath}")
        print(f"✓ Model type: {self.best_model_name}")
        print(f"✓ Expected features: {len(self.feature_columns)}")
        print(f"✓ Disease classes: {len(self.target_encoder.classes_)}")
        print(f"\nFeature columns saved:")
        for i, col in enumerate(self.feature_columns, 1):
            print(f"  {i:2d}. {col}")
        
        return filepath
    
    def test_prediction(self, X_test, y_test):
        """
        Test a sample prediction to verify the model works
        """
        print("\n" + "=" * 70)
        print("TESTING SAMPLE PREDICTION")
        print("=" * 70)
        
        # Get a random sample
        sample_idx = np.random.randint(0, len(X_test))
        sample = X_test[sample_idx:sample_idx+1]
        true_label = self.target_encoder.inverse_transform([y_test[sample_idx]])[0]
        
        # Make prediction
        prediction = self.best_model.predict(sample)
        predicted_label = self.target_encoder.inverse_transform(prediction)[0]
        
        # Get probability
        if hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(sample)[0]
            confidence = probabilities.max()
            print(f"✓ Sample prediction test:")
            print(f"  - True Disease: {true_label}")
            print(f"  - Predicted Disease: {predicted_label}")
            print(f"  - Confidence: {confidence:.2%}")
            print(f"  - Match: {'✓ Correct' if true_label == predicted_label else '✗ Incorrect'}")
        else:
            print(f"✓ Sample prediction test:")
            print(f"  - True Disease: {true_label}")
            print(f"  - Predicted Disease: {predicted_label}")
            print(f"  - Match: {'✓ Correct' if true_label == predicted_label else '✗ Incorrect'}")

def main():
    """
    Main function to run the training pipeline
    """
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  DISEASE PREDICTION - BEST MODEL TRAINING PIPELINE".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Initialize trainer
    trainer = BestModelTrainer(random_state=42)
    
    # Load and preprocess data
    dataset_path = '/Users/lukman/Desktop/Ideathon/Deployment/ML/Patient_dataset.csv'
    X_train, X_test, y_train, y_test = trainer.load_and_preprocess_data(dataset_path)
    
    # Create models
    trainer.create_models()
    
    # Train and evaluate all models
    results = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Select best model
    best_model, best_model_name = trainer.select_best_model(results)
    
    # Show detailed report
    trainer.show_detailed_report(X_test, y_test)
    
    # Save the best model
    model_filepath = '/Users/lukman/Desktop/Ideathon/Deployment/ML/best_disease_model.pkl'
    trainer.save_best_model(model_filepath)
    
    # Test prediction
    trainer.test_prediction(X_test, y_test)
    
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n✓ Best Model: {best_model_name}")
    print(f"✓ Model File: {model_filepath}")
    print(f"✓ Ready for deployment!")
    print("\n")

if __name__ == "__main__":
    main()

