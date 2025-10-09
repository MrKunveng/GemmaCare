#!/usr/bin/env python3
"""
Robust Ensemble Machine Learning Model for Disease Prediction
============================================================

This script creates a comprehensive ensemble model to predict diseases based on
patient health monitoring data including vital signs, biometrics, and alert indicators.

Features:
- Multiple ensemble approaches (Voting, Stacking, Bagging)
- Advanced models (XGBoost, LightGBM, CatBoost, Random Forest)
- Comprehensive hyperparameter tuning
- Feature engineering and selection
- Cross-validation and robust evaluation
- Feature importance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

class DiseasePredictionEnsemble:
    """
    Ensemble model for disease prediction
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.models = {}
        self.ensemble_model = None
        self.feature_names = None
        
    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess the dataset
        """
        print("Loading and preprocessing data...")
        
        # Load data
        self.df = pd.read_csv(file_path)
        print(f"Dataset shape: {self.df.shape}")
        
        # Display basic info
        print(f"Unique diseases: {self.df['Predicted Disease'].value_counts()}")
        
        # Feature engineering
        self.df = self._engineer_features()
        
        # Prepare features and target
        X, y = self._prepare_features_target()
        
        # Encode categorical variables
        X_encoded = self._encode_categorical_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Feature selection
        X_train_selected, X_test_selected = self._select_features(X_train, X_test, y_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        self.X_train = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        self.X_test = pd.DataFrame(X_test_scaled, columns=self.feature_names)
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def _engineer_features(self):
        """
        Engineer additional features for better prediction
        """
        df = self.df.copy()
        
        # Create BMI categories
        df['BMI_Category'] = pd.cut(df['BMI'], 
                                   bins=[0, 18.5, 25, 30, float('inf')], 
                                   labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        
        # Create vital signs risk score
        df['Vital_Risk_Score'] = 0
        df.loc[df['Heart Rate (bpm)'] < 60, 'Vital_Risk_Score'] += 1
        df.loc[df['Heart Rate (bpm)'] > 100, 'Vital_Risk_Score'] += 1
        df.loc[df['SpO2 Level (%)'] < 95, 'Vital_Risk_Score'] += 1
        df.loc[df['Systolic Blood Pressure (mmHg)'] > 140, 'Vital_Risk_Score'] += 1
        df.loc[df['Diastolic Blood Pressure (mmHg)'] > 90, 'Vital_Risk_Score'] += 1
        df.loc[df['Body Temperature (C)'] < 36.0, 'Vital_Risk_Score'] += 1
        df.loc[df['Body Temperature (C)'] > 37.5, 'Vital_Risk_Score'] += 1
        
        # Create alert count
        alert_columns = ['Heart Rate Alert', 'SpO2 Level Alert', 'Blood Pressure Alert', 'Temperature Alert']
        df['Alert_Count'] = df[alert_columns].apply(lambda x: (x == 'ABNORMAL').sum(), axis=1)
        
        # Create age groups (if age available, otherwise use patient number as proxy)
        df['Age_Group'] = pd.cut(df['Patient Number'] % 100, 
                                bins=[0, 25, 45, 65, float('inf')], 
                                labels=['Young', 'Adult', 'Middle_Age', 'Senior'])
        
        return df
    
    def _prepare_features_target(self):
        """
        Prepare features and target variable
        """
        # Select features
        feature_columns = [
            'Gender', 'Heart Rate (bpm)', 'SpO2 Level (%)', 
            'Systolic Blood Pressure (mmHg)', 'Diastolic Blood Pressure (mmHg)',
            'Body Temperature (C)', 'Weight_kg', 'Height_cm', 'BMI',
            'BMI_Category', 'Vital_Risk_Score', 'Alert_Count', 'Age_Group',
            'Heart Rate Alert', 'SpO2 Level Alert', 'Blood Pressure Alert', 
            'Temperature Alert', 'Data Accuracy (%)'
        ]
        
        X = self.df[feature_columns].copy()
        y = self.df['Predicted Disease'].copy()
        
        # Encode target variable
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        self.target_encoder = target_encoder
        
        return X, y_encoded
    
    def _encode_categorical_features(self, X):
        """
        Encode categorical features
        """
        X_encoded = X.copy()
        
        categorical_columns = ['Gender', 'BMI_Category', 'Age_Group', 
                             'Heart Rate Alert', 'SpO2 Level Alert', 
                             'Blood Pressure Alert', 'Temperature Alert']
        
        for col in categorical_columns:
            if col in X_encoded.columns:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col])
                self.label_encoders[col] = le
        
        return X_encoded
    
    def _select_features(self, X_train, X_test, y_train, k=15):
        """
        Select most important features
        """
        self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_test_selected = self.feature_selector.transform(X_test)
        
        # Get selected feature names
        selected_indices = self.feature_selector.get_support(indices=True)
        self.feature_names = X_train.columns[selected_indices].tolist()
        
        print(f"Selected {len(self.feature_names)} features: {self.feature_names}")
        
        return X_train_selected, X_test_selected
    
    def create_individual_models(self):
        """
        Create individual models for the ensemble
        """
        print("Creating individual models...")
        
        # Random Forest
        self.models['RandomForest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # XGBoost
        self.models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric='mlogloss'
        )
        
        # LightGBM
        self.models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbose=-1
        )
        
        # CatBoost
        self.models['CatBoost'] = CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            verbose=False
        )
        
        # SVM
        self.models['SVM'] = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=self.random_state
        )
        
        # Logistic Regression
        self.models['LogisticRegression'] = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            multi_class='ovr'
        )
        
        return self.models
    
    def train_models(self):
        """
        Train all individual models
        """
        print("Training individual models...")
        model_scores = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='f1_macro')
            model_scores[name] = cv_scores.mean()
            print(f"{name} CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return model_scores
    
    def create_ensemble_models(self):
        """
        Create ensemble models
        """
        print("Creating ensemble models...")
        
        # Voting Classifier (Soft Voting)
        voting_models = [
            ('rf', self.models['RandomForest']),
            ('xgb', self.models['XGBoost']),
            ('lgb', self.models['LightGBM']),
            ('cat', self.models['CatBoost'])
        ]
        
        self.ensemble_model = VotingClassifier(
            estimators=voting_models,
            voting='soft'
        )
        
        # Train ensemble
        print("Training ensemble model...")
        self.ensemble_model.fit(self.X_train, self.y_train)
        
        # Cross-validation for ensemble
        cv_scores = cross_val_score(self.ensemble_model, self.X_train, self.y_train, cv=5, scoring='f1_macro')
        print(f"Ensemble CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self.ensemble_model
    
    def evaluate_models(self):
        """
        Evaluate all models on test set
        """
        print("Evaluating models on test set...")
        
        results = {}
        
        # Evaluate individual models
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='macro')
            results[name] = {'accuracy': accuracy, 'f1': f1}
            print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Evaluate ensemble
        y_pred_ensemble = self.ensemble_model.predict(self.X_test)
        accuracy_ensemble = accuracy_score(self.y_test, y_pred_ensemble)
        f1_ensemble = f1_score(self.y_test, y_pred_ensemble, average='macro')
        results['Ensemble'] = {'accuracy': accuracy_ensemble, 'f1': f1_ensemble}
        print(f"Ensemble - Accuracy: {accuracy_ensemble:.4f}, F1: {f1_ensemble:.4f}")
        
        # Detailed classification report for ensemble
        print("\nDetailed Classification Report for Ensemble Model:")
        # Convert encoded labels back to original disease names for reporting
        y_test_original = self.target_encoder.inverse_transform(self.y_test)
        y_pred_ensemble_original = self.target_encoder.inverse_transform(y_pred_ensemble)
        print(classification_report(y_test_original, y_pred_ensemble_original))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred_ensemble)
        plt.figure(figsize=(10, 8))
        disease_names = self.target_encoder.classes_
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=disease_names,
                   yticklabels=disease_names)
        plt.title('Confusion Matrix - Ensemble Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('/Users/lukman/Desktop/Ideathon/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results, y_pred_ensemble
    
    def analyze_feature_importance(self):
        """
        Analyze feature importance across models
        """
        print("Analyzing feature importance...")
        
        # Get feature importance from tree-based models
        importance_data = {}
        
        for name, model in self.models.items():
            try:
                # Check if model has feature_importances_ attribute
                if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
                    importance_data[name] = model.feature_importances_
                    print(f"✓ {name}: Feature importance extracted")
                # Special handling for XGBoost
                elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'get_score'):
                    try:
                        # XGBoost feature importance
                        booster = model.get_booster()
                        importance_dict = booster.get_score(importance_type='weight')
                        # Convert to array matching feature names
                        importance_array = []
                        for feature in self.feature_names:
                            importance_array.append(importance_dict.get(f'f{self.feature_names.index(feature)}', 0))
                        importance_data[name] = np.array(importance_array)
                        print(f"✓ {name}: XGBoost feature importance extracted")
                    except Exception as e:
                        print(f"⚠ {name}: Could not extract XGBoost importance: {e}")
                # Special handling for LightGBM
                elif hasattr(model, 'feature_importance_'):
                    importance_data[name] = model.feature_importance_
                    print(f"✓ {name}: LightGBM feature importance extracted")
                # Special handling for CatBoost
                elif hasattr(model, 'feature_importances_'):
                    importance_data[name] = model.feature_importances_
                    print(f"✓ {name}: CatBoost feature importance extracted")
                else:
                    print(f"⚠ {name}: No feature importance available")
            except Exception as e:
                print(f"❌ {name}: Error extracting feature importance: {e}")
        
        if not importance_data:
            print("❌ No feature importance data could be extracted from any model")
            return None, None
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame(importance_data, index=self.feature_names)
        
        print(f"\nFeature importance DataFrame shape: {importance_df.shape}")
        print("Models with feature importance:")
        for col in importance_df.columns:
            print(f"  - {col}: {importance_df[col].sum():.2f} total importance")
        
        # Plot feature importance
        plt.figure(figsize=(15, 10))
        
        # Create subplots for better visualization
        n_models = len(importance_data)
        if n_models <= 2:
            fig, axes = plt.subplots(1, n_models, figsize=(15, 8))
            if n_models == 1:
                axes = [axes]
        else:
            fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(15, 10))
            axes = axes.flatten()
        
        # Plot each model separately
        for i, (model_name, importance_values) in enumerate(importance_data.items()):
            if i < len(axes):
                # Sort features by importance for this model
                sorted_idx = np.argsort(importance_values)[::-1]
                sorted_features = [self.feature_names[idx] for idx in sorted_idx]
                sorted_importance = importance_values[sorted_idx]
                
                # Plot
                axes[i].barh(sorted_features, sorted_importance, alpha=0.7)
                axes[i].set_title(f'{model_name} Feature Importance', fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Importance')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(importance_data), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Feature Importance Across All Models', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('/Users/lukman/Desktop/Ideathon/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Also create a combined plot
        plt.figure(figsize=(12, 8))
        importance_df.plot(kind='bar', figsize=(12, 8), width=0.8)
        plt.title('Feature Importance Comparison Across Models', fontsize=14, fontweight='bold')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('/Users/lukman/Desktop/Ideathon/feature_importance_combined.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Average feature importance
        avg_importance = importance_df.mean(axis=1).sort_values(ascending=False)
        print("\nAverage Feature Importance:")
        for i, (feature, importance) in enumerate(avg_importance.items(), 1):
            print(f"{i:2d}. {feature:<30} {importance:8.2f}")
        
        return importance_df, avg_importance
    
    def hyperparameter_tuning(self):
        """
        Perform hyperparameter tuning for key models
        """
        print("Performing hyperparameter tuning...")
        
        # XGBoost tuning
        xgb_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        xgb_grid = GridSearchCV(
            xgb.XGBClassifier(random_state=self.random_state, eval_metric='mlogloss'),
            xgb_params,
            cv=3,
            scoring='f1_macro',
            n_jobs=-1
        )
        
        print("Tuning XGBoost...")
        xgb_grid.fit(self.X_train, self.y_train)
        self.models['XGBoost_Tuned'] = xgb_grid.best_estimator_
        print(f"Best XGBoost params: {xgb_grid.best_params_}")
        print(f"Best XGBoost score: {xgb_grid.best_score_:.4f}")
        
        # Random Forest tuning
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            rf_params,
            cv=3,
            scoring='f1_macro',
            n_jobs=-1
        )
        
        print("Tuning Random Forest...")
        rf_grid.fit(self.X_train, self.y_train)
        self.models['RandomForest_Tuned'] = rf_grid.best_estimator_
        print(f"Best Random Forest params: {rf_grid.best_params_}")
        print(f"Best Random Forest score: {rf_grid.best_score_:.4f}")
    
    def save_model(self, filepath):
        """
        Save the trained ensemble model
        """
        import joblib
        
        model_data = {
            'ensemble_model': self.ensemble_model,
            'models': self.models,
            'label_encoders': self.label_encoders,
            'target_encoder': self.target_encoder,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def predict_disease(self, patient_data):
        """
        Predict disease for new patient data
        """
        # Preprocess the input data
        patient_df = pd.DataFrame([patient_data])
        
        # Apply same preprocessing steps
        # (This would need to be implemented based on the exact format of input data)
        # For now, return a placeholder
        return "Prediction functionality needs patient data in the correct format"

def main():
    """
    Main function to run the complete pipeline
    """
    print("=== Disease Prediction Ensemble Model ===")
    
    # Initialize the ensemble
    ensemble = DiseasePredictionEnsemble(random_state=42)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = ensemble.load_and_preprocess_data(
        '/Users/lukman/Desktop/Ideathon/Patient_dataset.csv'
    )
    
    # Create and train individual models
    ensemble.create_individual_models()
    model_scores = ensemble.train_models()
    
    # Perform hyperparameter tuning
    ensemble.hyperparameter_tuning()
    
    # Create ensemble model
    ensemble.create_ensemble_models()
    
    # Evaluate models
    results, y_pred = ensemble.evaluate_models()
    
    # Analyze feature importance
    importance_df, avg_importance = ensemble.analyze_feature_importance()
    
    # Save the model
    ensemble.save_model('/Users/lukman/Desktop/Ideathon/disease_prediction_ensemble.pkl')
    
    print("\n=== Summary ===")
    print("Best performing models:")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
    for model_name, metrics in sorted_results[:5]:
        print(f"{model_name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    print(f"\nModel saved successfully!")
    print("Feature importance analysis and confusion matrix plots saved.")

if __name__ == "__main__":
    main()
