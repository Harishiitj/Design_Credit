# Design Credit Project - Technical Implementation Guide
## Python-Based ML Pipeline for Environmental Health Analysis

---

## Project Repository Structure

```
Design_Credit/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── LICENSE
│
├── data/
│   ├── raw/
│   │   ├── household_survey.csv
│   │   ├── environmental_measurements.csv
│   │   └── health_indicators.csv
│   ├── processed/
│   │   ├── cleaned_data.csv
│   │   └── data_quality_report.json
│   └── features/
│       └── feature_matrix.csv
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_development.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_results_visualization.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_builder.py
│   ├── evaluation.py
│   └── utils.py
│
├── models/
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── model_metadata.json
│
├── results/
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   ├── predictions/
│   └── visualizations/
│
├── docs/
│   ├── data_dictionary.md
│   ├── methodology.md
│   ├── results_summary.md
│   └── api_documentation.md
│
└── tests/
    ├── test_preprocessing.py
    ├── test_feature_engineering.py
    └── test_models.py
```

---

## Python Environment Setup

### Requirements.txt
```
# Data Processing
pandas==1.5.3
numpy==1.24.3
scipy==1.10.1

# Machine Learning
scikit-learn==1.3.0
xgboost==2.0.0
lightgbm==4.0.0
catboost==1.2.0

# Model Interpretation
shap==0.42.1
lime==0.2.0

# Visualization
matplotlib==3.7.1
seaborn==0.12.2
plotly==5.14.0

# Jupyter & Development
jupyter==1.0.0
ipython==8.12.2
notebook==6.5.4

# Utilities
python-dotenv==1.0.0
tqdm==4.65.0
joblib==1.2.0

# Testing & Quality
pytest==7.3.1
black==23.3.0
flake8==6.0.0
```

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/Design_Credit.git
cd Design_Credit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook
```

---

## Step-by-Step Implementation Code

### Step 1: Data Loading & Initial Exploration
```python
# notebooks/01_exploratory_data_analysis.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load data
households_df = pd.read_csv('data/raw/household_survey.csv')
environment_df = pd.read_csv('data/raw/environmental_measurements.csv')
health_df = pd.read_csv('data/raw/health_indicators.csv')

# Initial exploration
print("Dataset shapes:")
print(f"Households: {households_df.shape}")
print(f"Environment: {environment_df.shape}")
print(f"Health: {health_df.shape}")

# Check data types
print("\nData types:")
print(households_df.dtypes)

# Basic statistics
print("\nBasic statistics:")
print(households_df.describe())

# Missing value analysis
missing_percent = (households_df.isnull().sum() / len(households_df)) * 100
print("\nMissing value percentages:")
print(missing_percent[missing_percent > 0].sort_values(ascending=False))

# Visualize distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
households_df['household_income'].hist(ax=axes[0, 0], bins=50)
households_df['household_size'].hist(ax=axes[0, 1], bins=20)
households_df['age'].hist(ax=axes[1, 0], bins=30)
health_df['health_score'].hist(ax=axes[1, 1], bins=20)
plt.tight_layout()
plt.savefig('results/visualizations/distribution_exploration.png', dpi=300)
plt.show()
```

### Step 2: Data Preprocessing
```python
# src/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer

class DataPreprocessor:
    def __init__(self):
        self.imputer = None
        self.scaler = None
        
    def handle_missing_values(self, df, method='mean', threshold=0.3):
        """
        Handle missing values in dataframe
        
        Parameters:
        - df: DataFrame
        - method: 'mean', 'median', 'knn', or 'drop'
        - threshold: Drop columns if missing > threshold
        """
        # Drop columns with missing > threshold
        df = df.loc[:, (df.isnull().sum() / len(df)) < threshold]
        
        if method == 'mean':
            imputer = SimpleImputer(strategy='mean')
        elif method == 'median':
            imputer = SimpleImputer(strategy='median')
        elif method == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        
        # Store imputer for test set
        self.imputer = imputer
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df),
            columns=df.columns,
            index=df.index
        )
        return df_imputed
    
    def remove_outliers_iqr(self, df, column, iqr_multiplier=1.5):
        """Remove outliers using IQR method"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    def standardize_categorical(self, df):
        """Standardize categorical variables"""
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.lower().str.strip()
        return df
    
    def validate_data(self, df):
        """Run comprehensive data validation"""
        issues = []
        
        # Check logical consistency
        if (df['age'] < 0).any():
            issues.append("Age cannot be negative")
        
        if (df['household_size'] <= 0).any():
            issues.append("Household size must be positive")
        
        # Check for duplicates
        if df.duplicated().sum() > 0:
            issues.append(f"Found {df.duplicated().sum()} duplicate rows")
        
        if issues:
            print("Data Validation Issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✓ All data validation checks passed!")
        
        return df

# Usage
preprocessor = DataPreprocessor()
df_clean = preprocessor.handle_missing_values(df, method='median')
df_clean = preprocessor.standardize_categorical(df_clean)
df_clean = preprocessor.validate_data(df_clean)
df_clean.to_csv('data/processed/cleaned_data.csv', index=False)
```

### Step 3: Feature Engineering
```python
# src/feature_engineering.py

import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        self.feature_metadata = {}
    
    def create_air_quality_index(self, df):
        """
        Create Air Quality Index from pollutant measurements
        Formula: AQI = (PM2.5*0.4 + PM10*0.3 + NO2*0.2 + SO2*0.1) / 100
        """
        df['air_quality_index'] = (
            df['PM2.5'] * 0.4 +
            df['PM10'] * 0.3 +
            df['NO2'] * 0.2 +
            df['SO2'] * 0.1
        ) / 100
        
        self.feature_metadata['air_quality_index'] = {
            'type': 'composite',
            'components': ['PM2.5', 'PM10', 'NO2', 'SO2'],
            'formula': 'Weighted average of pollutants'
        }
        return df
    
    def create_sustainability_index(self, df):
        """Create household sustainability practices index"""
        sustainability_features = [
            'clean_cooking_fuel',
            'waste_management_practice',
            'water_conservation',
            'renewable_energy_use',
            'organic_farming'
        ]
        
        # Normalize features to 0-1 scale
        for feat in sustainability_features:
            if feat in df.columns:
                df[f'{feat}_norm'] = (df[feat] - df[feat].min()) / (df[feat].max() - df[feat].min())
        
        # Calculate composite index
        norm_features = [f'{feat}_norm' for feat in sustainability_features if f'{feat}_norm' in df.columns]
        df['sustainability_index'] = df[norm_features].mean(axis=1)
        
        self.feature_metadata['sustainability_index'] = {
            'type': 'composite',
            'components': sustainability_features,
            'formula': 'Average of normalized sustainability practices'
        }
        return df
    
    def create_temporal_features(self, df, date_column):
        """Extract temporal features from date columns"""
        df[date_column] = pd.to_datetime(df[date_column])
        
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['quarter'] = df[date_column].dt.quarter
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['season'] = df['month'].apply(self._get_season)
        
        self.feature_metadata['temporal_features'] = {
            'type': 'derived',
            'base_column': date_column,
            'features': ['year', 'month', 'quarter', 'day_of_week', 'season']
        }
        return df
    
    @staticmethod
    def _get_season(month):
        """Map month to season"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'summer'
        elif month in [6, 7, 8, 9]:
            return 'monsoon'
        else:
            return 'post_monsoon'
    
    def create_interaction_features(self, df):
        """Create interaction features between key variables"""
        # Income × Education interaction
        if 'household_income' in df.columns and 'education_level' in df.columns:
            df['income_education_interaction'] = df['household_income'] * df['education_level']
        
        # Air Quality × Ventilation interaction
        if 'air_quality_index' in df.columns and 'ventilation_score' in df.columns:
            df['exposure_risk'] = df['air_quality_index'] * (1 - df['ventilation_score'])
        
        self.feature_metadata['interaction_features'] = {
            'type': 'derived',
            'created': ['income_education_interaction', 'exposure_risk']
        }
        return df
    
    def encode_categorical(self, df):
        """One-hot encode categorical variables"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        self.feature_metadata['encoded_categorical'] = {
            'type': 'encoding',
            'original_columns': list(categorical_cols),
            'encoding_method': 'one-hot'
        }
        return df_encoded

# Usage
engineer = FeatureEngineer()
df_engineered = engineer.create_air_quality_index(df_clean)
df_engineered = engineer.create_sustainability_index(df_engineered)
df_engineered = engineer.create_temporal_features(df_engineered, 'survey_date')
df_engineered = engineer.create_interaction_features(df_engineered)
df_engineered = engineer.encode_categorical(df_engineered)

df_engineered.to_csv('data/features/feature_matrix.csv', index=False)
```

### Step 4: Model Development & Training
```python
# src/model_builder.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

class ModelBuilder:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.results = {}
    
    def prepare_data(self, df, target_column, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Train-test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Train-validation split
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state
        )
        
        print(f"Train set size: {X_train.shape[0]} samples")
        print(f"Validation set size: {X_val.shape[0]} samples")
        print(f"Test set size: {X_test.shape[0]} samples")
        
        return (X_train, X_val, X_test, y_train, y_val, y_test)
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train Random Forest model with hyperparameter tuning"""
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Validation performance
        y_pred_val = model.predict(X_val)
        val_r2 = r2_score(y_val, y_pred_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        
        print(f"Random Forest - Validation R²: {val_r2:.4f}, RMSE: {val_rmse:.4f}")
        
        self.models['random_forest'] = model
        self.results['random_forest'] = {
            'val_r2': val_r2,
            'val_rmse': val_rmse
        }
        
        return model
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=0
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Validation performance
        y_pred_val = model.predict(X_val)
        val_r2 = r2_score(y_val, y_pred_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        
        print(f"XGBoost - Validation R²: {val_r2:.4f}, RMSE: {val_rmse:.4f}")
        
        self.models['xgboost'] = model
        self.results['xgboost'] = {
            'val_r2': val_r2,
            'val_rmse': val_rmse
        }
        
        return model
    
    def train_linear_models(self, X_train, y_train, X_val, y_val):
        """Train linear models (Ridge, Lasso)"""
        from sklearn.linear_model import Ridge, Lasso
        
        # Scale features for linear models
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        models_linear = {
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1)
        }
        
        for name, model in models_linear.items():
            model.fit(X_train_scaled, y_train)
            y_pred_val = model.predict(X_val_scaled)
            val_r2 = r2_score(y_val, y_pred_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            
            print(f"{name.capitalize()} - Validation R²: {val_r2:.4f}, RMSE: {val_rmse:.4f}")
            
            self.models[name] = model
            self.results[name] = {'val_r2': val_r2, 'val_rmse': val_rmse}
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models on test set"""
        results_test = {}
        
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            results_test[model_name] = {
                'R2': r2,
                'RMSE': rmse,
                'MAE': mae
            }
            
            print(f"\n{model_name.upper()} Test Performance:")
            print(f"  R²: {r2:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
        
        return results_test
    
    def save_models(self, path='models/'):
        """Save trained models"""
        for model_name, model in self.models.items():
            joblib.dump(model, f'{path}{model_name}_model.pkl')
            print(f"Saved {model_name} model to {path}{model_name}_model.pkl")

# Usage
builder = ModelBuilder()
X_train, X_val, X_test, y_train, y_val, y_test = builder.prepare_data(
    df_engineered, target_column='health_score'
)

builder.train_random_forest(X_train, y_train, X_val, y_val)
builder.train_xgboost(X_train, y_train, X_val, y_val)
builder.train_linear_models(X_train, y_train, X_val, y_val)

test_results = builder.evaluate_models(X_test, y_test)
builder.save_models()
```

### Step 5: Model Evaluation & Interpretation
```python
# notebooks/05_model_evaluation.ipynb

import shap
import matplotlib.pyplot as plt
import numpy as np

# Load best model
best_model = joblib.load('models/xgboost_model.pkl')

# SHAP Interpretation
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Feature importance summary
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('results/visualizations/feature_importance_shap.png', dpi=300)
plt.show()

# SHAP dependence plot for top feature
top_feature = shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
shap.dependence_plot("air_quality_index", shap_values, X_test)
plt.savefig('results/visualizations/shap_dependence_air_quality.png', dpi=300)
plt.show()

# Residual analysis
y_pred = best_model.predict(X_test)
residuals = y_test - y_pred

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.subplot(1, 2, 2)
plt.hist(residuals, bins=30, edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.tight_layout()
plt.savefig('results/visualizations/residual_analysis.png', dpi=300)
plt.show()
```

---

## Model Deployment

### Creating a Prediction API
```python
# src/api.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('models/xgboost_model.pkl')
feature_names = ['air_quality_index', 'sustainability_index', ...]  # All features

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.get_json()
        
        # Create DataFrame with correct feature order
        input_df = pd.DataFrame([data], columns=feature_names)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        confidence = float(np.exp(prediction) / (1 + np.exp(prediction)))  # If binary
        
        return jsonify({
            'prediction': float(prediction),
            'health_score': float(prediction),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

if __name__ == '__main__':
    app.run(debug=False, port=5000)
```

### Usage Example
```bash
# Start server
python src/api.py

# Make prediction request
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "air_quality_index": 120,
    "sustainability_index": 0.7,
    "household_income": 50000,
    "household_size": 4
  }'
```

---

## Testing

### Unit Tests
```python
# tests/test_preprocessing.py

import unittest
import pandas as pd
from src.preprocessing import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = DataPreprocessor()
        self.test_df = pd.DataFrame({
            'age': [25, 30, np.nan, 35],
            'income': [50000, 60000, 55000, np.nan],
            'health_score': [7, 8, 6, 9]
        })
    
    def test_missing_value_handling(self):
        df_cleaned = self.preprocessor.handle_missing_values(
            self.test_df, method='mean'
        )
        self.assertEqual(df_cleaned.isnull().sum().sum(), 0)
    
    def test_outlier_removal(self):
        df = pd.DataFrame({'values': [1, 2, 3, 100]})  # 100 is outlier
        df_cleaned = self.preprocessor.remove_outliers_iqr(df, 'values')
        self.assertNotIn(100, df_cleaned['values'].values)

if __name__ == '__main__':
    unittest.main()
```

---

## Key Performance Metrics Reference

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| R² Score | 1 - (SS_res/SS_tot) | % variance explained (0-1) |
| RMSE | √(MSE) | Average magnitude of error |
| MAE | (1/n)Σ\|y_i - ŷ_i\| | Mean absolute error (interpretable) |
| MAPE | (1/n)Σ\|residual/y_i\|×100 | Percentage error |
| AUC-ROC | Area under ROC curve | Classification discrimination (0-1) |
| F1-Score | 2×(Precision×Recall)/(Precision+Recall) | Balance of precision/recall |

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Data leakage | Preprocess before splitting; don't fit scaler on full dataset |
| Overfitting | Use regularization, early stopping, cross-validation |
| Class imbalance | Use weighted loss, stratified sampling, SMOTE |
| Missing values | Don't just drop; understand MCAR/MAR/MNAR |
| Feature scaling | Scale for distance-based; not for tree models |
| Model comparison | Use cross-validation; don't just test set |

---

## Git Workflow

```bash
# Initialize repository
git init
git add .
git commit -m "Initial commit: project structure"

# Feature branches
git checkout -b feature/data-preprocessing
# ... make changes ...
git add src/preprocessing.py
git commit -m "Add data preprocessing module"
git push origin feature/data-preprocessing

# Create pull request and merge
git checkout main
git merge feature/data-preprocessing

# Version tagging
git tag -a v0.1.0 -m "Initial model development"
git push origin v0.1.0
```

---

## Resources & Documentation

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Pandas User Guide](https://pandas.pydata.org/docs/)
- [Jupyter Notebook Documentation](https://jupyter.org/)

---

*This technical guide should be used alongside the main Action Plan document.*
*Keep this guide updated as your implementation progresses.*
