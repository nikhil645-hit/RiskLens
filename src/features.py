import pandas as pd
import numpy as np
import time
import json
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from category_encoders import LeaveOneOutEncoder

def build_pipeline(encoder_name, cat_cols, num_cols):
    """
    Builds a sklearn Pipeline with specified categorical encoder,
    imputation, and Logistic Regression.
    """
    # Numeric preprocessing: Median imputation -> Scaling
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing: 'Missing' imputation -> Encoding
    if encoder_name == "onehot":
        # Using min_frequency to avoid explosive feature growth
        encoder = OneHotEncoder(min_frequency=50, handle_unknown="ignore", sparse_output=False)
    elif encoder_name == "target":
        encoder = TargetEncoder(target_type="binary", cv=5)
    elif encoder_name == "loo":
        encoder = LeaveOneOutEncoder(sigma=0.05)
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")
        
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('encoder', encoder)
    ])
    
    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ],
        remainder='drop' # Just in case
    )
    
    # Full pipeline with Logistic Regression (class_weight="balanced" is critical!)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    ])
    
    return pipeline

if __name__ == "__main__":
    print("Loading data splits...")
    X_train = pd.read_pickle("data/processed/X_train.pkl")
    y_train = pd.read_pickle("data/processed/y_train.pkl")
    X_val = pd.read_pickle("data/processed/X_val.pkl")
    y_val = pd.read_pickle("data/processed/y_val.pkl")
    
    # Identify column types
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_train.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    results = []
    
    for enc_name in ["onehot", "target", "loo"]:
        print(f"\nTraining pipeline with '{enc_name}' encoder...")
        pipeline = build_pipeline(enc_name, cat_cols, num_cols)
        
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        fit_time = time.time() - start_time
        
        # Count features output by preprocessor
        X_sample_transformed = pipeline.named_steps['preprocessor'].transform(X_train.iloc[:5])
        num_features = X_sample_transformed.shape[1]
        
        # Predict on validation (using probabilities)
        y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
        
        # Calculate scores
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        pr_auc = average_precision_score(y_val, y_pred_proba)
        
        res_dict = {
            "encoder": enc_name,
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "num_features": int(num_features),
            "fit_time_seconds": round(fit_time, 2)
        }
        results.append(res_dict)
        
        print(f"  ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Features: {num_features} | Time: {fit_time:.1f}s")
        
    with open("reports/encoder_comparison.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to reports/encoder_comparison.json")
