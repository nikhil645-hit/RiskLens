import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.metrics import roc_auc_score, average_precision_score
from src.features import build_pipeline

os.makedirs("models", exist_ok=True)

def main():
    print("Loading data splits...")
    X_train = pd.read_pickle("data/processed/X_train.pkl")
    y_train = pd.read_pickle("data/processed/y_train.pkl")
    X_val = pd.read_pickle("data/processed/X_val.pkl")
    y_val = pd.read_pickle("data/processed/y_val.pkl")
    X_test = pd.read_pickle("data/processed/X_test.pkl")
    y_test = pd.read_pickle("data/processed/y_test.pkl")
    
    print("Combining Train and Validation sets...")
    X_full = pd.concat([X_train, X_val], ignore_index=True)
    y_full = pd.concat([y_train, y_val], ignore_index=True)
    
    cat_cols = X_full.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_full.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    print(f"Training Final Pipeline on {len(X_full)} rows...")
    pipeline = build_pipeline("onehot", cat_cols, num_cols)
    pipeline.fit(X_full, y_full)
    
    print("Evaluating on UNTOUCHED Test Set...")
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    print(f"\n--- Final Test Set Performance ---")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")
    
    print("\nSaving model to models/model.joblib...")
    joblib.dump(pipeline, "models/model.joblib")
    
    print("Saving sample request to models/sample_request.json...")
    # Extract one row, replace NaNs with None for valid JSON
    sample_series = X_test.iloc[0].replace({np.nan: None})
    
    # Convert numpy types to native python types
    sample_dict = {}
    for k, v in sample_series.items():
        if pd.isna(v) or v is None:
            sample_dict[k] = None
        elif isinstance(v, (np.int64, np.int32)):
            sample_dict[k] = int(v)
        elif isinstance(v, (np.float64, np.float32)):
            sample_dict[k] = float(v)
        else:
            sample_dict[k] = str(v)
            
    with open("models/sample_request.json", "w") as f:
        json.dump(sample_dict, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
