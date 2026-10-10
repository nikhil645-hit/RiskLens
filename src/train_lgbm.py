import pandas as pd
import numpy as np
import time
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

def main():
    print("Loading data splits...")
    X_train = pd.read_pickle("data/processed/X_train.pkl")
    y_train = pd.read_pickle("data/processed/y_train.pkl")
    X_val = pd.read_pickle("data/processed/X_val.pkl")
    y_val = pd.read_pickle("data/processed/y_val.pkl")
    
    # Ensure categorical columns are of type 'category' for LightGBM
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        X_train[col] = X_train[col].astype('category')
        X_val[col] = X_val[col].astype('category')
        
    print("Training LightGBM model...")
    lgbm = LGBMClassifier(class_weight="balanced", random_state=42)
    
    start_time = time.time()
    lgbm.fit(X_train, y_train)
    fit_time = time.time() - start_time
    
    print("Evaluating on Validation set...")
    y_pred_proba = lgbm.predict_proba(X_val)[:, 1]
    
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    pr_auc = average_precision_score(y_val, y_pred_proba)
    
    print(f"\n--- LightGBM Benchmark Results ---")
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}")
    print(f"Time    : {fit_time:.1f}s")
    
    print("\n--- Comparison with Logistic Regression (One-Hot) ---")
    print("LogReg ROC-AUC : 0.7477")
    print("LogReg PR-AUC  : 0.2282")
    
    diff_roc = roc_auc - 0.7477
    print(f"\nLightGBM difference in ROC-AUC: {diff_roc:+.4f}")
    if diff_roc > 0.01:
        print("Verdict: LightGBM is vastly better.")
    else:
        print("Verdict: LightGBM provides only marginal or no improvement.")

if __name__ == "__main__":
    main()
