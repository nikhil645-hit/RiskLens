import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score,
                             confusion_matrix, precision_score, recall_score, f1_score)
from sklearn.calibration import calibration_curve
from src.features import build_pipeline
import os

sns.set_theme(style="whitegrid")
output_dir = "reports/figures"
os.makedirs(output_dir, exist_ok=True)

def main():
    print("Loading data splits...")
    X_train = pd.read_pickle("data/processed/X_train.pkl")
    y_train = pd.read_pickle("data/processed/y_train.pkl")
    X_val = pd.read_pickle("data/processed/X_val.pkl")
    y_val = pd.read_pickle("data/processed/y_val.pkl")
    
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_train.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    print("Building and training One-Hot Logistic Regression model...")
    pipeline = build_pipeline("onehot", cat_cols, num_cols)
    pipeline.fit(X_train, y_train)
    
    print("Generating predictions on Validation set...")
    y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
    
    # 1. Save Validation Predictions
    val_preds = pd.DataFrame({
        "true_label": y_val,
        "predicted_probability": y_pred_proba
    })
    val_preds.to_csv("reports/validation_predictions.csv", index=False)
    
    # 2. ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(f"{output_dir}/05_roc_curve.png", bbox_inches='tight')
    plt.close()
    
    roc_points = [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr, tpr)][::50] # Downsample for frontend
    with open("reports/roc_points.json", "w") as f:
        json.dump(roc_points, f)
        
    # 3. PR Curve
    precision, recall, pr_thresholds = precision_recall_curve(y_val, y_pred_proba)
    pr_auc = average_precision_score(y_val, y_pred_proba)
    
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (area = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.savefig(f"{output_dir}/06_pr_curve.png", bbox_inches='tight')
    plt.close()
    
    pr_points = [{"recall": float(r), "precision": float(p)} for r, p in zip(recall, precision)][::50]
    with open("reports/pr_points.json", "w") as f:
        json.dump(pr_points, f)
        
    # 4. Calibration Curve
    prob_true, prob_pred = calibration_curve(y_val, y_pred_proba, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.savefig(f"{output_dir}/07_calibration_curve.png", bbox_inches='tight')
    plt.close()
    
    # 5. Threshold search based on Business Cost
    thresholds = np.linspace(0.01, 0.99, 99)
    costs_5_1 = []
    costs_3_1 = []
    costs_10_1 = []
    
    precisions = []
    recalls = []
    f1s = []
    
    best_t_5_1 = 0
    min_cost_5_1 = float('inf')
    best_t_3_1 = 0
    min_cost_3_1 = float('inf')
    best_t_10_1 = 0
    min_cost_10_1 = float('inf')
    
    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        
        # Calculate costs
        cost51 = (fn * 5) + (fp * 1)
        cost31 = (fn * 3) + (fp * 1)
        cost101 = (fn * 10) + (fp * 1)
        
        if cost51 < min_cost_5_1:
            min_cost_5_1 = cost51
            best_t_5_1 = t
            
        if cost31 < min_cost_3_1:
            min_cost_3_1 = cost31
            best_t_3_1 = t
            
        if cost101 < min_cost_10_1:
            min_cost_10_1 = cost101
            best_t_10_1 = t
            
        # For the F1 vs Threshold chart
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1s, label='F1 Score')
    plt.axvline(best_t_5_1, color='red', linestyle='--', label=f'Best 5:1 Threshold ({best_t_5_1:.2f})')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 vs Decision Threshold')
    plt.legend()
    plt.savefig(f"{output_dir}/08_metrics_vs_threshold.png", bbox_inches='tight')
    plt.close()
    
    print("\n--- Business Cost Search Results ---")
    print(f"Cost Ratio 3:1  -> Best Threshold: {best_t_3_1:.2f}")
    print(f"Cost Ratio 5:1  -> Best Threshold: {best_t_5_1:.2f} (RECOMMENDED)")
    print(f"Cost Ratio 10:1 -> Best Threshold: {best_t_10_1:.2f}")
    
    # Calculate final metrics at 5:1 threshold
    y_pred_final = (y_pred_proba >= best_t_5_1).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred_final).ravel()
    
    print(f"\n--- Final Confusion Matrix at Threshold {best_t_5_1:.2f} ---")
    print(f"True Negatives (Correctly Approved): {tn}")
    print(f"False Positives (Wrongly Rejected) : {fp}")
    print(f"False Negatives (Wrongly Approved) : {fn}")
    print(f"True Positives (Correctly Rejected): {tp}")
    print(f"Precision: {precision_score(y_val, y_pred_final):.3f}")
    print(f"Recall: {recall_score(y_val, y_pred_final):.3f}")
    print(f"F1 Score: {f1_score(y_val, y_pred_final):.3f}")
    
if __name__ == "__main__":
    main()
