import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from src.features import build_pipeline

sns.set_theme(style="whitegrid")
output_dir = "reports/figures"
os.makedirs(output_dir, exist_ok=True)

def get_top_3_reasons(pipeline, X_applicant, feature_names):
    """
    Returns the top 3 reasons raising the risk for a single applicant
    by multiplying the model coefficients with the applicant's transformed values.
    """
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']
    
    # Transform applicant
    X_transformed = preprocessor.transform(X_applicant)
    
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
        
    # Coefficients
    coef = classifier.coef_[0]
    
    # Calculate contribution: coefficient * value
    contributions = coef * X_transformed[0]
    
    # Sort indices by highest contribution
    top_indices = np.argsort(contributions)[::-1][:3]
    
    reasons = []
    for idx in top_indices:
        feat = feature_names[idx]
        cont = contributions[idx]
        val = X_transformed[0][idx]
        reasons.append({"feature": feat, "contribution": float(cont), "value": float(val)})
        
    return reasons

def main():
    print("Loading data splits...")
    X_train = pd.read_pickle("data/processed/X_train.pkl")
    y_train = pd.read_pickle("data/processed/y_train.pkl")
    X_val = pd.read_pickle("data/processed/X_val.pkl")
    y_val = pd.read_pickle("data/processed/y_val.pkl")
    
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_train.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    print("Building and training pipeline...")
    pipeline = build_pipeline("onehot", cat_cols, num_cols)
    pipeline.fit(X_train, y_train)
    
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']
    
    # Get feature names
    feature_names = preprocessor.get_feature_names_out()
    # Clean up feature names (e.g. 'num__', 'cat__')
    feature_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]
    
    coef = classifier.coef_[0]
    odds_ratios = np.exp(coef)
    
    # Create DataFrame for analysis
    df_coef = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coef,
        "odds_ratio": odds_ratios
    })
    
    # Top 10 raising risk (highest odds ratio)
    top_10_raising = df_coef.sort_values(by="odds_ratio", ascending=False).head(10)
    # Top 10 lowering risk (lowest odds ratio)
    top_10_lowering = df_coef.sort_values(by="odds_ratio", ascending=True).head(10)
    
    # Save to JSON
    odds_data = {
        "raising_risk": top_10_raising.to_dict(orient="records"),
        "lowering_risk": top_10_lowering.to_dict(orient="records")
    }
    with open("reports/odds_ratios.json", "w") as f:
        json.dump(odds_data, f, indent=4)
        
    # Plot Odds Ratios
    combined_top = pd.concat([top_10_raising, top_10_lowering]).sort_values(by="odds_ratio", ascending=True)
    
    plt.figure(figsize=(10, 8))
    # Corrected sns.barplot for palette with hue
    sns.barplot(x="odds_ratio", y="feature", data=combined_top, 
                hue="feature", palette=['green' if x < 1 else 'red' for x in combined_top["odds_ratio"]], legend=False)
    plt.axvline(1.0, color='black', linestyle='--')
    plt.title("Top 10 Risk Raising vs Risk Lowering Features (Odds Ratios)")
    plt.xlabel("Odds Ratio (>1 increases risk, <1 decreases risk)")
    plt.savefig(f"{output_dir}/09_odds_ratios.png", bbox_inches='tight')
    plt.close()
    
    print("Saved odds ratios to reports/odds_ratios.json and reports/figures/09_odds_ratios.png")
    
    # SHAP Analysis
    print("Running SHAP on a sample of 500 validation rows...")
    X_val_sample = X_val.sample(500, random_state=42)
    X_val_sample_transformed = preprocessor.transform(X_val_sample)
    
    if hasattr(X_val_sample_transformed, "toarray"):
        X_val_sample_transformed = X_val_sample_transformed.toarray()
        
    explainer = shap.LinearExplainer(classifier, X_val_sample_transformed)
    shap_values = explainer.shap_values(X_val_sample_transformed)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_val_sample_transformed, feature_names=feature_names, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/10_shap_summary.png", bbox_inches='tight')
    plt.close()
    
    # Plain Language Explanation for One Applicant
    print("\n--- Example Plain Language Explanation ---")
    applicant = X_val.iloc[[42]] # Just pick a random one
    prob = pipeline.predict_proba(applicant)[0, 1]
    
    top_3 = get_top_3_reasons(pipeline, applicant, feature_names)
    
    print(f"Applicant Risk Probability: {prob*100:.1f}%")
    print("Top 3 reasons driving this risk score up:")
    for i, r in enumerate(top_3, 1):
        print(f"  {i}. {r['feature']} (Contribution: {r['contribution']:.3f}, Transformed Value: {r['value']:.2f})")
        
    print("\nIn plain language:")
    print(f"This applicant was flagged with a {prob*100:.1f}% probability of default.")
    print(f"The primary reason was '{top_3[0]['feature']}'. The model learned that this specific trait")
    print(f"is highly associated with previous defaults, so it heavily increased their risk score.")

if __name__ == "__main__":
    main()
