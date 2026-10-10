import sys
import os

# Add root project dir to pythonpath so src module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json

app = FastAPI(title="RiskLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for caching model and feature names
pipeline = None
feature_names = None

@app.on_event("startup")
def load_model():
    global pipeline
    global feature_names
    
    # Load model from disk exactly once on startup
    print("Loading model.joblib...")
    pipeline = joblib.load("models/model.joblib")
    
    # Extract feature names from preprocessor for SHAP/Explanation
    preprocessor = pipeline.named_steps['preprocessor']
    names = preprocessor.get_feature_names_out()
    feature_names = [f.replace("num__", "").replace("cat__", "") for f in names]
    print("Model loaded successfully!")

def get_top_3_reasons(pipeline, X_applicant_df, feature_names):
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']
    
    # Transform
    X_transformed = preprocessor.transform(X_applicant_df)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
        
    # Get contributions
    coef = classifier.coef_[0]
    contributions = coef * X_transformed[0]
    
    # Get top 3 indices that push risk UP (highest positive contributions)
    top_indices = np.argsort(contributions)[::-1][:3]
    
    reasons = []
    for idx in top_indices:
        feat = feature_names[idx]
        cont = contributions[idx]
        val = X_transformed[0][idx]
        reasons.append({"feature": feat, "contribution": float(cont), "value": float(val)})
        
    return reasons

@app.post("/predict")
async def predict(applicant: dict):
    # Convert arbitrary dict to a DataFrame with 1 row
    df = pd.DataFrame([applicant])
    
    # 1. Calculate the raw probability (Risk Score)
    prob = pipeline.predict_proba(df)[0, 1]
    
    # 2. Apply our optimized business threshold (0.67)
    threshold = 0.67
    is_rejected = prob >= threshold
    status = "Rejected" if is_rejected else "Approved"
    
    # 3. Generate the customized Top 3 Reasons
    top_3 = get_top_3_reasons(pipeline, df, feature_names)
    
    return {
        "probability_of_default": float(prob),
        "status": status,
        "threshold_used": threshold,
        "top_3_reasons": top_3
    }

@app.get("/explain")
async def explain():
    # Return the global odds ratios for the frontend charts
    try:
        with open("reports/odds_ratios.json", "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        return {"error": str(e)}
