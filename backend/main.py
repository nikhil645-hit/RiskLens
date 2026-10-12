import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import calibrated pipeline module
import calibrated_pipeline

app = FastAPI(
    title="RiskLens Credit Risk Assessment API",
    description="Underwriting risk scoring engine powered by calibrated Logistic Regression.",
    version="1.0.0",
)

# CORS configuration via FRONTEND_ORIGIN environment variable
frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
origins = [o.strip() for o in frontend_origin.split(",") if o.strip()]
for default_origin in ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000", "http://127.0.0.1:8000"]:
    if default_origin not in origins:
        origins.append(default_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global bundle state
bundle: Dict[str, Any] = {}
pipeline = None
threshold: float = 0.16
base_row: Dict[str, Any] = {}
blank_template: pd.DataFrame = pd.DataFrame()
feature_names_out: List[str] = []
coefs: np.ndarray = np.array([])
preprocessor = None

# Mapping between friendly field names and Home Credit column names
FRIENDLY_TO_COL = {
    "annual_income": "AMT_INCOME_TOTAL",
    "loan_amount": "AMT_CREDIT",
    "annuity": "AMT_ANNUITY",
    "education": "NAME_EDUCATION_TYPE",
    "income_type": "NAME_INCOME_TYPE",
    "occupation": "OCCUPATION_TYPE",
    "organisation_type": "ORGANIZATION_TYPE",
    "organization_type": "ORGANIZATION_TYPE",
    "ext_source_1": "EXT_SOURCE_1",
    "ext_source_2": "EXT_SOURCE_2",
    "ext_source_3": "EXT_SOURCE_3",
}

COL_TO_FRIENDLY = {v.lower(): k for k, v in FRIENDLY_TO_COL.items()}
COL_TO_FRIENDLY["days_birth"] = "age"
COL_TO_FRIENDLY["days_employed"] = "years_employed"


def build_blank_template(pipe) -> pd.DataFrame:
    """Create a 1-row blank DataFrame with the exact column names and dtypes expected by the model."""
    names = list(pipe.feature_names_in_)
    dtypes = {}
    train_csv = ROOT_DIR / "data" / "raw" / "application_train.csv"
    try:
        ref = pd.read_csv(train_csv, nrows=1000)
        dtypes = {c: ref[c].dtype for c in names if c in ref.columns}
    except Exception:
        pass
    data = {}
    for c in names:
        dt = dtypes.get(c)
        numeric = dt is not None and pd.api.types.is_numeric_dtype(dt)
        data[c] = pd.Series([np.nan], dtype="float64" if numeric else object)
    return pd.DataFrame(data)[names]


def load_artifacts():
    global bundle, pipeline, threshold, base_row, blank_template, feature_names_out, coefs, preprocessor
    model_path = ROOT_DIR / "models" / "risk_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    threshold = float(bundle.get("threshold", 0.16))

    # Load base sample request row
    sample_path = ROOT_DIR / "models" / "sample_request.json"
    if sample_path.exists():
        base_row = json.loads(sample_path.read_text(encoding="utf-8"))
    else:
        frontend_sample = ROOT_DIR / "frontend" / "sample_request.json"
        if frontend_sample.exists():
            base_row = json.loads(frontend_sample.read_text(encoding="utf-8"))

    # Build blank template for exact dtype alignment
    blank_template = build_blank_template(pipeline)

    # Extract underlying sklearn pipeline preprocessor and classifier weights
    raw_pipe = getattr(pipeline, "pipeline", pipeline)
    if hasattr(raw_pipe, "named_steps"):
        preprocessor = raw_pipe.named_steps.get("preprocessor")
        classifier = raw_pipe.named_steps.get("classifier")
        if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
            feature_names_out = list(preprocessor.get_feature_names_out())
        if classifier is not None and hasattr(classifier, "coef_"):
            coefs = classifier.coef_[0]

    print(f"RiskLens API loaded bundle: threshold={threshold}, features={len(feature_names_out)}")


# Load artifacts eagerly on module import
try:
    load_artifacts()
except Exception as e:
    print(f"Warning: load_artifacts() on import raised: {e}")


@app.on_event("startup")
def startup_event():
    if pipeline is None:
        load_artifacts()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": bundle.get("model_type", "LogReg"),
        "calibrated": bool(bundle.get("calibrated", True)),
    }


@app.get("/schema")
def schema():
    return {
        "fields": {
            "annual_income": {
                "type": "float",
                "min": 1000,
                "max": 100000000,
                "default": 147000,
                "label": "Annual Income",
            },
            "loan_amount": {
                "type": "float",
                "min": 1000,
                "max": 100000000,
                "default": 500000,
                "label": "Loan Amount",
            },
            "annuity": {
                "type": "float",
                "min": 100,
                "max": 10000000,
                "default": 25000,
                "label": "Monthly Annuity",
            },
            "age": {
                "type": "float",
                "min": 18,
                "max": 100,
                "default": 43,
                "label": "Age",
            },
            "years_employed": {
                "type": "float",
                "min": 0,
                "max": 70,
                "default": 6,
                "label": "Years Employed",
            },
            "ext_source_1": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5,
                "label": "External Source 1",
            },
            "ext_source_2": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5,
                "label": "External Source 2",
            },
            "ext_source_3": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5,
                "label": "External Source 3",
            },
            "education": {
                "type": "string",
                "options": [
                    "Secondary / secondary special",
                    "Higher education",
                    "Incomplete higher",
                    "Lower secondary",
                    "Academic degree",
                ],
                "default": "Secondary / secondary special",
            },
            "income_type": {
                "type": "string",
                "options": [
                    "Working",
                    "Commercial associate",
                    "Pensioner",
                    "State servant",
                    "Businessman",
                    "Student",
                    "Unemployed",
                ],
                "default": "Working",
            },
            "occupation": {
                "type": "string",
                "options": [
                    "Laborers",
                    "Sales staff",
                    "Core staff",
                    "Managers",
                    "Drivers",
                    "High skill tech staff",
                    "Accountants",
                    "Medicine staff",
                    "Security staff",
                    "Cooking staff",
                    "Cleaning staff",
                    "Private service staff",
                    "Low-skill Laborers",
                    "Waiters/barmen staff",
                    "Secretaries",
                    "Realty agents",
                    "HR staff",
                    "IT staff",
                ],
                "default": "Laborers",
            },
            "organisation_type": {
                "type": "string",
                "options": [
                    "Business Entity Type 3",
                    "Self-employed",
                    "Other",
                    "Medicine",
                    "Government",
                    "School",
                    "Trade: type 7",
                    "Construction",
                    "Kindergarten",
                    "Business Entity Type 2",
                    "Bank",
                ],
                "default": "Business Entity Type 3",
            },
        }
    }


@app.get("/threshold")
def get_threshold():
    threshold_path = ROOT_DIR / "reports" / "threshold.json"
    if threshold_path.exists():
        try:
            return json.loads(threshold_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "model": bundle.get("model_type", "LogReg"),
        "calibrated": True,
        "chosen_threshold": threshold,
        "chosen_cost_ratio": "5:1",
    }


@app.get("/insights")
def get_insights():
    odds_path = ROOT_DIR / "reports" / "odds_ratios.json"
    if odds_path.exists():
        try:
            return json.loads(odds_path.read_text(encoding="utf-8"))
        except Exception as e:
            return {"error": str(e)}
    return {"error": "odds_ratios.json not found"}


def validate_payload(applicant: Dict[str, Any]) -> tuple[List[str], List[str]]:
    """Strict validation for input ranges returning (errors, warnings)."""
    errors: List[str] = []
    warnings: List[str] = []

    # Annual Income check
    inc = applicant.get("annual_income") if "annual_income" in applicant else applicant.get("AMT_INCOME_TOTAL")
    if inc is not None:
        if isinstance(inc, bool):
            errors.append("annual_income must be a number")
        else:
            try:
                inc_f = float(inc)
                if np.isnan(inc_f) or np.isinf(inc_f) or inc_f > 1e12:
                    errors.append("annual_income is not a valid number")
                elif inc_f < 0:
                    errors.append("annual_income cannot be negative")
                elif inc_f < 30000 or inc_f > 2000000:
                    warnings.append(f"annual_income ({inc_f:,.0f}) is outside typical range ($30k - $2M)")
            except (ValueError, TypeError):
                errors.append("annual_income must be a valid number")

    # Loan Amount check
    loan = applicant.get("loan_amount") if "loan_amount" in applicant else applicant.get("AMT_CREDIT")
    if loan is not None:
        if isinstance(loan, bool):
            errors.append("loan_amount must be a number")
        else:
            try:
                loan_f = float(loan)
                if np.isnan(loan_f) or np.isinf(loan_f) or loan_f > 1e12:
                    errors.append("loan_amount is not a valid number")
                elif loan_f <= 0:
                    errors.append("loan_amount must be positive")
                elif loan_f < 20000 or loan_f > 3000000:
                    warnings.append(f"loan_amount ({loan_f:,.0f}) is outside typical range ($20k - $3M)")
            except (ValueError, TypeError):
                errors.append("loan_amount must be a valid number")

    # Annuity check
    ann = applicant.get("annuity") if "annuity" in applicant else applicant.get("AMT_ANNUITY")
    if ann is not None:
        if isinstance(ann, bool):
            errors.append("annuity must be a number")
        else:
            try:
                ann_f = float(ann)
                if np.isnan(ann_f) or np.isinf(ann_f) or ann_f > 1e12:
                    errors.append("annuity is not a valid number")
                elif ann_f <= 0:
                    errors.append("annuity must be positive")
            except (ValueError, TypeError):
                errors.append("annuity must be a valid number")

    # Age check (check both 'age' and 'DAYS_BIRTH')
    age = applicant.get("age")
    if age is None and "DAYS_BIRTH" in applicant and applicant["DAYS_BIRTH"] is not None:
        try:
            days_birth = float(applicant["DAYS_BIRTH"])
            age = -days_birth / 365.0
        except Exception:
            errors.append("DAYS_BIRTH must be a valid number")

    if age is not None:
        if isinstance(age, bool):
            errors.append("age must be a number")
        else:
            try:
                age_f = float(age)
                if np.isnan(age_f) or np.isinf(age_f):
                    errors.append("age is not a valid number")
                elif age_f < 18 or age_f > 120:
                    errors.append(f"age must be between 18 and 120 (got {age_f})")
            except (ValueError, TypeError):
                errors.append("age must be a valid number")

    # Years Employed check (check both 'years_employed' and 'DAYS_EMPLOYED')
    y_emp = applicant.get("years_employed")
    if y_emp is None and "DAYS_EMPLOYED" in applicant and applicant["DAYS_EMPLOYED"] is not None:
        try:
            days_emp = float(applicant["DAYS_EMPLOYED"])
            y_emp = -days_emp / 365.0
        except Exception:
            errors.append("DAYS_EMPLOYED must be a valid number")

    if y_emp is not None:
        if isinstance(y_emp, bool):
            errors.append("years_employed must be a number")
        else:
            try:
                y_emp_f = float(y_emp)
                if np.isnan(y_emp_f) or np.isinf(y_emp_f):
                    errors.append("years_employed is not a valid number")
                elif y_emp_f < 0:
                    errors.append("years_employed cannot be negative")
                elif age is not None:
                    try:
                        if y_emp_f > (float(age) - 14):
                            warnings.append(f"years_employed ({y_emp_f}) exceeds possible working years for age {age}")
                    except Exception:
                        pass
            except (ValueError, TypeError):
                errors.append("years_employed must be a valid number")

    # External source scores check
    for ext_key in ["ext_source_1", "ext_source_2", "ext_source_3"]:
        ext_val = applicant.get(ext_key) if ext_key in applicant else applicant.get(ext_key.upper())
        if ext_val is not None:
            if isinstance(ext_val, bool):
                errors.append(f"{ext_key} must be a number")
            else:
                try:
                    ext_f = float(ext_val)
                    if np.isnan(ext_f) or np.isinf(ext_f):
                        errors.append(f"{ext_key} is not a valid number")
                    elif ext_f < 0.0 or ext_f > 1.0:
                        errors.append(f"{ext_key} must be between 0.0 and 1.0 (got {ext_f})")
                except (ValueError, TypeError):
                    errors.append(f"{ext_key} must be a valid number")

    return errors, warnings


FORM_FIELDS = {
    "annual_income", "amt_income_total",
    "loan_amount", "amt_credit", "amt_goods_price",
    "annuity", "amt_annuity",
    "age", "days_birth",
    "years_employed", "days_employed",
    "education", "name_education_type",
    "income_type", "name_income_type",
    "occupation", "occupation_type",
    "organisation_type", "organization_type",
    "ext_source_1",
    "ext_source_2",
    "ext_source_3",
}


def compute_reasons(df_row: pd.DataFrame, applicant_input: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compute odds multipliers only for user-entered fields, capped at 3."""
    if preprocessor is None or len(coefs) == 0:
        return []

    try:
        X_trans = preprocessor.transform(df_row)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        contributions = coefs * X_trans[0]
    except Exception:
        return []

    # Only include keys that the user provided AND belong to form fields
    # If a full dataset row (>= 50 cols) was passed, only consider keys that differ from base_row
    if len(applicant_input) >= 50:
        user_keys_lower = {}
        for k, v in applicant_input.items():
            if v is not None and k.lower() in FORM_FIELDS:
                if k not in base_row or v != base_row[k]:
                    user_keys_lower[k.lower()] = k
    else:
        user_keys_lower = {k.lower(): k for k, v in applicant_input.items() if v is not None and k.lower() in FORM_FIELDS}
    if not user_keys_lower:
        return []

    feature_contribs: Dict[str, float] = {}

    for idx, f_out in enumerate(feature_names_out):
        clean = f_out.replace("num__", "").replace("cat__", "")
        # Find which user key this corresponds to
        matched_user_key = None
        for u_k_lower, orig_key in user_keys_lower.items():
            col_target = FRIENDLY_TO_COL.get(u_k_lower, u_k_lower).lower()
            if clean.lower().startswith(col_target) or clean.lower() == col_target or u_k_lower == clean.lower():
                matched_user_key = orig_key
                break

        if matched_user_key:
            c = float(contributions[idx])
            # Keep maximum absolute contribution for this feature
            if matched_user_key not in feature_contribs or abs(c) > abs(feature_contribs[matched_user_key]):
                feature_contribs[matched_user_key] = c

    # Sort by absolute contribution descending
    sorted_features = sorted(feature_contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top_3 = sorted_features[:3]

    reasons: List[Dict[str, Any]] = []
    for feat_name, c in top_3:
        if np.isnan(c) or np.isinf(c):
            continue
        c_clipped = float(np.clip(c, -30.0, 30.0))
        mult = float(np.exp(c_clipped))
        mult_rounded = round(mult, 2)
        if mult_rounded <= 0:
            mult_rounded = 0.01
        reasons.append({
            "feature": feat_name,
            "odds_multiplier": mult_rounded,
            "direction": "raises_risk" if mult > 1.0 else "lowers_risk",
            "contribution": round(c_clipped, 4),
        })

    return reasons


@app.post("/predict")
async def predict(applicant: Dict[str, Any]):
    # Validate payload
    errors, warnings = validate_payload(applicant)
    if errors:
        raise HTTPException(status_code=422, detail="; ".join(errors))

    # Compute DTI informational field
    inc = applicant.get("annual_income") if "annual_income" in applicant else applicant.get("AMT_INCOME_TOTAL")
    ann = applicant.get("annuity") if "annuity" in applicant else applicant.get("AMT_ANNUITY")
    dti_ratio = None
    if inc is not None and ann is not None:
        try:
            inc_f = float(inc)
            ann_f = float(ann)
            if inc_f > 0:
                monthly_income = inc_f / 12.0
                dti_val = ann_f / monthly_income
                if not (np.isnan(dti_val) or np.isinf(dti_val)):
                    dti_ratio = round(float(np.clip(dti_val, 0.0, 999.0)), 4)
                if dti_val > 1.0:
                    warnings.append(f"DTI ratio is high ({dti_val:.1%}, annuity exceeds monthly income)")
        except Exception:
            pass

    # Build model input row overlaying user inputs on backend base row
    row = dict(base_row)
    for k, v in applicant.items():
        if k == "age" and v is not None:
            try:
                row["DAYS_BIRTH"] = -float(v) * 365.0
            except Exception:
                pass
        elif k == "years_employed" and v is not None:
            try:
                row["DAYS_EMPLOYED"] = -float(v) * 365.0
            except Exception:
                pass
        elif k in FRIENDLY_TO_COL and v is not None:
            row[FRIENDLY_TO_COL[k]] = v
        elif k in row:
            row[k] = v

    # Sync AMT_GOODS_PRICE if AMT_CREDIT is updated
    if "AMT_CREDIT" in row and row["AMT_CREDIT"] is not None and "AMT_GOODS_PRICE" in row:
        row["AMT_GOODS_PRICE"] = row["AMT_CREDIT"]

    # Align columns and dtypes using blank_template
    df_row = blank_template.copy()
    for c in blank_template.columns:
        val = row.get(c, np.nan)
        if val is None or pd.isna(val):
            df_row.at[0, c] = np.nan
        elif pd.api.types.is_numeric_dtype(df_row[c]):
            try:
                val_f = float(val)
                df_row.at[0, c] = np.nan if (np.isnan(val_f) or np.isinf(val_f)) else val_f
            except Exception:
                df_row.at[0, c] = np.nan
        else:
            df_row.at[0, c] = str(val)

    # 1. Compute calibrated default probability
    try:
        prob = float(pipeline.predict_proba(df_row)[0, 1])
        prob = float(np.clip(prob, 0.0, 1.0))
        if np.isnan(prob) or np.isinf(prob):
            prob = 0.5
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model scoring error: {e}")

    # 2. Decision logic: strictly based on optimized business threshold
    is_approved = prob < threshold
    status = "Approved" if is_approved else "Rejected"

    # 3. Generate top 3 reasons for user-entered fields
    top_3 = compute_reasons(df_row, applicant)

    return {
        "probability_of_default": round(prob, 4),
        "status": status,
        "threshold_used": round(threshold, 4),
        "calibrated": bool(bundle.get("calibrated", True)),
        "dti_ratio": dti_ratio,
        "top_3_reasons": top_3,
        "warnings": warnings,
    }


FRONTEND_DIR = ROOT_DIR / "frontend"

@app.get("/risk-scorer", include_in_schema=False)
async def serve_risk_scorer():
    from fastapi.responses import FileResponse
    scorer_file = FRONTEND_DIR / "risk_scorer.html"
    if scorer_file.exists():
        return FileResponse(scorer_file)
    raise HTTPException(status_code=404, detail="risk_scorer.html not found")

