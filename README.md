# RiskLens — Credit Risk Scoring Engine & Underwriter Telemetry Platform

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Disclaimer**: *Educational project. Not for real lending decisions.* (In compliance with regulatory and project fair-lending guidelines).

---

## 1. Executive Summary & Problem Formulation

RiskLens is an end-to-end credit scoring engine and underwriter decision portal designed for consumer loan default prediction. Built on the **Home Credit Default Risk** benchmark (~307,000 real-world loan applications), RiskLens addresses the two primary failure modes of real-world financial ML systems:
1. **Extreme Class Imbalance (8.07% Default Rate)**: Naive classification accuracy is a deceptive metric (a trivial model predicting "always approve" achieves 91.93% accuracy but bankrupts the lender).
2. **Asymmetric Business Costs**: A False Negative (approving a borrower who defaults on a $500,000 loan) costs a financial institution approximately **5× more** than a False Positive (rejecting a creditworthy applicant who would generate interest margin).

RiskLens implements a leak-free Scikit-Learn pipeline, non-parametric **Isotonic probability calibration**, cost-optimal decision thresholding ($\tau = 0.160$), and adverse-action odds explainability wrapped in an asynchronous **FastAPI** gateway and interactive underwriter UI.

---

## 2. System Architecture & End-to-End Workflow

The architecture is partitioned into four decoupled layers to ensure strict separation of concerns, single-point decision authority, and regulatory traceability:

```
[ Underwriter Client Layer ]  -->  Enters financial parameters (risk_scorer.html workspace)
             |
             v
[ FastAPI Ingestion Gateway ] -->  Validates schema (Pydantic v2), computes DTI & day offsets
             |
             v
[ Calibrated ML Core ]        -->  117-col Median Imputer -> L2 Logistic Classifier -> Isotonic Calibrator
             |
             v
[ Governance & Policy Gate ]  -->  Evaluates 5:1 Cost Gate (tau = 0.160) -> Extracts Odds Multipliers
             |
             v
[ Underwriter Telemetry UI ]  <--  Returns Needle Gauge, Probability, Verdict, and Top-3 Drivers
```

*An interactive animated SVG/HTML diagram of this data pipeline is available in [workflow.html](workflow.html) (rendered image: [workflow.png](workflow.png)).*

---

## 3. Empirical Model Benchmarks & Validation Results

In accordance with strict reproducibility standards (zero data leakage; test set touched exactly once at final evaluation), the model was benchmarked across standard probabilistic and classification metrics:

### Validation & Benchmark Comparison (Validation Set)

| Model Architecture | Hyperparameters / Setup | Validation ROC-AUC | Validation PR-AUC | Training Time | Decision Verdict |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Regularized Logistic Regression** | `L2 penalty, C=0.001, class_weight='balanced'` | **0.7425** | **0.2210** | 29.9s | **Selected Production Model** |
| **LightGBM Classifier** | `n_estimators=500, lr=0.05, num_leaves=31` | **0.7437** | **0.2280** | 16.0s | Excluded (Gain $< 0.010$) |

### Engineering Rationale: Why Logistic Regression Won
Under our project selection rule, an opaque gradient-boosted model (LightGBM) is only adopted if it beats the interpretable linear baseline by at least **+0.010 ROC-AUC**. Because LightGBM achieved a marginal delta of only **+0.0012 ROC-AUC**, the calibrated linear classifier was selected. Under consumer credit regulations (e.g., Fair Credit Reporting Act, Equal Credit Opportunity Act), linear log-odds allow exact closed-form adverse action explanations:

$$\text{Odds Multiplier} = \exp(\beta_j \cdot x_j)$$

This gives underwriters and compliance auditors exact mathematical attribution for why an applicant was approved or rejected.

---

### Final Test Evaluation (Held-Out Test Set: 46,127 applications)

Evaluated exactly once on the final unseen test set:

| Metric | Measured Score | Industry / Baseline Context |
| :--- | :---: | :--- |
| **Test ROC-AUC** | **0.7431** | Strongly competitive for tabular credit underwriting without external data leakage |
| **Test PR-AUC** | **0.2069** | Substantially outperforms random baseline ($8.07\%$) |
| **Mean Predicted Probability** | **8.10%** | Aligns with true population default frequency ($8.1\%$) via Isotonic Calibration |
| **Decision Threshold ($\tau$)** | **0.1600** | Cost-optimal threshold under a 5:1 loss matrix |
| **Test Flag Rate** | **10.54%** | Flags 4,862 applications out of 46,127 test rows |
| **Test Precision** | **24.80%** | 1 in 4 flagged applicants is a true defaulter |
| **Test Recall** | **32.38%** | Successfully catches ~1/3 of all defaulting loans at only 10.5% review rate |
| **Test F1 Score** | **0.2809** | Harmonic balance between credit loss avoidance and customer capture |

#### Confusion Matrix (Test Set, $N = 46,127$)
* **True Positives (TP)**: 1,206 defaults prevented
* **False Positives (FP)**: 3,656 good borrowers flagged for review
* **False Negatives (FN)**: 2,518 unflagged defaults
* **True Negatives (TN)**: 38,747 good loans cleanly approved

---

## 4. Cost-Sensitive Optimization (The 5:1 Loss Matrix)

Standard ML classification defaults to an arbitrary threshold of $\tau = 0.50$. In credit lending, default costs dwarf processing costs:
* **Cost of False Negative ($C_{\text{FN}}$)**: Full loss of unpaid principal and recovery costs ($\approx \$50,000$).
* **Cost of False Positive ($C_{\text{FP}}$)**: Lost interest margin and manual underwriter review time ($\approx \$10,000$).

$$\text{Cost Ratio} = \frac{C_{\text{FN}}}{C_{\text{FP}}} = 5:1$$

The total expected business loss function is:
$$\mathcal{L}(\tau) = 5 \cdot \text{FN}(\tau) + 1 \cdot \text{FP}(\tau)$$

By evaluating the loss curve across thresholds $\tau \in [0.01, 0.99]$ on the validation set, the global cost minimum is achieved at:
$$\tau^* = 0.160$$

Operating at $\tau = 0.160$ reduces portfolio credit loss by **38.4%** compared to naive thresholding.

---

## 5. Repository Structure

```
RiskLens/
├── backend/
│   ├── main.py                    # FastAPI application, Pydantic schemas, endpoints
│   └── tests/                     # Unit and integration tests for API routes
├── models/
│   ├── risk_model.joblib          # Serialized production pipeline (Scikit-Learn + Isotonic)
│   └── sample_request.json        # 117-column baseline request payload
├── frontend/
│   ├── index.html                 # Platform overview & entry point
│   ├── risk_scorer.html           # Live interactive underwriting workspace
│   ├── threshold_lab.html         # Interactive loss matrix & ROC explorer
│   ├── model_insights.html        # Feature importances & calibration curves
│   └── about.html                 # Regulatory compliance & methodology documentation
├── reports/
│   ├── benchmark.json             # LightGBM vs Logistic Regression benchmark evidence
│   ├── final_test_metrics.json    # Exact test evaluation metrics & confusion matrix
│   ├── threshold.json             # 5:1 cost optimization curve & validation sweep
│   └── RiskLens_Report.docx       # Generated executive briefing document
├── tests/
│   ├── check_stages.py            # Automated curriculum stage verification suite
│   ├── diagnose.py                # Deep diagnostic test suite for ML & API integrity
│   └── test_backend.py            # Pytest test cases for FastAPI endpoints
├── Dockerfile                     # Multi-stage production container build
├── .dockerignore                  # Docker exclusion rules
├── .env.example                   # Environment configuration template
├── requirements.txt               # Pinned Python package dependencies
└── workflow.html                  # Self-contained interactive architecture diagram
```

---

## 6. Quickstart & Installation

### Option A: Local Python Environment

1. **Clone & Set Up Virtual Environment**:
   ```bash
   git clone https://github.com/nikhil645-hit/RiskLens.git
   cd RiskLens
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   ```bash
   cp .env.example .env
   ```

3. **Start the FastAPI Backend**:
   ```bash
   uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
   ```
   *Interactive API docs available at `http://127.0.0.1:8000/docs`.*

4. **Serve the Underwriter Frontend**:
   ```bash
   python -m http.server 3000 --directory frontend
   ```
   *Access the web application at `http://localhost:3000`.*

---

### Option B: Docker Container Deployment

1. **Build the Container Image**:
   ```bash
   docker build -t risklens-api:latest .
   ```

2. **Run the Container with Healthcheck**:
   ```bash
   docker run -d -p 8000:8000 --name risklens risklens-api:latest
   ```

3. **Verify Container Health**:
   ```bash
   curl http://localhost:8000/health
   # Returns: {"status": "ok", "model_loaded": true, "threshold": 0.16}
   ```

---

## 7. Running Tests & Quality Verification

Run the automated test and diagnostic suites:

```bash
# Verify all project stages (Stages 0 through 12)
python tests/check_stages.py

# Run backend unit and endpoint tests
pytest tests/test_backend.py -v

# Run the comprehensive diagnostic verification suite
python tests/diagnose.py
```

---

## 8. Portfolio & Resume Highlights (STAR Format)

If presenting RiskLens on your resume or in technical interviews:

* **End-to-End ML Pipeline & Anti-Leakage Design**:  
  *Architected a production credit scoring engine on 307K Home Credit applications, enforcing leak-free feature pipelines with median imputation, negative day offset normalization, and Debt-to-Income (DTI) feature engineering.*

* **Empirical Model Selection & Adverse-Action Explainability**:  
  *Benchmarked LightGBM against regularized L2 Logistic Regression. Selected calibrated Logistic Regression (0.7431 Test ROC-AUC vs 0.7437 LightGBM) to guarantee closed-form odds multipliers $\exp(\beta \cdot x)$ for regulatory fair-lending compliance.*

* **Asymmetric Cost-Sensitive Thresholding**:  
  *Formulated a 5:1 False Negative to False Positive loss matrix on validation data, optimizing the decision threshold from 0.50 to $\tau = 0.160$ and reducing expected default portfolio loss by 38.4%.*

* **Production API & Deployment Architecture**:  
  *Built an asynchronous FastAPI scoring microservice with Pydantic v2 validation, CORS controls, Docker containerization, and interactive telemetry dashboards with animated workflow visualizations.*
