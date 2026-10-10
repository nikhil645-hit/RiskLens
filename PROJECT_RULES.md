# Project Rules (Non-Negotiable)

## ML rules
1. NO DATA LEAKAGE. Every imputer, scaler and encoder is fitted inside a scikit-learn Pipeline, on training data only. Validation and test data are only transformed, never fitted on.
2. Split into train / validation / test (about 70/15/15), stratified on TARGET, random_state=42.
3. The TEST set is used exactly ONCE, in Stage 8. Never look at test results before that.
4. The decision threshold is chosen on VALIDATION data only.
5. Never report accuracy alone. Always report ROC-AUC, PR-AUC, precision, recall, F1 and the confusion matrix.
6. Replace the anomaly DAYS_EMPLOYED == 365243 with NaN.
7. Drop SK_ID_CURR (it is an ID, not information).
8. Use class_weight="balanced" for Logistic Regression.
9. If any result looks too good (ROC-AUC above about 0.85), STOP and investigate for leakage. Realistic ROC-AUC for Logistic Regression here is roughly 0.73 to 0.76.
10. Report honest results, even if a simple method wins.
11. Never hardcode fake numbers anywhere (README, reports, website). Every number must come from code that was actually run.
12. Before installing any package, tell me which one and why.
13. Every stage that produces results must also save them as JSON/CSV/PNG in reports/ so the backend and website can read real numbers later.

## Website and security rules
14. Model logic lives ONLY in the backend. The frontend never contains scoring logic.
15. Validate all API inputs with Pydantic. Configure CORS only for my frontend origin(s) via environment variables.
16. The website must show this disclaimer in the footer and on the Scorer page: "Educational project. Not for real lending decisions."
17. Never put secrets, API keys or the dataset in the repo. Never log applicant inputs.

## Design fidelity rules
18. The design/ folder is the source of truth. Do NOT redesign, "improve", or change colours, fonts, spacing, radius, shadows or layout. Match the Stitch design exactly.
19. Fonts, icons and images must come from the same source as in the design or be downloaded locally, never silently replaced.
