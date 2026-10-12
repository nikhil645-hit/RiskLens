"""
calibrated_pipeline  (project-root level)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Importable as `import calibrated_pipeline` from any process whose CWD is the
project root (check_stages.py, the API, diagnose.py).

Also importable as `from src.calibrated_pipeline import ...` from scripts
inside src/ that add the project root to sys.path.

joblib pickles the full dotted import path of every class it serialises.
Both `calibrated_pipeline.CalibratedPipeline` and
`src.calibrated_pipeline.CalibratedPipeline` are kept in sync so the bundle
loads regardless of which path was active when it was saved.
"""
from __future__ import annotations

import numpy as np


class CalibratedPipeline:
    """
    Wraps a fitted sklearn pipeline + an IsotonicRegression calibrator.

    Exposes predict_proba(X) -> ndarray shape (n, 2) so the API and
    diagnose.py treat it identically to a plain sklearn pipeline.
    Also exposes feature_names_in_ for diagnose.py's blank_row().
    """

    def __init__(self, pipeline, calibrator, feature_names):
        self.pipeline = pipeline
        self.calibrator = calibrator
        self.feature_names_in_ = np.array([str(f) for f in feature_names], dtype=object)

    def predict_proba(self, X):
        if hasattr(X, "columns"):
            X = X.copy()
            X.columns = [str(c) for c in X.columns]
        raw = self.pipeline.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        cal = np.clip(cal, 0.0, 1.0)
        return np.column_stack([1.0 - cal, cal])
