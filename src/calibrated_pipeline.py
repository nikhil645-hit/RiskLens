"""
src.calibrated_pipeline
~~~~~~~~~~~~~~~~~~~~~~~
Stable importable home for CalibratedPipeline.

joblib pickles the *full dotted path* of a class.  If the class is defined
inside a __main__ script (phase_a_train.py), any other process that tries to
load the bundle gets AttributeError.  Keeping the class here means every
consumer (API, diagnose.py, check_stages.py) can resolve it.
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
