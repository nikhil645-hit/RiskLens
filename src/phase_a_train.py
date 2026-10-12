#!/usr/bin/env python3
"""
Phase A: clean retrain, benchmark, calibrate, threshold, final test.

Drops: AMT_GOODS_PRICE (collinear), CODE_GENDER (protected), NAME_FAMILY_STATUS (gender proxy).
Note: cv='prefit' was removed in sklearn 1.9 -- uses IsotonicRegression directly instead.
Keeps: CNT_CHILDREN (legitimate financial-burden signal).
Calibrates with isotonic regression on validation set.
Chooses threshold via cost optimisation (3:1, 5:1, 10:1) on validation set.
Runs LightGBM benchmark; uses winner as production model.
Test set touched exactly once at the very end (Stage 8 gate).

Run from project root:
    .\.venv\Scripts\python.exe src\phase_a_train.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import from root-level module so joblib pickles 'calibrated_pipeline.CalibratedPipeline'
# -- resolvable from any process whose CWD is the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calibrated_pipeline import CalibratedPipeline  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
REPORTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

# Columns to drop before any modelling
DROP_COLS = [
    "AMT_GOODS_PRICE",    # r~0.97 with AMT_CREDIT -> collinear, coefficients cancel
    "CODE_GENDER",        # protected class (sex/gender) -- illegal for credit scoring
    "NAME_FAMILY_STATUS", # proxy for CODE_GENDER / marital status -- protected
]

# Cost ratios for threshold optimisation (fn_cost : fp_cost)
COST_RATIOS = {"3:1": (3, 1), "5:1": (5, 1), "10:1": (10, 1)}
CHOSEN_RATIO = "5:1"   # primary threshold used in production


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def drop_protected(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in DROP_COLS if c in df.columns])


def build_logreg_pipeline(cat_cols, num_cols):
    num_t = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_t = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", OneHotEncoder(min_frequency=50, handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([
        ("num", num_t, num_cols),
        ("cat", cat_t, cat_cols),
    ], remainder="drop")
    return Pipeline([
        ("preprocessor", pre),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])


def cost_threshold(y_true, y_prob, fn_w, fp_w):
    best_t, best_cost = 0.5, float("inf")
    for t in np.linspace(0.01, 0.99, 99):
        pred = y_prob >= t
        fn = int(((~pred) & (y_true == 1)).sum())
        fp = int((pred & (y_true == 0)).sum())
        cost = fn * fn_w + fp * fp_w
        if cost < best_cost:
            best_t, best_cost = t, cost
    return float(best_t)


def threshold_stats(y_true, y_prob, t):
    pred = y_prob >= t
    tp = int((pred & (y_true == 1)).sum())
    fp = int((pred & (y_true == 0)).sum())
    fn = int(((~pred) & (y_true == 1)).sum())
    flag_rate = float(pred.mean())
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    return {"threshold": round(t, 4), "flag_rate": round(flag_rate, 4),
            "precision": round(precision, 4), "recall": round(recall, 4)}


# ---------------------------------------------------------------------------
# STEP 0: AMT_CREDIT / AMT_GOODS_PRICE correlation
# ---------------------------------------------------------------------------
def step0_correlation():
    print("\n" + "=" * 60)
    print("STEP 0  AMT_CREDIT vs AMT_GOODS_PRICE correlation")
    print("=" * 60)
    X_train = pd.read_pickle(DATA / "X_train.pkl")
    if "AMT_CREDIT" in X_train.columns and "AMT_GOODS_PRICE" in X_train.columns:
        r = X_train["AMT_CREDIT"].corr(X_train["AMT_GOODS_PRICE"])
        print(f"  Pearson r(AMT_CREDIT, AMT_GOODS_PRICE) = {r:.4f}")
        if r > 0.90:
            print(f"  -> r={r:.2f} confirms severe collinearity. Dropping AMT_GOODS_PRICE.")
    else:
        print("  One or both columns not in processed data -- already dropped upstream.")


# ---------------------------------------------------------------------------
# STEP 1: LogReg baseline (no protected cols) on validation
# ---------------------------------------------------------------------------
def step1_logreg(X_train, y_train, X_val, y_val):
    print("\n" + "=" * 60)
    print("STEP 1  Logistic Regression (cleaned) on validation")
    print("=" * 60)
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()
    pipe = build_logreg_pipeline(cat_cols, num_cols)
    t0 = time.time()
    pipe.fit(X_train, y_train)
    elapsed = time.time() - t0
    proba_val = pipe.predict_proba(X_val)[:, 1]
    roc = roc_auc_score(y_val, proba_val)
    pr  = average_precision_score(y_val, proba_val)
    print(f"  ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  time={elapsed:.1f}s")
    return pipe, proba_val, {"name": "LogReg", "roc_auc": round(roc, 4), "pr_auc": round(pr, 4), "fit_time": round(elapsed, 1)}


# ---------------------------------------------------------------------------
# STEP 2: LightGBM benchmark (Stage 7)
# ---------------------------------------------------------------------------
def step2_lgbm(X_train, y_train, X_val, y_val):
    print("\n" + "=" * 60)
    print("STEP 2  LightGBM benchmark (Stage 7)")
    print("=" * 60)
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        print("  lightgbm not installed -- skipping. Install with: pip install lightgbm")
        return None, None, None
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_val[col]   = X_val[col].astype("category")
    lgbm = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    t0 = time.time()
    lgbm.fit(X_train, y_train)
    elapsed = time.time() - t0
    proba_val = lgbm.predict_proba(X_val)[:, 1]
    roc = roc_auc_score(y_val, proba_val)
    pr  = average_precision_score(y_val, proba_val)
    print(f"  ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  time={elapsed:.1f}s")
    return lgbm, proba_val, {"name": "LightGBM", "roc_auc": round(roc, 4), "pr_auc": round(pr, 4), "fit_time": round(elapsed, 1)}


# ---------------------------------------------------------------------------
# STEP 3: pick winner and calibrate
# ---------------------------------------------------------------------------
def step3_calibrate(winner_pipe, X_val, y_val, winner_name):
    print("\n" + "=" * 60)
    print(f"STEP 3  Calibrating {winner_name} with isotonic regression (sklearn 1.9+)")
    print("=" * 60)
    # Get raw probabilities from the already-fitted pipeline
    proba_raw = winner_pipe.predict_proba(X_val)[:, 1]
    y_arr = y_val.to_numpy() if hasattr(y_val, "to_numpy") else np.array(y_val)
    # Fit isotonic calibrator on raw -> true label mapping
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(proba_raw, y_arr)
    proba_cal = np.clip(calibrator.predict(proba_raw), 0.0, 1.0)
    # Wrap into a single object the API can call as pipeline.predict_proba(X)
    feature_names = list(winner_pipe.feature_names_in_) if hasattr(winner_pipe, "feature_names_in_") else []
    cal_pipeline = CalibratedPipeline(winner_pipe, calibrator, feature_names)
    base_rate = float(y_arr.mean())
    mean_pred = float(proba_cal.mean())
    ratio = mean_pred / base_rate if base_rate else float("nan")
    print(f"  True default rate (val): {base_rate:.3f}")
    print(f"  Mean calibrated prob:    {mean_pred:.3f}  (ratio {ratio:.2f}; ideal = 1.0)")
    return cal_pipeline, proba_cal


# ---------------------------------------------------------------------------
# STEP 4: threshold selection on validation
# ---------------------------------------------------------------------------
def step4_threshold(y_val, proba_cal, winner_name):
    print("\n" + "=" * 60)
    print("STEP 4  Threshold selection on validation (cost optimisation)")
    print("=" * 60)
    y_arr = y_val.to_numpy() if hasattr(y_val, "to_numpy") else np.array(y_val)
    base_rate = float(y_arr.mean())
    mean_pred  = float(proba_cal.mean())
    ratio = mean_pred / base_rate if base_rate else float("nan")

    scenarios = {}
    for label, (fn_w, fp_w) in COST_RATIOS.items():
        t = cost_threshold(y_arr, proba_cal, fn_w, fp_w)
        stats = threshold_stats(y_arr, proba_cal, t)
        scenarios[label] = stats
        print(f"  {label:5}  threshold={t:.3f}  flag_rate={stats['flag_rate']:.1%}  "
              f"precision={stats['precision']:.2f}  recall={stats['recall']:.2f}")

    chosen_t = scenarios[CHOSEN_RATIO]["threshold"]
    print(f"\n  -> Chosen threshold ({CHOSEN_RATIO}): {chosen_t:.4f}")

    threshold_doc = {
        "model": winner_name,
        "calibrated": True,
        "base_rate_validation": round(base_rate, 4),
        "mean_predicted_validation": round(mean_pred, 4),
        "calibration_ratio": round(ratio, 3),
        "chosen_cost_ratio": CHOSEN_RATIO,
        "chosen_threshold": chosen_t,
        "scenarios": scenarios,
    }
    out = REPORTS / "threshold.json"
    out.write_text(json.dumps(threshold_doc, indent=2), encoding="utf-8")
    print(f"  Saved {out}")
    return chosen_t, threshold_doc


# ---------------------------------------------------------------------------
# STEP 5: final test set evaluation (Stage 8 gate -- touched once)
# ---------------------------------------------------------------------------
def step5_final_test(calibrated_model, chosen_t, X_test, y_test, winner_name, logreg_stats, lgbm_stats):
    print("\n" + "=" * 60)
    print("STEP 5  Final test set evaluation (touched once -- Stage 8 gate)")
    print("=" * 60)
    # LightGBM needs category dtype on test set too
    cat_cols = X_test.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        X_test[col] = X_test[col].astype("category")

    y_arr = y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.array(y_test)
    proba = calibrated_model.predict_proba(X_test)[:, 1]
    roc   = roc_auc_score(y_arr, proba)
    pr    = average_precision_score(y_arr, proba)
    pred  = proba >= chosen_t
    tp = int((pred & (y_arr == 1)).sum())
    fp = int((pred & (y_arr == 0)).sum())
    fn = int(((~pred) & (y_arr == 1)).sum())
    tn = int(((~pred) & (y_arr == 0)).sum())
    flag_rate = float(pred.mean())
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    print(f"  ROC-AUC   = {roc:.4f}")
    print(f"  PR-AUC    = {pr:.4f}")
    print(f"  Threshold = {chosen_t:.4f}  (5:1 cost optimum from validation)")
    print(f"  Flag rate = {flag_rate:.1%}")
    print(f"  Precision = {precision:.3f}  Recall = {recall:.3f}  F1 = {f1:.3f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    metrics = {
        "model": winner_name,
        "calibrated": True,
        "threshold": round(chosen_t, 4),
        "cost_ratio_used_for_threshold": CHOSEN_RATIO,
        "test_roc_auc": round(float(roc), 4),
        "test_pr_auc": round(float(pr), 4),
        "test_flag_rate": round(flag_rate, 4),
        "test_precision": round(precision, 4),
        "test_recall": round(recall, 4),
        "test_f1": round(f1, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "dropped_columns": DROP_COLS,
        "benchmark": {
            "logreg": logreg_stats,
            "lgbm": lgbm_stats,
        },
    }
    out = REPORTS / "final_test_metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  Saved {out}")
    return metrics


# ---------------------------------------------------------------------------
# STEP 6: save model bundle + sample_request
# ---------------------------------------------------------------------------
def step6_save_bundle(calibrated_model, chosen_t, X_test_sample, winner_name):
    print("\n" + "=" * 60)
    print("STEP 6  Saving model bundle (Stage 8 artefact)")
    print("=" * 60)
    bundle = {
        "pipeline":        calibrated_model,
        "threshold":       chosen_t,
        "calibrated":      True,
        "model_type":      winner_name,
        "dropped_columns": DROP_COLS,
    }
    mp = MODELS / "risk_model.joblib"
    joblib.dump(bundle, mp)
    print(f"  Saved {mp}  ({mp.stat().st_size / 1024:.0f} KB)")

    # Save a clean sample_request.json (protected cols removed)
    row = X_test_sample.iloc[0].copy()
    sample_dict: dict = {}
    for k, v in row.items():
        if pd.isna(v):
            sample_dict[k] = None
        elif isinstance(v, (np.int64, np.int32)):
            sample_dict[k] = int(v)
        elif isinstance(v, (np.float64, np.float32, float)):
            sample_dict[k] = float(v)
        elif hasattr(v, "item"):
            sample_dict[k] = v.item()
        else:
            sample_dict[k] = str(v)

    sp = MODELS / "sample_request.json"
    sp.write_text(json.dumps(sample_dict, indent=2), encoding="utf-8")
    print(f"  Saved {sp}  ({len(sample_dict)} columns, protected cols removed)")

    fp_sp = ROOT / "frontend" / "sample_request.json"
    fp_sp.write_text(json.dumps(sample_dict, indent=2), encoding="utf-8")
    print(f"  Copied to {fp_sp}")


# ---------------------------------------------------------------------------
# STEP 7: save benchmark.json (Stage 7 artefact)
# ---------------------------------------------------------------------------
def step7_benchmark(logreg_stats, lgbm_stats, winner_name):
    print("\n" + "=" * 60)
    print("STEP 7  Saving benchmark.json (Stage 7 artefact)")
    print("=" * 60)
    doc = {
        "winner": winner_name,
        "selection_rule": "LightGBM used if val ROC-AUC > LogReg + 0.01, else LogReg",
        "models": [logreg_stats, lgbm_stats] if lgbm_stats else [logreg_stats],
    }
    out = REPORTS / "benchmark.json"
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Phase A: clean retrain + calibration + threshold + test")
    print("=" * 60)

    print("\nLoading processed splits ...")
    X_train = pd.read_pickle(DATA / "X_train.pkl")
    y_train = pd.read_pickle(DATA / "y_train.pkl")
    X_val   = pd.read_pickle(DATA / "X_val.pkl")
    y_val   = pd.read_pickle(DATA / "y_val.pkl")
    X_test  = pd.read_pickle(DATA / "X_test.pkl")
    y_test  = pd.read_pickle(DATA / "y_test.pkl")
    print(f"  Train={len(X_train):,}  Val={len(X_val):,}  Test={len(X_test):,}")
    print(f"  Default rate -- train={float(y_train.mean()):.3f}  val={float(y_val.mean()):.3f}  test={float(y_test.mean()):.3f}")

    step0_correlation()

    print(f"\n  Dropping protected/collinear: {DROP_COLS}")
    X_train = drop_protected(X_train)
    X_val   = drop_protected(X_val)
    X_test  = drop_protected(X_test)

    # LightGBM copies (category dtype set inside step2)
    X_train_lgbm = X_train.copy()
    X_val_lgbm   = X_val.copy()
    X_test_lgbm  = X_test.copy()

    logreg_pipe, logreg_proba_val, logreg_stats = step1_logreg(X_train, y_train, X_val, y_val)
    lgbm_model, lgbm_proba_val, lgbm_stats      = step2_lgbm(X_train_lgbm, y_train, X_val_lgbm, y_val)

    print("\n" + "=" * 60)
    print("WINNER SELECTION")
    print("=" * 60)
    if lgbm_stats is not None:
        diff = lgbm_stats["roc_auc"] - logreg_stats["roc_auc"]
        print(f"  LogReg   ROC-AUC = {logreg_stats['roc_auc']:.4f}")
        print(f"  LightGBM ROC-AUC = {lgbm_stats['roc_auc']:.4f}  (delta={diff:+.4f})")
        if diff > 0.01:
            winner_name   = "LightGBM"
            winner_pipe   = lgbm_model
            winner_X_val  = X_val_lgbm
            winner_X_test = X_test_lgbm
            print("  -> LightGBM wins by >1% ROC-AUC. Using LightGBM as production model.")
        else:
            winner_name   = "LogReg"
            winner_pipe   = logreg_pipe
            winner_X_val  = X_val
            winner_X_test = X_test
            print("  -> Margin <1%. Using LogReg (simpler, more explainable).")
    else:
        winner_name   = "LogReg"
        winner_pipe   = logreg_pipe
        winner_X_val  = X_val
        winner_X_test = X_test
        print("  -> LightGBM not available. Using LogReg.")

    cal_model, cal_proba_val = step3_calibrate(winner_pipe, winner_X_val, y_val, winner_name)
    chosen_t, threshold_doc  = step4_threshold(y_val, cal_proba_val, winner_name)
    metrics = step5_final_test(cal_model, chosen_t, winner_X_test, y_test,
                               winner_name, logreg_stats, lgbm_stats)
    step6_save_bundle(cal_model, chosen_t, winner_X_test, winner_name)
    step7_benchmark(logreg_stats, lgbm_stats, winner_name)

    print("\n" + "=" * 60)
    print("Phase A COMPLETE")
    print(f"  Winner    : {winner_name} (calibrated, isotonic)")
    print(f"  Threshold : {chosen_t:.4f}  ({CHOSEN_RATIO} cost optimum on validation)")
    print(f"  Test ROC-AUC  : {metrics['test_roc_auc']:.4f}")
    print(f"  Test flag rate: {metrics['test_flag_rate']:.1%}")
    print(f"\nNext step -- run gate check:")
    print(f"  .venv\\Scripts\\python.exe tests\\check_stages.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
