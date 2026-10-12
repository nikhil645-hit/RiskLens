#!/usr/bin/env python3
"""
RiskLens layered diagnosis: finds WHERE a bug lives.

  Layer 0  scan of frontend source code       -> owner FRONTEND (hardcoded values)
  Layer 1  ML model called directly           -> owner ML
  Layer 2  FastAPI called over HTTP           -> owner BACKEND (wiring)
  Layer 3  real browser with Playwright       -> owner FRONTEND (wiring / stale UI)

The idea: the same inputs go through every layer. The FIRST layer whose answer
disagrees with the layer before it is where the bug is.

Examples (run from the project root):
  python tests/diagnose.py --show-schema
  python tests/diagnose.py --layers 0,1,2
  python tests/diagnose.py --layers 0,1,2,3 --headed
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:  # Windows consoles can choke on non-ASCII characters
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CFG = json.loads((HERE / "scenarios.json").read_text(encoding="utf-8"))
BASE = CFG["baseline"]

NUMERIC_KEYS = {"annual_income", "loan_amount", "annuity", "age", "years_employed",
                "ext_source_1", "ext_source_2", "ext_source_3"}
FRIENDLY_TO_COL = {
    "annual_income": "AMT_INCOME_TOTAL", "loan_amount": "AMT_CREDIT", "annuity": "AMT_ANNUITY",
    "education": "NAME_EDUCATION_TYPE", "income_type": "NAME_INCOME_TYPE",
    "occupation": "OCCUPATION_TYPE", "organisation_type": "ORGANIZATION_TYPE",
    "ext_source_1": "EXT_SOURCE_1", "ext_source_2": "EXT_SOURCE_2", "ext_source_3": "EXT_SOURCE_3",
}
DAYS = 365.0  # years -> days. Your frontend uses 365 (override with --days-per-year).
BASE_ROW: dict = {}   # full sample applicant (frontend/sample_request.json): the UI overlays the form on top of it
SYNC_GOODS = True     # the UI copies the loan amount into AMT_GOODS_PRICE

# Names the API response might use. Add yours here if the script cannot find a field.
ALIASES = {
    "probability": ["probability_of_default", "probability", "default_probability", "prob", "pd", "p_default", "risk_probability"],
    "threshold": ["threshold_used", "threshold", "cutoff", "decision_threshold"],
    "decision": ["status", "decision", "verdict", "recommendation", "outcome"],
    "reasons": ["top_3_reasons", "reasons", "top_reasons", "factors", "top_factors", "explanations"],
    "multiplier": ["odds_multiplier", "multiplier", "odds_ratio", "odds_ratio_multiplier"],
    "feature": ["feature", "name", "label", "field"],
}

RESULTS: list[dict] = []


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def rec(layer: int, owner: str, check: str, status: str, detail: str = "") -> None:
    RESULTS.append({"layer": layer, "owner": owner, "check": check, "status": status, "detail": detail})
    print(f"[{status:4}] L{layer} {owner:8} {check}: {detail}")


def num(v):
    if isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_inputs(sc: dict) -> dict:
    inputs = copy.deepcopy(BASE) if sc.get("use_baseline", True) else {}
    inputs.update(copy.deepcopy(sc.get("inputs", {})))
    for k in sc.get("omit", []):
        inputs.pop(k, None)
    return inputs


def with_override(**kw) -> dict:
    d = copy.deepcopy(BASE)
    d.update(kw)
    return d


def pick(d, name):
    """Find a field in an API response using the alias list (one level of nesting allowed)."""
    if not isinstance(d, dict):
        return None
    for k in ALIASES[name]:
        if k in d:
            return d[k]
    for v in d.values():
        if isinstance(v, dict):
            for k in ALIASES[name]:
                if k in v:
                    return v[k]
    return None


def scenario_by_id(sid: str) -> dict:
    return next(s for s in CFG["scenarios"] if s["id"] == sid)


def ok_scenarios():
    return [s for s in CFG["scenarios"] if s["expect"] == "ok"]


# ----------------------------------------------------------------------------
# model helpers (layer 1)
# ----------------------------------------------------------------------------
def load_bundle(path: Path):
    import joblib
    b = joblib.load(path)
    if isinstance(b, dict) and "pipeline" in b:
        return b["pipeline"], (float(b["threshold"]) if b.get("threshold") is not None else None)
    return b, None  # a plain scikit-learn pipeline with no threshold inside


def blank_row(pipe, train_csv: Path) -> pd.DataFrame:
    """One row, every column NaN, with the same dtypes the model saw in training."""
    names = list(pipe.feature_names_in_)
    dtypes = {}
    try:
        ref = pd.read_csv(train_csv, nrows=3000)
        dtypes = {c: ref[c].dtype for c in names if c in ref.columns}
    except Exception:
        pass
    data = {}
    for c in names:
        dt = dtypes.get(c)
        numeric = dt is not None and pd.api.types.is_numeric_dtype(dt)
        data[c] = pd.Series([np.nan], dtype="float64" if numeric else object)
    return pd.DataFrame(data)[names]


def friendly_to_columns(inputs: dict) -> dict:
    """Form fields -> dataset columns, exactly like the website does it."""
    out = {}
    for k, v in inputs.items():
        if k == "age" and num(v) is not None:
            out["DAYS_BIRTH"] = -float(v) * DAYS
        elif k == "years_employed" and num(v) is not None:
            out["DAYS_EMPLOYED"] = -float(v) * DAYS
        elif k in FRIENDLY_TO_COL:
            out[FRIENDLY_TO_COL[k]] = v
    if SYNC_GOODS and "AMT_CREDIT" in out and num(out["AMT_CREDIT"]) is not None:
        out["AMT_GOODS_PRICE"] = out["AMT_CREDIT"]
    return out


def to_model_row(blank: pd.DataFrame, inputs: dict) -> pd.DataFrame:
    row = blank.copy()
    idx = row.index[0]
    merged = {c: v for c, v in BASE_ROW.items() if v is not None}
    merged.update(friendly_to_columns(inputs))  # an explicit null from the form means "missing", like in the API
    for col, val in merged.items():
        if col not in row.columns:
            continue
        if val is None:
            row.at[idx, col] = np.nan
            continue
        if pd.api.types.is_numeric_dtype(row[col]):
            val = float(val)
        row.at[idx, col] = val
    return row


def model_prob(pipe, blank, inputs):
    try:
        return float(pipe.predict_proba(to_model_row(blank, inputs))[0, 1]), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


# ----------------------------------------------------------------------------
# API helpers (layer 2)
# ----------------------------------------------------------------------------
def api_payload(inputs: dict, style: str) -> dict:
    if style == "friendly":
        return inputs
    cols = friendly_to_columns(inputs)
    if style == "full":
        out = dict(BASE_ROW)
        out.update(cols)
        return out
    return cols


def api_call(args, inputs: dict):
    import requests
    url = args.api.rstrip("/") + args.predict_path
    try:
        r = requests.post(url, json=api_payload(inputs, args.payload_style), timeout=15)
        try:
            body = r.json()
        except Exception:  # noqa: BLE001
            body = r.text
        return r.status_code, body
    except Exception as e:  # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def api_prob(args, inputs):
    status, body = api_call(args, inputs)
    p = num(pick(body, "probability")) if status == 200 else None
    return p, status, body


# ----------------------------------------------------------------------------
# LAYER 0: static scan of the frontend source
# ----------------------------------------------------------------------------
def layer0(args):
    fdir = Path(args.frontend_dir)
    if not fdir.exists():
        rec(0, "FRONTEND", "frontend folder", "WARN", f"{fdir} not found, skipping scan")
        return
    skip_dirs = {"node_modules", ".next", "dist", "build", ".git", "out", "coverage"}
    exts = {".ts", ".tsx", ".js", ".jsx", ".html", ".css", ".mdx", ".vue"}
    hits: dict[str, list[str]] = {}
    for p in fdir.rglob("*"):
        if not p.is_file() or p.suffix not in exts or any(part in skip_dirs for part in p.parts):
            continue
        if "lock" in p.name.lower():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            for s in CFG["banned_strings"]:
                if s in line:
                    hits.setdefault(s, []).append(f"{p.relative_to(fdir)}:{i}")
    if not hits:
        rec(0, "FRONTEND", "banned strings in source", "PASS", "none found")
    for s, where in hits.items():
        # bare numbers like "814" can be harmless CSS values, so those are warnings
        status = "WARN" if re.fullmatch(r"[\d.,]+", s) else "FAIL"
        rec(0, "FRONTEND", f"banned string '{s}'", status, "; ".join(where[:5]) + (" ..." if len(where) > 5 else ""))


# ----------------------------------------------------------------------------
# LAYER 1: ML model directly
# ----------------------------------------------------------------------------
def validation_report(thr: float, val_csv: Path):
    if not val_csv.exists():
        rec(1, "ML", "validation file", "WARN", f"{val_csv} not found, skipping threshold analysis")
        return
    df = pd.read_csv(val_csv)
    ycol = next((c for c in ["y_true", "TARGET", "target", "label", "y"] if c in df.columns), None)
    pcol = next((c for c in ["y_prob", "proba", "probability", "y_pred_proba", "pred_proba", "prob", "p"] if c in df.columns), None)
    if not ycol or not pcol:
        rec(1, "ML", "validation columns", "WARN", f"could not find label/probability columns in {list(df.columns)}")
        return
    y, p = df[ycol].astype(int).to_numpy(), df[pcol].astype(float).to_numpy()
    base_rate, mean_pred = y.mean(), p.mean()
    flagged = p >= thr
    flag_rate = flagged.mean()
    tp = int((flagged & (y == 1)).sum())
    precision = tp / max(int(flagged.sum()), 1)
    recall = tp / max(int((y == 1).sum()), 1)
    rec(1, "ML", "saved threshold stats", "INFO",
        f"threshold={thr:.3f}; flagged {flag_rate:.1%} of validation rows; precision={precision:.2f}; recall={recall:.2f}")
    if flag_rate < 0.02 or flag_rate > 0.60:
        rec(1, "ML", "threshold sanity", "FAIL",
            f"threshold {thr:.2f} flags only {flag_rate:.1%} of applicants. A cost-based cut-off should flag a meaningful share. Re-check Stage 5/8.")
    else:
        rec(1, "ML", "threshold sanity", "PASS", f"flag rate {flag_rate:.1%}")
    ratio = mean_pred / base_rate if base_rate else float("nan")
    if ratio > 1.5 or ratio < 0.67:
        rec(1, "ML", "calibration", "WARN",
            f"mean predicted risk {mean_pred:.1%} vs real default rate {base_rate:.1%}. Probabilities are NOT calibrated "
            f"(class_weight='balanced' inflates them). The UI must not say 'calibrated'.")
    else:
        rec(1, "ML", "calibration", "PASS", f"mean predicted {mean_pred:.1%} vs actual {base_rate:.1%}")
    best_t, best_cost = None, float("inf")
    for t in np.linspace(0.01, 0.99, 99):
        pred = p >= t
        fn = int(((~pred) & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        cost = fn * 5 + fp * 1
        if cost < best_cost:
            best_t, best_cost = t, cost
    diff = abs(best_t - thr)
    rec(1, "ML", "threshold vs 5:1 cost optimum", "PASS" if diff <= 0.15 else "WARN",
        f"saved {thr:.2f} vs recomputed optimum {best_t:.2f} (assumes cost ratio 5:1)")


def layer1(args):
    mpath = Path(args.model)
    if not mpath.exists():
        rec(1, "ML", "model file", "FAIL", f"{mpath} not found")
        return None
    try:
        pipe, thr = load_bundle(mpath)
    except Exception as e:  # noqa: BLE001
        rec(1, "ML", "load model", "FAIL", f"{type(e).__name__}: {e}")
        return None
    bundled = thr is not None
    if not bundled:
        thr = args.threshold
        rec(1, "ML", "threshold stored with model", "WARN",
            f"{mpath.name} is a plain pipeline with no threshold. Assuming {thr} (the value hardcoded in your API). "
            "Stage 8 should save the threshold with the model.")
    blank = blank_row(pipe, Path(args.train_csv))
    rec(1, "ML", "load model", "PASS", f"{mpath.name}: {len(blank.columns)} input columns, threshold={thr:.3f}")
    if not BASE_ROW:
        rec(1, "ML", "base row", "WARN", "no sample_request.json found; all non-form columns are empty in the model call")

    validation_report(thr, Path(args.validation_csv))

    model_cache = {}
    for sc in CFG["scenarios"]:
        if sc["expect"] not in ("ok", "ok_or_reject"):
            continue
        inputs = build_inputs(sc)
        p, err = model_prob(pipe, blank, inputs)
        model_cache[sc["id"]] = p
        if p is None:
            status = "FAIL" if sc["expect"] == "ok" else "INFO"
            rec(1, "ML", f"{sc['id']} predicts", status, f"model raised {err}")
            continue
        if not (0.0 <= p <= 1.0) or np.isnan(p):
            rec(1, "ML", f"{sc['id']} probability range", "FAIL", f"p={p}")
        p2, _ = model_prob(pipe, blank, inputs)
        if p2 != p:
            rec(1, "ML", f"{sc['id']} deterministic", "FAIL", f"{p} vs {p2}")
    ok_ps = {k: v for k, v in model_cache.items() if v is not None}
    if ok_ps:
        rec(1, "ML", "scenario probabilities", "INFO",
            ", ".join(f"{k}={v:.3f}" for k, v in list(ok_ps.items())[:8]))
        spread = max(ok_ps.values()) - min(ok_ps.values())
        rec(1, "ML", "model reacts to inputs", "PASS" if spread > 0.05 else "FAIL",
            f"probability spread across scenarios = {spread:.3f}")

    lo, hi = model_cache.get("LOW_RISK_STRONG"), model_cache.get("HIGH_RISK_STRONG")
    if lo is not None and hi is not None:
        rec(1, "ML", "safe applicant < risky applicant", "PASS" if lo < hi else "FAIL",
            f"LOW_RISK={lo:.3f}, HIGH_RISK={hi:.3f}")

    curves = {}
    for field, values in CFG["sweeps"].items():
        curves[field] = []
        for v in values:
            p, _ = model_prob(pipe, blank, with_override(**{field: v}))
            curves[field].append(p)
    for field in CFG["monotone_non_increasing"]:
        c = curves[field]
        if any(x is None for x in c):
            rec(1, "ML", f"{field} monotone", "FAIL", "model failed on part of the sweep")
            continue
        ok = all(c[i + 1] <= c[i] + 1e-9 for i in range(len(c) - 1))
        rec(1, "ML", f"{field}: higher score => lower risk", "PASS" if ok else "FAIL",
            "curve " + ", ".join(f"{x:.3f}" for x in c))

    try:
        p, err = model_prob(pipe, blank, with_override(occupation="Astronaut"))
        rec(1, "ML", "unseen category handled", "PASS" if p is not None else "FAIL", err or f"p={p:.3f}")
    except Exception as e:  # noqa: BLE001
        rec(1, "ML", "unseen category handled", "FAIL", str(e))

    return {"pipe": pipe, "thr": thr, "bundled": bundled, "blank": blank, "curves": curves, "cache": model_cache}


# ----------------------------------------------------------------------------
# LAYER 2: API
# ----------------------------------------------------------------------------
def layer2(args, m):
    import requests
    base = args.api.rstrip("/")
    try:
        r = requests.get(base + "/health", timeout=10)
        if r.status_code == 200:
            rec(2, "BACKEND", "GET /health", "PASS", "status 200")
        else:
            rec(2, "BACKEND", "GET /health", "WARN", f"status {r.status_code} (the Stage 9 plan requires this route)")
    except Exception as e:  # noqa: BLE001
        rec(2, "BACKEND", "API reachable", "FAIL", f"{base} not reachable: {e}")
        return None

    # which routes does this API really have? (FastAPI publishes /openapi.json)
    try:
        spec = requests.get(base + "/openapi.json", timeout=10).json()
        if "paths" not in spec:
            raise ValueError("not an OpenAPI document")
        paths = sorted(spec.get("paths", {}))
        listing = ", ".join(f"{p} [{'/'.join(x.upper() for x in spec['paths'][p])}]" for p in paths)
        rec(2, "BACKEND", "routes the API really has", "INFO", listing or "none")
        if args.predict_path not in paths:
            near = [p for p in paths if "predict" in p or "score" in p]
            rec(2, "BACKEND", f"route {args.predict_path} exists", "FAIL",
                f"not found. Similar routes: {near or 'none'}. Re-run with --predict-path <route>.")
            return None
        for needed in ("/health", "/schema", "/threshold", "/insights"):
            if needed not in paths:
                rec(2, "BACKEND", f"route {needed} exists", "WARN", "missing (the Stage 9 plan lists it)")
    except Exception:  # noqa: BLE001
        rec(2, "BACKEND", "routes the API really has", "INFO", "no /openapi.json, skipping route listing")

    # PREFLIGHT: one normal request. If this fails, every test below fails the same way.
    status, body = api_call(args, dict(BASE))
    if status != 200:
        hint = ""
        if status == 422:
            hint = " The API rejected the payload format: try --payload-style columns, or read the 422 text above."
        elif status >= 500:
            hint = (" The server crashed on a NORMAL request. Read the Python traceback in the terminal where "
                    "uvicorn is running: that is the real error. Two common causes: (1) payload format: this API may need the FULL "
                    "dataset row, so use --payload-style full with --base-row frontend/sample_request.json; "
                    "(2) the model file is missing or was saved with a different scikit-learn version.")
        rec(2, "BACKEND", "PREFLIGHT normal request", "FAIL",
            f"status {status}: {str(body)[:300]}.{hint} Layer 2 stops here.")
        return None
    rec(2, "BACKEND", "PREFLIGHT normal request", "PASS", "API answered a normal request")

    thr_model = m["thr"] if (m and m.get("bundled")) else None
    seen_thresholds, probs_api, decisions_bad, reason_issues = set(), {}, [], []
    latencies = []
    n_dec = n_rea = 0
    big_contrib = []

    for sc in CFG["scenarios"]:
        inputs = build_inputs(sc)
        status, body = api_call(args, inputs)
        exp = sc["expect"]
        if status == 0 or status >= 500:
            rec(2, "BACKEND", f"{sc['id']} no server error", "FAIL", f"status {status}: {str(body)[:120]}")
            continue
        if exp == "ok":
            if status != 200:
                rec(2, "BACKEND", f"{sc['id']} accepted", "FAIL", f"status {status}: {str(body)[:150]}")
                continue
        elif exp == "reject":
            if 400 <= status < 500:
                continue
            rec(2, "BACKEND", f"{sc['id']} rejected", "FAIL", f"status {status}, expected 4xx (invalid input accepted)")
            continue
        elif exp == "reject_or_warn":
            if status == 200 and not any("warn" in str(k).lower() for k in (body or {})):
                rec(2, "BACKEND", f"{sc['id']} impossible input", "WARN", "accepted with no warning field")
            continue
        if status != 200:
            continue

        p = num(pick(body, "probability"))
        thr = num(pick(body, "threshold"))
        decision = pick(body, "decision")
        if p is None:
            rec(2, "BACKEND", f"{sc['id']} has probability", "FAIL", f"keys: {list(body)[:10] if isinstance(body, dict) else body}")
            continue
        probs_api[sc["id"]] = p
        if not (0 <= p <= 1):
            rec(2, "BACKEND", f"{sc['id']} probability scale", "FAIL",
                f"p={p}. Must be a fraction 0..1 (percent scaling bug?)")
        if thr is not None:
            seen_thresholds.add(round(thr, 6))
            if decision is not None:
                n_dec += 1
                approve = "approv" in str(decision).lower()
                if (p < thr) != approve:
                    decisions_bad.append(f"{sc['id']} (p={p:.3f}, thr={thr:.3f}, decision='{decision}')")
        if exp == "ok" and sc.get("expect_range_warning"):
            has_warn = any(("warn" in str(k).lower() or "range" in str(k).lower()) for k in (body or {}))
            if not has_warn:
                rec(2, "BACKEND", f"{sc['id']} out-of-range warning", "WARN", "no warning/range field in response")

        reasons = pick(body, "reasons")
        if isinstance(reasons, list):
            n_rea += 1
            allowed = {k.lower() for k in inputs} | {FRIENDLY_TO_COL.get(k, k).lower() for k in inputs}
            if "age" in inputs:
                allowed.add("days_birth")
            if "years_employed" in inputs:
                allowed.add("days_employed")
            if SYNC_GOODS and "loan_amount" in inputs:
                allowed.add("amt_goods_price")
            for r_ in reasons:
                if not isinstance(r_, dict):
                    continue
                contrib = num(r_.get("contribution"))
                if contrib is not None and pick(r_, "multiplier") is None and abs(contrib) > 0.5:
                    big_contrib.append(f"{sc['id']}: {r_.get('feature')} contribution {contrib:+.2f}")
                mult = num(pick(r_, "multiplier"))
                if mult is not None and mult <= 0:
                    reason_issues.append(f"{sc['id']}: multiplier {mult} <= 0")
                fname = str(pick(r_, "feature") or "").lower()
                if fname and not any(a in fname for a in allowed):
                    reason_issues.append(f"{sc['id']}: reason '{fname}' is not an input the user gave")
            if len(reasons) > 3:
                reason_issues.append(f"{sc['id']}: {len(reasons)} reasons returned (expected 3)")

        if m and sc["id"] in m["cache"] and m["cache"][sc["id"]] is not None:
            d = abs(p - m["cache"][sc["id"]])
            if d > 0.01:
                rec(2, "BACKEND", f"{sc['id']} API vs model", "FAIL", f"API={p:.4f}, model={m['cache'][sc['id']]:.4f}")
            elif d > 0.002:
                rec(2, "BACKEND", f"{sc['id']} API vs model", "WARN", f"small gap {d:.4f} (365 vs 365.25 days?)")

    if probs_api:
        rec(2, "BACKEND", "scenario probabilities (API)", "INFO",
            ", ".join(f"{k}={v:.3f}" for k, v in list(probs_api.items())[:8]))
    if len(seen_thresholds) == 0:
        rec(2, "BACKEND", "threshold returned", "FAIL", "responses contain no threshold")
    elif len(seen_thresholds) > 1:
        rec(2, "BACKEND", "single threshold", "FAIL", f"responses use different thresholds: {sorted(seen_thresholds)}")
    else:
        t = next(iter(seen_thresholds))
        if thr_model is not None and abs(t - thr_model) > 1e-4:
            rec(2, "BACKEND", "API threshold == saved model threshold", "FAIL", f"API={t}, model file={thr_model}")
        else:
            rec(2, "BACKEND", "single threshold", "PASS", f"{t}")
    if n_dec == 0:
        rec(2, "BACKEND", "decision matches threshold", "WARN", "NOT TESTED: no response had both a threshold and a decision")
    else:
        rec(2, "BACKEND", "decision matches threshold", "PASS" if not decisions_bad else "FAIL",
            f"{n_dec} responses consistent" if not decisions_bad else "; ".join(decisions_bad[:4]))
    if big_contrib:
        rec(2, "BACKEND", "reason contributions are log-odds", "WARN",
            "API returns 'contribution' in LOG-ODDS. If the UI prints it as a percent it is wrong "
            "(use exp(contribution) as an odds multiplier). Large values: " + "; ".join(big_contrib[:4]))
    if n_rea == 0:
        rec(2, "BACKEND", "reasons are sane", "WARN", "NOT TESTED: no response contained a reasons list")
    else:
        rec(2, "BACKEND", "reasons are sane", "PASS" if not reason_issues else "FAIL",
            "multipliers > 0 and only user-entered fields" if not reason_issues else "; ".join(reason_issues[:5]))

    # sweeps: is every field actually wired through to the model?
    api_curves = {}
    for field, values in CFG["sweeps"].items():
        api_curves[field] = [api_prob(args, with_override(**{field: v}))[0] for v in values]
    for field in CFG["monotone_non_increasing"]:
        c = api_curves[field]
        if any(x is None for x in c):
            rec(2, "BACKEND", f"{field} sweep", "FAIL", "API failed on part of the sweep")
            continue
        ok = all(c[i + 1] <= c[i] + 1e-9 for i in range(len(c) - 1))
        rec(2, "BACKEND", f"{field}: higher score => lower risk (API)", "PASS" if ok else "FAIL",
            "curve " + ", ".join(f"{x:.3f}" for x in c))
    for field in CFG["sweeps"]:
        a = [x for x in api_curves[field] if x is not None]
        mc = [x for x in (m["curves"][field] if m else []) if x is not None]
        d_api = (max(a) - min(a)) if a else 0.0
        d_model = (max(mc) - min(mc)) if mc else None
        if d_model is not None and d_model > 1e-4 and d_api < 1e-6:
            rec(2, "BACKEND", f"field '{field}' wired", "FAIL",
                f"model reacts (delta {d_model:.3f}) but API ignores it (delta {d_api:.3f}). Field dropped or mis-named in the backend.")
        elif d_model is not None and d_model > 1e-4 and abs(d_api - d_model) > 0.02:
            rec(2, "BACKEND", f"field '{field}' wired", "WARN",
                f"effect size differs: model delta {d_model:.3f} vs API delta {d_api:.3f} (unit/sign conversion?)")
        elif d_api < 1e-6 and (d_model is None or d_model < 1e-4):
            rec(2, "ML", f"field '{field}' has any effect", "WARN", "no effect in model or API (unused feature?)")
        else:
            rec(2, "BACKEND", f"field '{field}' wired", "PASS", f"delta API {d_api:.3f}")

    # stability
    p1, _, _ = api_prob(args, dict(BASE))
    p2, _, _ = api_prob(args, dict(reversed(list(BASE.items()))))
    if p1 is not None and p2 is not None:
        rec(2, "BACKEND", "same input, same answer, any key order", "PASS" if abs(p1 - p2) < 1e-9 else "FAIL", f"{p1} vs {p2}")
    return api_curves


# ----------------------------------------------------------------------------
# LAYER 3: browser (needs data-testid attributes, see the prompt)
# ----------------------------------------------------------------------------
PCT = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
ANYNUM = re.compile(r"-?\d[\d,]*\.?\d*")


def first_pct(text: str):
    m = PCT.search(text or "")
    return float(m.group(1)) / 100 if m else None


def flatten_numbers(obj, out):
    if isinstance(obj, dict):
        for v in obj.values():
            flatten_numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            flatten_numbers(v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.add(float(obj))


def fill_field(page, key, value):
    el = page.locator(f'[data-testid="input-{key}"]').first
    label = CFG["ui_label_map"].get(value, value) if isinstance(value, str) else value
    tag = el.evaluate("e => e.tagName.toLowerCase()")
    typ = el.evaluate("e => e.type || ''")
    if tag == "select":
        try:
            el.select_option(label=str(label))
        except Exception:  # noqa: BLE001
            el.select_option(str(value))
    elif typ == "range":
        el.evaluate(
            "(e,v)=>{const s=Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set;"
            "s.call(e,v);e.dispatchEvent(new Event('input',{bubbles:true}));"
            "e.dispatchEvent(new Event('change',{bubbles:true}));}", str(value))
    elif tag in ("input", "textarea"):
        el.fill("" if value is None else str(value))
    else:  # custom dropdown button
        el.click()
        page.get_by_role("option", name=str(label)).first.click()


def run_scenario_in_ui(page, args, inputs, wait_ms=400):
    """Fill the form, submit, return (ui_probability, ui_threshold, request_fired, payload)."""
    pat = re.compile(r".*" + re.escape(args.predict_path) + r".*")
    fired, payload = False, None
    try:
        with page.expect_response(pat, timeout=6000) as resp_info:
            for k in NUMERIC_KEYS | {"education", "income_type", "occupation", "organisation_type"}:
                if k in inputs:
                    fill_field(page, k, inputs[k])
            sub = page.locator('[data-testid="submit"]')
            if sub.count():
                sub.first.click()
        fired = True
        payload = resp_info.value.request.post_data_json
    except Exception:  # noqa: BLE001
        fired = False
    page.wait_for_timeout(wait_ms)
    prob_txt = page.locator('[data-testid="prob-value"]').first.inner_text() if page.locator('[data-testid="prob-value"]').count() else ""
    thr_txt = page.locator('[data-testid="threshold-value"]').first.inner_text() if page.locator('[data-testid="threshold-value"]').count() else ""
    return first_pct(prob_txt), first_pct(thr_txt), fired, payload


def payload_mismatches(payload, inputs):
    """Compare what the page sent with what was typed (after the standard conversions)."""
    wrong = []
    if not isinstance(payload, dict):
        return ["payload is not JSON"]
    for col, want in friendly_to_columns(inputs).items():
        w = num(want)
        if w is None or col not in payload:
            continue
        got = num(payload[col])
        if got is None or abs(got - w) > 1e-6 * max(1.0, abs(w)):
            wrong.append(f"{col}: typed/expected {want} but page sent {payload[col]}")
    return wrong


def layer3(args):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        rec(3, "FRONTEND", "playwright", "FAIL", "pip install playwright && playwright install chromium")
        return
    url = args.front.rstrip("/") + args.scorer_path
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(url)
        page.wait_for_load_state("networkidle")

        missing = [t for t in ["prob-value", "threshold-value", "decision", "submit", "result-panel"]
                   if page.locator(f'[data-testid="{t}"]').count() == 0]
        missing += [f"input-{k}" for k in NUMERIC_KEYS if page.locator(f'[data-testid="input-{k}"]').count() == 0]
        if missing:
            rec(3, "FRONTEND", "data-testid attributes", "FAIL", f"missing: {sorted(missing)}. Add them first.")
            browser.close()
            return

        # (a) rendered page must not contain banned text
        def scan_dom(tag):
            body = page.inner_text("body")
            bad = [s for s in CFG["banned_strings"] if s in body and not re.fullmatch(r"[\d.,]+", s)]
            rec(3, "FRONTEND", f"banned text on screen ({tag})", "PASS" if not bad else "FAIL", ", ".join(bad) or "none")

        base_inputs = build_inputs(scenario_by_id("BASE_TYPICAL"))
        run_scenario_in_ui(page, args, base_inputs)
        scan_dom("after scoring")

        # (b) what does the form actually send?
        _, _, fired, payload = run_scenario_in_ui(page, args, base_inputs)
        if not fired:
            rec(3, "FRONTEND", "form sends request", "FAIL", "no POST to the predict endpoint after submit")
        else:
            rec(3, "FRONTEND", "request payload", "INFO", json.dumps(payload)[:300])
            wrong = payload_mismatches(payload, base_inputs)
            rec(3, "FRONTEND", "typed values == sent values", "PASS" if not wrong else "FAIL", "; ".join(wrong) or "every converted field matches")

        # (c) SENTINEL test: feed the UI a fake API answer. The UI must show exactly that.
        template_p, status, template = api_prob(args, base_inputs)
        sent = CFG["sentinel"]
        if status == 200 and isinstance(template, dict):
            fake = copy.deepcopy(template)
            holder = fake if any(k in fake for k in ALIASES["probability"]) else next(
                (v for v in fake.values() if isinstance(v, dict)), fake)
            for name, val in (("probability", sent["probability"]), ("threshold", sent["threshold"]),
                              ("decision", sent["decision"])):
                for k in ALIASES[name]:
                    if k in holder:
                        holder[k] = val
            reasons = holder.get(next((k for k in ALIASES["reasons"] if k in holder), ""), [])
            for i, r_ in enumerate(reasons if isinstance(reasons, list) else []):
                if isinstance(r_, dict):
                    for k in ALIASES["multiplier"]:
                        if k in r_:
                            r_[k] = sent["multiplier_a"] if i % 2 == 0 else sent["multiplier_b"]
            page.route(re.compile(r".*" + re.escape(args.predict_path) + r".*"),
                       lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps(fake)))
            ui_p, ui_t, fired, _ = run_scenario_in_ui(page, args, base_inputs)
            ok_p = ui_p is not None and abs(ui_p - sent["probability"]) < 0.0006
            ok_t = ui_t is not None and abs(ui_t - sent["threshold"]) < 0.0006
            rec(3, "FRONTEND", "SENTINEL probability shown", "PASS" if ok_p else "FAIL",
                f"API said 43.21%, UI shows {ui_p if ui_p is None else f'{ui_p:.2%}'} "
                + ("" if ok_p else "=> UI is not using the API probability (hardcoded or derived)"))
            rec(3, "FRONTEND", "SENTINEL threshold shown", "PASS" if ok_t else "FAIL",
                f"API said 22.22%, UI shows {ui_t if ui_t is None else f'{ui_t:.2%}'} "
                + ("" if ok_t else "=> UI threshold is not from the API"))
            panel = page.locator('[data-testid="result-panel"]').first.inner_text()
            allowed = {0.0, 100.0}
            nums = set()
            flatten_numbers(fake, nums)
            for n in list(nums) + [float(base_inputs[k]) for k in NUMERIC_KEYS if k in base_inputs]:
                for scale in (1, 100):
                    for d in (0, 1, 2):
                        allowed.add(round(n * scale, d))
            found = []
            for tok in ANYNUM.findall(panel):
                try:
                    v = float(tok.replace(",", ""))
                except ValueError:
                    continue
                if not any(abs(v - a) < 0.051 for a in allowed):
                    found.append(tok)
            rec(3, "FRONTEND", "unexplained numbers in result panel", "PASS" if not found else "FAIL",
                "none" if not found else f"{sorted(set(found))[:12]} do not come from the API response or the form "
                f"(hardcoded or computed in the UI)")
            if isinstance(reasons, list) and reasons:
                shown = panel
                has_mult = any(x in shown for x in ("1.77", "0.33", "77%", "67%"))
                rec(3, "FRONTEND", "SENTINEL reasons shown", "PASS" if has_mult else "WARN",
                    "multiplier from API visible" if has_mult else "sentinel multipliers not visible in the panel; check how reasons are rendered")
            page.unroute(re.compile(r".*" + re.escape(args.predict_path) + r".*"))
        else:
            rec(3, "FRONTEND", "sentinel test", "WARN", "could not get a live API response to use as template")

        # (d) live UI vs live API for key scenarios
        for sc in [s for s in CFG["scenarios"] if s.get("ui")]:
            inputs = build_inputs(sc)
            p_api, st, body = api_prob(args, inputs)
            ui_p, ui_t, fired, sent_payload = run_scenario_in_ui(page, args, inputs)
            if fired:
                bad_sent = payload_mismatches(sent_payload, inputs)
                if bad_sent:
                    rec(3, "FRONTEND", f"{sc['id']} page sends typed values", "FAIL", "; ".join(bad_sent[:4]))
            if p_api is None:
                continue
            thr_api = num(pick(body, "threshold"))
            if not fired:
                rec(3, "FRONTEND", f"{sc['id']} request sent", "FAIL", "no request fired (UI not refreshing)")
            elif ui_p is None or abs(ui_p - p_api) > 0.0006:
                rec(3, "FRONTEND", f"{sc['id']} UI == API", "FAIL",
                    f"API={p_api:.2%}, UI={'nothing' if ui_p is None else f'{ui_p:.2%}'}")
            elif thr_api is not None and (ui_t is None or abs(ui_t - thr_api) > 0.0006):
                rec(3, "FRONTEND", f"{sc['id']} threshold UI == API", "FAIL",
                    f"API={thr_api:.2%}, UI={'nothing' if ui_t is None else f'{ui_t:.2%}'}")
            else:
                rec(3, "FRONTEND", f"{sc['id']} UI == API", "PASS", f"{ui_p:.2%}")

        # (e) stale-state sequences: A -> B -> A must give A's number again
        for seq in CFG["ui_sequences"]:
            bad = []
            for sid in seq:
                inputs = build_inputs(scenario_by_id(sid))
                p_api, _, _ = api_prob(args, inputs)
                ui_p, _, fired, _ = run_scenario_in_ui(page, args, inputs)
                if not fired:
                    bad.append(f"{sid}: no request")
                elif p_api is not None and (ui_p is None or abs(ui_p - p_api) > 0.0006):
                    bad.append(f"{sid}: API {p_api:.2%} vs UI {ui_p if ui_p is None else f'{ui_p:.2%}'}")
            rec(3, "FRONTEND", "stale state " + " > ".join(seq), "PASS" if not bad else "FAIL", "; ".join(bad) or "UI followed every change")

        # (f) invalid input: UI must show an error, not a stale result
        for sc in [s for s in CFG["scenarios"] if s.get("ui_error")]:
            before, _, _, _ = run_scenario_in_ui(page, args, base_inputs)
            run_scenario_in_ui(page, args, build_inputs(sc), wait_ms=600)
            err = page.locator('[data-testid="error-message"]')
            shown_err = err.count() > 0 and err.first.is_visible()
            after = first_pct(page.locator('[data-testid="prob-value"]').first.inner_text()) if page.locator('[data-testid="prob-value"]').count() else None
            stale = after is not None and before is not None and abs(after - before) < 1e-9
            rec(3, "FRONTEND", f"{sc['id']} shows an error", "PASS" if shown_err else ("FAIL" if stale else "WARN"),
                "error message visible" if shown_err else ("old result still displayed (stale)" if stale else "no error element found"))
        browser.close()


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------
def report():
    owners = {"ML": "ML model / threshold (Stages 3-8)", "BACKEND": "Backend wiring (FastAPI, Stage 9)",
              "FRONTEND": "Frontend (hardcoded values, wiring, stale UI, Stage 10)"}
    lines = ["# RiskLens diagnosis", "", "| Where | FAIL | WARN | PASS |", "|---|---|---|---|"]
    print("\n" + "=" * 70 + "\nWHERE IS THE BUG?\n" + "=" * 70)
    for o, label in owners.items():
        rs = [r for r in RESULTS if r["owner"] == o]
        f, w, p = (sum(1 for r in rs if r["status"] == s) for s in ("FAIL", "WARN", "PASS"))
        lines.append(f"| {label} | {f} | {w} | {p} |")
        print(f"{label:62} FAIL={f:3} WARN={w:3} PASS={p:3}")
    lines += ["", "## Failures and warnings", "", "| Layer | Owner | Check | Status | Detail |", "|---|---|---|---|---|"]
    for r in RESULTS:
        if r["status"] in ("FAIL", "WARN"):
            lines.append(f"| L{r['layer']} | {r['owner']} | {r['check']} | {r['status']} | {r['detail'].replace('|', '/')} |")
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "diagnosis.json").write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
    (out / "diagnosis.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved reports/diagnosis.md and reports/diagnosis.json")
    print("""
HOW TO READ THIS
  L1 fails (ML)       -> the model itself is wrong. Fix training/threshold. Backend and UI cannot be trusted yet.
  L1 ok, L2 fails     -> model is fine, backend wiring is wrong (field names, unit conversion, threshold source).
  L2 ok, L3 fails     -> API is fine, the website is wrong: SENTINEL/unexplained numbers = hardcoded or computed
                         in the UI; 'stale state' / 'no request' = the UI is not refreshing.
  L0 hits             -> hardcoded Stitch leftovers, found by file and line.""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,1,2,3")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--front", default="http://127.0.0.1:3000")
    ap.add_argument("--predict-path", default="/predict")
    ap.add_argument("--scorer-path", default="/risk-scorer")
    ap.add_argument("--payload-style", choices=["friendly", "columns", "full"], default=None,
                    help="friendly = keys like annual_income; columns = dataset names like AMT_INCOME_TOTAL")
    ap.add_argument("--model", default=None, help="default: models/risk_model.joblib, else models/model.joblib")
    ap.add_argument("--base-row", default=None, help="full sample applicant JSON (default: frontend/sample_request.json)")
    ap.add_argument("--threshold", type=float, default=0.67, help="used only if the model file stores no threshold")
    ap.add_argument("--days-per-year", type=float, default=365.0)
    ap.add_argument("--no-sync-goods", action="store_true", help="do not copy loan amount into AMT_GOODS_PRICE")
    ap.add_argument("--train-csv", default=str(ROOT / "data" / "raw" / "application_train.csv"))
    ap.add_argument("--validation-csv", default=str(ROOT / "reports" / "validation_predictions.csv"))
    ap.add_argument("--frontend-dir", default=str(ROOT / "frontend"))
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--show-schema", action="store_true")
    args = ap.parse_args()

    import platform
    print(f"Python {platform.python_version()} at {sys.executable}")
    try:
        import sklearn
        print(f"scikit-learn {sklearn.__version__} (must match the version used to train the model)")
    except ImportError:
        pass
    if args.show_schema:
        import requests
        for route in ("/schema", "/openapi.json"):
            r = requests.get(args.api.rstrip("/") + route, timeout=10)
            print(f"GET {route} -> {r.status_code}")
            try:
                data = r.json()
                if route == "/openapi.json":
                    data = {p: sorted(v) for p, v in data.get("paths", {}).items()}
                print(json.dumps(data, indent=2)[:3000])
            except Exception:  # noqa: BLE001
                print(r.text[:500])
        return
    global DAYS, BASE_ROW, SYNC_GOODS
    DAYS = args.days_per_year
    SYNC_GOODS = not args.no_sync_goods
    if args.model is None:
        mdir = ROOT / "models"
        args.model = str(mdir / "risk_model.joblib") if (mdir / "risk_model.joblib").exists() else str(mdir / "model.joblib")
    base_candidates = [Path(args.base_row)] if args.base_row else [ROOT / "frontend" / "sample_request.json", ROOT / "models" / "sample_request.json"]
    for cand in base_candidates:
        if cand.exists():
            BASE_ROW = json.loads(cand.read_text(encoding="utf-8"))
            print(f"Base row: {cand} ({len(BASE_ROW)} columns)")
            break
    if args.payload_style is None:
        args.payload_style = "full" if BASE_ROW else "friendly"
    print(f"Payload style: {args.payload_style}; days per year: {DAYS}; sync goods price: {SYNC_GOODS}")
    layers = {int(x) for x in args.layers.split(",") if x.strip()}
    m = None
    try:
        if 0 in layers:
            layer0(args)
        if 1 in layers or 2 in layers:
            m = layer1(args)
        if 2 in layers:
            layer2(args, m)
        if 3 in layers:
            layer3(args)
    except KeyboardInterrupt:
        print("\nStopped by user. Showing what was collected so far.")
    report()


if __name__ == "__main__":
    main()