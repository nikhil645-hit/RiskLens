#!/usr/bin/env python3
"""
RiskLens stage checker: shows which of the 13 stages have REAL evidence on disk.

An AI agent saying "Stage 8 done" is not evidence. A saved model file that loads is.
This only checks that expected outputs exist and look sane. It cannot judge quality,
but it stops you from building on stages that were never finished.

Run from the project root:
  .\\.venv\\Scripts\\python.exe tests\\check_stages.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "reports"


def exists(p: Path, min_bytes: int = 1) -> bool:
    return p.exists() and p.is_file() and p.stat().st_size >= min_bytes


def load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def text_of(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def frontend_files():
    fdir = ROOT / "frontend"
    skip = {"node_modules", ".next", "dist", "build", ".git", "out"}
    exts = {".html", ".js", ".jsx", ".ts", ".tsx", ".vue"}
    if not fdir.exists():
        return []
    return [p for p in fdir.rglob("*")
            if p.is_file() and p.suffix in exts and not any(x in p.parts for x in skip)]


def stage0():
    missing = [n for n in [".gitignore", "requirements.txt", "PROJECT_RULES.md", "README.md"] if not exists(ROOT / n)]
    missing += [f"{d}/" for d in ["data/raw", "src", "models", "reports", "backend", "frontend", "design"] if not (ROOT / d).is_dir()]
    return missing


def stage1():
    figs = list((R / "figures").glob("*.png")) if (R / "figures").exists() else []
    return [] if len(figs) >= 3 else [f"reports/figures needs 3+ PNG charts (found {len(figs)})"]


def stage2():
    return [] if exists(ROOT / "src" / "data.py", 200) else ["src/data.py"]


def stage3():
    miss = [] if exists(ROOT / "src" / "features.py", 200) else ["src/features.py"]
    d = load_json(R / "encoder_comparison.json")
    if d is None:
        return miss + ["reports/encoder_comparison.json"]
    blob = json.dumps(d).lower()
    for enc in ("onehot", "target", "loo"):
        if enc not in blob and not (enc == "onehot" and "one-hot" in blob) and not (enc == "loo" and "leave" in blob):
            miss.append(f"encoder '{enc}' missing in encoder_comparison.json")
    return miss


def stage4():
    d = load_json(R / "encoder_comparison.json")
    if d is None:
        return ["reports/encoder_comparison.json"]
    return [] if "embed" in json.dumps(d).lower() else ["embedding result missing in encoder_comparison.json"]


def stage5():
    miss = [f"reports/{n}" for n in ("roc_points.json", "pr_points.json", "validation_predictions.csv") if not exists(R / n)]
    return miss


def stage6():
    miss = [f"reports/{n}" for n in ("odds_ratios.json",) if not exists(R / n)]
    if not exists(ROOT / "src" / "explain.py", 100):
        miss.append("src/explain.py")
    return miss


def stage7():
    return [] if exists(R / "benchmark.json") else ["reports/benchmark.json"]


def stage8():
    miss = []
    if not exists(R / "final_test_metrics.json"):
        miss.append("reports/final_test_metrics.json")
    mp = ROOT / "models" / "risk_model.joblib"
    if not exists(mp, 1000):
        miss.append("models/risk_model.joblib (not found)")
    else:
        try:
            import joblib
            b = joblib.load(mp)
            if not (isinstance(b, dict) and "pipeline" in b and "threshold" in b):
                miss.append("risk_model.joblib must be a dict with keys 'pipeline' and 'threshold'")
        except Exception as e:  # noqa: BLE001
            miss.append(f"risk_model.joblib does not load: {type(e).__name__}: {str(e)[:100]}")
    return miss


def stage9():
    main = ROOT / "backend" / "main.py"
    if not exists(main, 200):
        return ["backend/main.py"]
    t = text_of(main)
    miss = [f"route {r}" for r in ("/health", "/schema", "/predict", "/threshold", "/insights") if r not in t]
    if not list((ROOT / "backend").rglob("test*.py")) and not list((ROOT / "tests").glob("test_*.py")):
        miss.append("pytest tests for the backend")
    return miss


def stage10():
    files = frontend_files()
    if not files:
        return ["no frontend source files"]
    miss = []
    if not any(re.search(r"fetch\(|axios|XMLHttpRequest", text_of(p)) for p in files):
        miss.append("no file in frontend/ ever calls the API (no fetch/axios)")
    banned = ["99.2", "814", "SR 11-7", "Deploy Policy", "Dossier", "Synthetic", "0x8a", "v2.4", "STATE VIEW"]
    hits = sorted({b for p in files for b in banned if b in text_of(p)})
    if hits:
        miss.append("Stitch mock leftovers still present: " + ", ".join(hits))
    return miss


def stage11():
    miss = []
    if not list(ROOT.rglob("Dockerfile")) or all(".venv" in str(p) or "node_modules" in str(p) for p in ROOT.rglob("Dockerfile")):
        miss.append("Dockerfile")
    if not list(p for p in ROOT.rglob(".env.example") if "node_modules" not in str(p)):
        miss.append(".env.example")
    return miss


def stage12():
    rd = text_of(ROOT / "README.md")
    if len(rd) < 1500:
        return ["README.md is still a skeleton (under 1500 characters)"]
    return [] if re.search(r"ROC-?AUC", rd, re.I) else ["README has no results (ROC-AUC) yet"]


STAGES = [
    (0, "Setup", stage0), (1, "EDA", stage1), (2, "Cleaning and split", stage2),
    (3, "Encoding comparison", stage3), (4, "Embedding encoder", stage4),
    (5, "Evaluation and threshold", stage5), (6, "Explainability", stage6),
    (7, "LightGBM benchmark", stage7), (8, "Final test and saved model", stage8),
    (9, "FastAPI backend", stage9), (10, "Frontend wired to API", stage10),
    (11, "Polish and deploy prep", stage11), (12, "Docs and CV", stage12),
]


def main():
    print(f"Project: {ROOT}\n")
    first_bad = None
    rows = []
    for n, name, fn in STAGES:
        missing = fn()
        status = "DONE" if not missing else "NOT DONE"
        rows.append((n, name, status, missing))
        if missing and first_bad is None:
            first_bad = (n, name)
        print(f"Stage {n:>2}  {name:28} {status}")
        for m in missing:
            print(f"           - missing: {m}")
    print()
    if first_bad is None:
        print("All stages have evidence on disk. Quality still needs the tests in diagnose.py.")
    else:
        print(f"FIRST INCOMPLETE STAGE: {first_bad[0]} ({first_bad[1]}).")
        print("Do not build on later stages until this one has real evidence.")
    (R).mkdir(exist_ok=True)
    (R / "stage_status.json").write_text(json.dumps(
        [{"stage": n, "name": nm, "status": st, "missing": ms} for n, nm, st, ms in rows], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()