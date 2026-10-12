import os
import sys
from pathlib import Path

# Ensure project root is in sys.path so backend and calibrated_pipeline are importable
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Export FastAPI app for Vercel Serverless Function runtime
from backend.main import app
