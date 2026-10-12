# ==============================================================================
# RiskLens Credit Risk Assessment Engine - Container Definition
# Multi-stage slim build for FastAPI inference backend and static frontend
# ==============================================================================

FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install minimal OS dependencies for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application modules, models, and static frontend
COPY calibrated_pipeline.py .
COPY backend/ ./backend/
COPY models/ ./models/
COPY reports/ ./reports/
COPY frontend/ ./frontend/
COPY .env.example ./.env.example

# Expose FastAPI backend port
EXPOSE 8000

# Docker healthcheck pinging FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production ASGI server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
