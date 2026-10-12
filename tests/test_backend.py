import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "calibrated" in data
    assert data["calibrated"] is True


def test_schema_endpoint():
    response = client.get("/schema")
    assert response.status_code == 200
    data = response.json()
    assert "fields" in data
    assert "annual_income" in data["fields"]
    assert "ext_source_1" in data["fields"]
    assert "education" in data["fields"]


def test_threshold_endpoint():
    response = client.get("/threshold")
    assert response.status_code == 200
    data = response.json()
    assert "chosen_threshold" in data or "threshold" in data


def test_insights_endpoint():
    response = client.get("/insights")
    assert response.status_code == 200
    data = response.json()
    assert "raising_risk" in data or "odds_ratios" in data or "error" not in data


def test_predict_typical_applicant():
    payload = {
        "annual_income": 147000,
        "loan_amount": 500000,
        "annuity": 25000,
        "age": 43,
        "years_employed": 6,
        "education": "Secondary / secondary special",
        "income_type": "Working",
        "occupation": "Laborers",
        "organisation_type": "Business Entity Type 3",
        "ext_source_1": 0.5,
        "ext_source_2": 0.5,
        "ext_source_3": 0.5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "probability_of_default" in data
    assert 0.0 <= data["probability_of_default"] <= 1.0
    assert data["status"] in ("Approved", "Rejected")
    assert data["threshold_used"] == 0.16
    assert "dti_ratio" in data
    assert data["dti_ratio"] is not None
    # 25000 / (147000 / 12) = 2.0408
    assert round(data["dti_ratio"], 2) == 2.04
    assert "top_3_reasons" in data
    assert len(data["top_3_reasons"]) <= 3
    for reason in data["top_3_reasons"]:
        assert "odds_multiplier" in reason
        assert reason["odds_multiplier"] > 0
        assert "direction" in reason


def test_predict_reject_negative_income():
    payload = {"annual_income": -5000}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_reject_invalid_age():
    payload = {"age": 150}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

    payload_neg = {"age": -5}
    response_neg = client.post("/predict", json=payload_neg)
    assert response_neg.status_code == 422


def test_predict_reject_invalid_ext_source():
    payload = {"ext_source_1": 1.5}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

    payload_neg = {"ext_source_2": -0.2}
    response_neg = client.post("/predict", json=payload_neg)
    assert response_neg.status_code == 422


def test_predict_warnings_on_out_of_range():
    payload = {
        "annual_income": 60000,
        "loan_amount": 5000,
        "annuity": 500,
        "age": 38,
        "years_employed": 8.5,
        "education": "Higher education",
        "income_type": "Commercial associate",
        "occupation": "High skill tech staff",
        "organisation_type": "Business Entity Type 3",
        "ext_source_1": 0.85,
        "ext_source_2": 0.85,
        "ext_source_3": 0.85,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "warnings" in data
    assert len(data["warnings"]) > 0
