"""
FastAPI endpoint tests for GridWise LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_cases"


def test_health():
    """Health check returns 200 and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_optimize_energy_missing_fields():
    """Request missing required fields should return 422."""
    response = client.post("/optimize-energy", json={})
    assert response.status_code == 422


def test_optimize_energy_invalid_hours():
    """Request with fewer than 24 hours should return 422."""
    response = client.post(
        "/optimize-energy",
        json={
            "battery": {"capacity_kwh": 500, "max_charge_kw": 125, "max_discharge_kw": 125, "efficiency": 0.9},
            "hourly": [{"hour": 0, "demand_kwh": 100, "solar_forecast_kwh": 50, "tariff_rate": 0.1}],
            "operator_notes": [],
        },
    )
    assert response.status_code == 422


def test_optimize_energy_end_to_end_sample_01():
    """End-to-end test on official sample case 01 with mock LLM returning parsed directives."""
    cases_path = Path(__file__).resolve().parent.parent / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
    with open(cases_path, encoding="utf-8") as f:
        cases_data = json.load(f)

    sample_01 = cases_data["cases"][0]
    c_input = sample_01["input"]
    c_expected = sample_01["expected_output"]

    # Mock the LLM to return the expected directive interpretation
    mock_interpretation = c_expected["directive_interpretation"]
    with patch("app.main.interpret_notes", return_value=mock_interpretation):
        response = client.post(
            "/optimize-energy",
            json=c_input,
        )

    assert response.status_code == 200
    data = response.json()

    # Check official BUP CSE Fest response structure
    assert data["scenario_id"] == "SAMPLE-01"
    assert "directive_interpretation" in data
    assert "hourly_plan" in data
    assert "total_cost_bdt" in data
    assert "total_grid_kwh" in data
    assert "peak_grid_kwh" in data
    assert "plan_summary" in data

    assert len(data["hourly_plan"]) == 24
    assert len(data["directive_interpretation"]) == 2

    # Cost within tolerance of organizer reference
    expected_cost = c_expected["total_cost_bdt"]
    assert abs(data["total_cost_bdt"] - expected_cost) <= 0.05


def test_optimize_energy_graceful_llm_failure():
    """When LLM fails completely, all notes degrade to no_op and optimizer still produces a valid schedule."""
    cases_path = Path(__file__).resolve().parent.parent / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
    with open(cases_path, encoding="utf-8") as f:
        cases_data = json.load(f)

    sample_01 = cases_data["cases"][0]
    with patch("app.main.interpret_notes", side_effect=RuntimeError("Provider offline")):
        response = client.post("/optimize-energy", json=sample_01["input"])

    assert response.status_code == 200
    data = response.json()
    assert len(data["directive_interpretation"]) == len(sample_01["input"]["operator_notes"])
    for d in data["directive_interpretation"]:
        assert d["directive_type"] == "no_op"
        assert d["applies"] is False
        assert d["structured_adjustment"] is None
    assert len(data["hourly_plan"]) == 24
    assert data["total_cost_bdt"] > 0

