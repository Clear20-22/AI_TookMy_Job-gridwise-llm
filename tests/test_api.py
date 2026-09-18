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
    """End-to-end test on sample case 01 with mock LLM returning parsed directives."""
    with open(SAMPLE_DIR / "sample-case-01.json", encoding="utf-8") as f:
        sample_data = json.load(f)

    # Mock the LLM to return the expected directive interpretation
    mock_interpretation = sample_data["expected_directive_interpretation"]
    with patch("app.main.interpret_directives", return_value=mock_interpretation):
        response = client.post(
            "/optimize-energy",
            json={
                "battery": sample_data["battery"],
                "hourly": sample_data["hourly"],
                "operator_notes": sample_data["operator_notes"],
            },
        )

    assert response.status_code == 200
    data = response.json()

    # Check response structure
    assert "hourly_schedule" in data
    assert "cost_summary" in data
    assert "battery_metrics" in data
    assert "solver_metadata" in data
    assert "directive_interpretation" in data

    assert len(data["hourly_schedule"]) == 24
    assert data["solver_metadata"]["status"] == "optimal"

    # Energy balance and cost within tolerance
    expected_cost = sample_data["expected_cost_summary"]["total_cost"]
    assert abs(data["cost_summary"]["total_cost"] - expected_cost) <= 0.05 * expected_cost + 1.0
