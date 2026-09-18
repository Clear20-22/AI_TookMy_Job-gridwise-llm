"""
Unit tests for the guardrail validator.
"""

from __future__ import annotations

import pytest

from app.services.guardrails import validate_directives, directives_to_optimizer_format


# ---------------------------------------------------------------------------
# Valid directives pass through unchanged
# ---------------------------------------------------------------------------

class TestValidDirectives:
    """Valid directives should survive guardrail validation intact."""

    def test_valid_solar_reduction(self):
        raw = [{
            "note_index": 0,
            "applies": True,
            "directive_type": "solar_reduction",
            "structured_adjustment": {"hours": [12, 13], "factor": 0.25},
            "explanation": "Panel cleaning",
        }]
        result = validate_directives(raw, battery_capacity_kwh=200, battery_minimum_kwh=40)
        assert len(result) == 1
        assert result[0]["directive_type"] == "solar_reduction"
        assert result[0]["applies"] is True
        assert result[0]["structured_adjustment"]["factor"] == 0.25
        assert result[0]["structured_adjustment"]["hours"] == [12, 13]

    def test_valid_no_charge_window(self):
        raw = [{
            "note_index": 0,
            "applies": True,
            "directive_type": "no_charge_window",
            "structured_adjustment": {"hours": [2, 3, 4]},
            "explanation": "Maintenance",
        }]
        result = validate_directives(raw, 200, 40)
        assert result[0]["directive_type"] == "no_charge_window"
        assert result[0]["structured_adjustment"]["hours"] == [2, 3, 4]

    def test_valid_no_discharge_window(self):
        raw = [{
            "note_index": 0,
            "applies": True,
            "directive_type": "no_discharge_window",
            "structured_adjustment": {"hours": [18, 19]},
            "explanation": "Protection test",
        }]
        result = validate_directives(raw, 200, 40)
        assert result[0]["directive_type"] == "no_discharge_window"

    def test_valid_minimum_battery_reserve(self):
        raw = [{
            "note_index": 0,
            "applies": True,
            "directive_type": "minimum_battery_reserve",
            "structured_adjustment": {"hours": [18, 19, 20], "minimum_energy_kwh": 100},
            "explanation": "Emergency reserve",
        }]
        result = validate_directives(raw, 200, 40)
        assert result[0]["structured_adjustment"]["minimum_energy_kwh"] == 100

    def test_valid_max_grid_window(self):
        raw = [{
            "note_index": 0,
            "applies": True,
            "directive_type": "max_grid_window",
            "structured_adjustment": {"hours": [17, 18, 19], "max_grid_kwh": 150},
            "explanation": "Feeder limit",
        }]
        result = validate_directives(raw, 200, 40)
        assert result[0]["structured_adjustment"]["max_grid_kwh"] == 150

    def test_valid_no_op(self):
        raw = [{
            "note_index": 0,
            "applies": False,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": "Not relevant",
        }]
        result = validate_directives(raw, 200, 40)
        assert result[0]["directive_type"] == "no_op"
        assert result[0]["applies"] is False
        assert result[0]["structured_adjustment"] is None


# ---------------------------------------------------------------------------
# Malformed directives degrade to no_op
# ---------------------------------------------------------------------------

class TestMalformedDirectives:
    """Invalid directives should be demoted to no_op."""

    def test_unknown_type_becomes_no_op(self):
        raw = [{
            "note_index": 0,
            "applies": True,
            "directive_type": "unknown_type",
            "structured_adjustment": {"hours": [1]},
            "explanation": "bad",
        }]
        result = validate_directives(raw, 200, 40)
        assert result[0]["directive_type"] == "no_op"
        assert result[0]["applies"] is False

    def test_missing_structured_adjustment(self):
        raw = [{
            "note_index": 0,
            "applies": True,
            "directive_type": "solar_reduction",
            "structured_adjustment": None,
            "explanation": "no adj",
        }]
        result = validate_directives(raw, 200, 40)
        assert result[0]["directive_type"] == "no_op"

    def test_missing_hours(self):
        raw = [{
            "note_index": 0,
            "applies": True,
            "directive_type": "no_charge_window",
            "structured_adjustment": {},
            "explanation": "no hours",
        }]
        result = validate_directives(raw, 200, 40)
        assert result[0]["directive_type"] == "no_op"

    def test_invalid_hour_values(self):
        raw = [{
            "note_index": 0,
            "applies": True,
            "directive_type": "no_charge_window",
            "structured_adjustment": {"hours": [25, -1]},
            "explanation": "out of range",
        }]
        result = validate_directives(raw, 200, 40)
        assert result[0]["directive_type"] == "no_op"

    def test_solar_reduction_missing_factor(self):
        raw = [{
            "note_index": 0,
            "applies": True,
            "directive_type": "solar_reduction",
            "structured_adjustment": {"hours": [12]},
            "explanation": "no factor",
        }]
        result = validate_directives(raw, 200, 40)
        assert result[0]["directive_type"] == "no_op"

    def test_min_reserve_missing_kwh(self):
        raw = [{
            "note_index": 0,
            "applies": True,
            "directive_type": "minimum_battery_reserve",
            "structured_adjustment": {"hours": [18]},
            "explanation": "no kwh",
        }]
        result = validate_directives(raw, 200, 40)
        assert result[0]["directive_type"] == "no_op"

    def test_max_grid_missing_kwh(self):
        raw = [{
            "note_index": 0,
            "applies": True,
            "directive_type": "max_grid_window",
            "structured_adjustment": {"hours": [18]},
            "explanation": "no kwh",
        }]
        result = validate_directives(raw, 200, 40)
        assert result[0]["directive_type"] == "no_op"


# ---------------------------------------------------------------------------
# Clamping behaviour
# ---------------------------------------------------------------------------

class TestClamping:
    """Values out of range should be clamped, not rejected."""

    def test_factor_clamped_to_0_1(self):
        raw = [{
            "note_index": 0,
            "applies": True,
            "directive_type": "solar_reduction",
            "structured_adjustment": {"hours": [10], "factor": 1.5},
            "explanation": "over 1",
        }]
        result = validate_directives(raw, 200, 40)
        assert result[0]["structured_adjustment"]["factor"] == 1.0

    def test_factor_clamped_to_zero(self):
        raw = [{
            "note_index": 0,
            "applies": True,
            "directive_type": "solar_reduction",
            "structured_adjustment": {"hours": [10], "factor": -0.5},
            "explanation": "negative",
        }]
        result = validate_directives(raw, 200, 40)
        assert result[0]["structured_adjustment"]["factor"] == 0.0

    def test_reserve_clamped_to_capacity(self):
        raw = [{
            "note_index": 0,
            "applies": True,
            "directive_type": "minimum_battery_reserve",
            "structured_adjustment": {"hours": [18], "minimum_energy_kwh": 999},
            "explanation": "over capacity",
        }]
        result = validate_directives(raw, battery_capacity_kwh=200, battery_minimum_kwh=40)
        assert result[0]["structured_adjustment"]["minimum_energy_kwh"] == 200

    def test_hours_deduplicated_and_sorted(self):
        raw = [{
            "note_index": 0,
            "applies": True,
            "directive_type": "no_charge_window",
            "structured_adjustment": {"hours": [5, 3, 5, 3, 4]},
            "explanation": "duplicates",
        }]
        result = validate_directives(raw, 200, 40)
        assert result[0]["structured_adjustment"]["hours"] == [3, 4, 5]


# ---------------------------------------------------------------------------
# Optimizer format conversion
# ---------------------------------------------------------------------------

class TestOptimizerConversion:
    """Test directive → optimizer format conversion."""

    def test_no_op_excluded(self):
        directives = [
            {
                "note_index": 0,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "distractor",
            },
            {
                "note_index": 1,
                "applies": True,
                "directive_type": "no_charge_window",
                "structured_adjustment": {"hours": [2, 3]},
                "explanation": "maint",
            },
        ]
        opt = directives_to_optimizer_format(directives)
        assert len(opt) == 1
        assert opt[0]["type"] == "no_charge_window"
        assert opt[0]["hours"] == [2, 3]

    def test_solar_reduction_flattened(self):
        directives = [{
            "note_index": 0,
            "applies": True,
            "directive_type": "solar_reduction",
            "structured_adjustment": {"hours": [12, 13], "factor": 0.25},
            "explanation": "cleaning",
        }]
        opt = directives_to_optimizer_format(directives)
        assert opt[0] == {"type": "solar_reduction", "hours": [12, 13], "factor": 0.25}
