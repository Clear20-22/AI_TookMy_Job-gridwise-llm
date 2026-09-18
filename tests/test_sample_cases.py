"""
Integration tests: run all 10 sample cases through the optimizer.

For each case the expected directive_interpretation is hardcoded (bypassing
the LLM) so we can validate the optimizer and physical invariants
deterministically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.optimizer import solve_schedule
from app.services.guardrails import directives_to_optimizer_format

# ---------------------------------------------------------------------------
# Load sample cases
# ---------------------------------------------------------------------------

CASES_PATH = Path(__file__).resolve().parent.parent / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"

with open(CASES_PATH, encoding="utf-8") as f:
    _DATA = json.load(f)

CASES: list[dict[str, Any]] = _DATA["cases"]

# Create (id, case) pairs for parametrize
CASE_IDS = [c["id"] for c in CASES]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hours_list(case_input: dict) -> list[dict]:
    """Return the hours list from the case input, sorted by hour."""
    return sorted(case_input["hours"], key=lambda e: e["hour"])


def _battery(case_input: dict) -> dict:
    return case_input["battery"]


def _expected_directives_to_optimizer(expected_output: dict) -> list[dict]:
    """Convert expected directive_interpretation to optimizer format."""
    directives = expected_output["directive_interpretation"]
    return directives_to_optimizer_format(directives)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
class TestSampleCase:
    """Run each sample case through the optimizer and verify invariants."""

    @pytest.fixture(autouse=True)
    def _solve(self, case: dict):
        """Solve using hardcoded expected directives (bypass LLM)."""
        inp = case["input"]
        exp = case["expected_output"]

        hours = _hours_list(inp)
        battery = _battery(inp)
        opt_directives = _expected_directives_to_optimizer(exp)

        self.case_id = case["id"]
        self.hours = hours
        self.battery = battery
        self.expected = exp
        self.result = solve_schedule(hours, battery, opt_directives)
        self.plan = self.result["hourly_plan"]
        self.by_hour = {e["hour"]: e for e in self.plan}

    def test_24_entries(self, case):
        assert len(self.plan) == 24, f"{self.case_id}: expected 24 plan entries"

    def test_energy_balance(self, case):
        """grid + solar + discharge == demand + charge for every hour."""
        for entry in self.plan:
            h = entry["hour"]
            demand = self.hours[h]["demand_kwh"]
            supply = entry["grid_kwh"] + entry["solar_used_kwh"]
            if entry["battery_action"] == "discharge":
                supply += entry["battery_kwh"]
            sink = demand
            if entry["battery_action"] == "charge":
                sink += entry["battery_kwh"]
            assert abs(supply - sink) < 0.02, (
                f"{self.case_id} hour {h}: supply={supply:.4f}, sink={sink:.4f}"
            )

    def test_end_of_day_neutrality(self, case):
        """Battery energy at hour 23 must equal initial_energy_kwh."""
        final = self.plan[23]["battery_energy_after_kwh"]
        initial = self.battery["initial_energy_kwh"]
        assert abs(final - initial) < 0.02, (
            f"{self.case_id}: final_battery={final}, initial={initial}"
        )

    def test_battery_bounds(self, case):
        """Battery energy stays within [minimum, capacity]."""
        cap = self.battery["capacity_kwh"]
        # Note: effective minimum may be higher due to directives — we check
        # against the base minimum here. Directive-specific checks are below.
        bmin = self.battery["minimum_energy_kwh"]
        for entry in self.plan:
            e = entry["battery_energy_after_kwh"]
            assert e >= bmin - 0.02, (
                f"{self.case_id} hour {entry['hour']}: energy={e} < min={bmin}"
            )
            assert e <= cap + 0.02, (
                f"{self.case_id} hour {entry['hour']}: energy={e} > cap={cap}"
            )

    def test_solar_not_exceeding_available(self, case):
        """Solar used never exceeds the forecast (before directive reduction)."""
        for entry in self.plan:
            h = entry["hour"]
            avail = self.hours[h]["solar_kwh"]
            assert entry["solar_used_kwh"] <= avail + 0.02, (
                f"{self.case_id} hour {h}: solar_used={entry['solar_used_kwh']} > available={avail}"
            )

    def test_no_simultaneous_charge_discharge(self, case):
        for entry in self.plan:
            if entry["battery_action"] == "idle":
                assert entry["battery_kwh"] == 0

    def test_cost_within_tolerance(self, case):
        """Our optimal cost should be ≤ expected cost + tolerance.

        The solver may find a different-but-equivalent optimal schedule,
        so we allow a small tolerance.  Our cost should never be *worse*
        than the reference by more than a tiny amount.
        """
        expected_cost = self.expected["total_cost_bdt"]
        actual_cost = self.result["total_cost_bdt"]
        # Allow up to 0.5 BDT tolerance (floating point + alt optima)
        assert actual_cost <= expected_cost + 0.5, (
            f"{self.case_id}: actual_cost={actual_cost} > expected={expected_cost}+0.5"
        )

    def test_directive_compliance(self, case):
        """Verify specific directive constraints are respected."""
        for d in self.expected["directive_interpretation"]:
            if not d["applies"]:
                continue
            adj = d["structured_adjustment"]
            dtype = d["directive_type"]

            if dtype == "solar_reduction":
                factor = adj["factor"]
                for h in adj["hours"]:
                    max_solar = self.hours[h]["solar_kwh"] * factor
                    actual = self.by_hour[h]["solar_used_kwh"]
                    assert actual <= max_solar + 0.02, (
                        f"{self.case_id} hour {h}: solar={actual} > reduced_max={max_solar}"
                    )

            elif dtype == "no_charge_window":
                for h in adj["hours"]:
                    entry = self.by_hour[h]
                    assert entry["battery_action"] != "charge", (
                        f"{self.case_id} hour {h}: should not charge"
                    )

            elif dtype == "no_discharge_window":
                for h in adj["hours"]:
                    entry = self.by_hour[h]
                    assert entry["battery_action"] != "discharge", (
                        f"{self.case_id} hour {h}: should not discharge"
                    )

            elif dtype == "minimum_battery_reserve":
                min_e = adj["minimum_energy_kwh"]
                for h in adj["hours"]:
                    actual_e = self.by_hour[h]["battery_energy_after_kwh"]
                    assert actual_e >= min_e - 0.02, (
                        f"{self.case_id} hour {h}: battery={actual_e} < reserve={min_e}"
                    )

            elif dtype == "max_grid_window":
                max_g = adj["max_grid_kwh"]
                for h in adj["hours"]:
                    actual_g = self.by_hour[h]["grid_kwh"]
                    assert actual_g <= max_g + 0.02, (
                        f"{self.case_id} hour {h}: grid={actual_g} > max={max_g}"
                    )
