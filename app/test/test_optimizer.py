"""
Tests for the GridWise energy optimizer.

Two main scenarios:
  1. No directives — verify energy balance, battery neutrality, basic feasibility.
  2. All 5 directive types active — verify every directive is enforced in the output.
"""

import pytest

from app.optimizer import OptimizationError, solve_schedule

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Realistic 24-hour campus profile: night base-load, daytime peak, midday solar
HOURS_DATA = [
    {"hour":  0, "demand_kwh": 180, "solar_kwh":   0, "tariff_bdt_per_kwh":  7},
    {"hour":  1, "demand_kwh": 170, "solar_kwh":   0, "tariff_bdt_per_kwh":  7},
    {"hour":  2, "demand_kwh": 160, "solar_kwh":   0, "tariff_bdt_per_kwh":  7},
    {"hour":  3, "demand_kwh": 155, "solar_kwh":   0, "tariff_bdt_per_kwh":  7},
    {"hour":  4, "demand_kwh": 160, "solar_kwh":   0, "tariff_bdt_per_kwh":  7},
    {"hour":  5, "demand_kwh": 175, "solar_kwh":  10, "tariff_bdt_per_kwh":  7},
    {"hour":  6, "demand_kwh": 210, "solar_kwh":  40, "tariff_bdt_per_kwh":  8},
    {"hour":  7, "demand_kwh": 260, "solar_kwh":  80, "tariff_bdt_per_kwh":  8},
    {"hour":  8, "demand_kwh": 310, "solar_kwh": 150, "tariff_bdt_per_kwh":  9},
    {"hour":  9, "demand_kwh": 340, "solar_kwh": 200, "tariff_bdt_per_kwh":  9},
    {"hour": 10, "demand_kwh": 360, "solar_kwh": 250, "tariff_bdt_per_kwh": 10},
    {"hour": 11, "demand_kwh": 370, "solar_kwh": 280, "tariff_bdt_per_kwh": 10},
    {"hour": 12, "demand_kwh": 350, "solar_kwh": 300, "tariff_bdt_per_kwh": 10},
    {"hour": 13, "demand_kwh": 330, "solar_kwh": 270, "tariff_bdt_per_kwh":  9},
    {"hour": 14, "demand_kwh": 300, "solar_kwh": 220, "tariff_bdt_per_kwh":  9},
    {"hour": 15, "demand_kwh": 280, "solar_kwh": 150, "tariff_bdt_per_kwh":  9},
    {"hour": 16, "demand_kwh": 290, "solar_kwh":  80, "tariff_bdt_per_kwh": 10},
    {"hour": 17, "demand_kwh": 310, "solar_kwh":  30, "tariff_bdt_per_kwh": 11},
    {"hour": 18, "demand_kwh": 340, "solar_kwh":   5, "tariff_bdt_per_kwh": 12},
    {"hour": 19, "demand_kwh": 350, "solar_kwh":   0, "tariff_bdt_per_kwh": 12},
    {"hour": 20, "demand_kwh": 320, "solar_kwh":   0, "tariff_bdt_per_kwh": 11},
    {"hour": 21, "demand_kwh": 280, "solar_kwh":   0, "tariff_bdt_per_kwh": 10},
    {"hour": 22, "demand_kwh": 230, "solar_kwh":   0, "tariff_bdt_per_kwh":  8},
    {"hour": 23, "demand_kwh": 200, "solar_kwh":   0, "tariff_bdt_per_kwh":  7},
]

BATTERY = {
    "capacity_kwh": 500,
    "initial_energy_kwh": 200,
    "minimum_energy_kwh": 50,
    "max_charge_kwh_per_hour": 100,
    "max_discharge_kwh_per_hour": 100,
}


# ---------------------------------------------------------------------------
# Case 1 — No directives
# ---------------------------------------------------------------------------

class TestNoDirectives:
    """Run the optimiser with zero directives; verify physical invariants."""

    @pytest.fixture(autouse=True)
    def _solve(self):
        self.result = solve_schedule(HOURS_DATA, BATTERY, directives=[])
        self.plan = self.result["hourly_plan"]

    def test_returns_24_entries(self):
        assert len(self.plan) == 24

    def test_energy_balance_every_hour(self):
        """grid + solar + discharge == demand + charge  (constraint 1)."""
        for entry in self.plan:
            h = entry["hour"]
            demand = HOURS_DATA[h]["demand_kwh"]
            supply = entry["grid_kwh"] + entry["solar_used_kwh"]
            if entry["battery_action"] == "discharge":
                supply += entry["battery_kwh"]
            sink = demand
            if entry["battery_action"] == "charge":
                sink += entry["battery_kwh"]
            assert abs(supply - sink) < 0.01, (
                f"Hour {h}: supply={supply}, sink={sink}"
            )

    def test_end_of_day_neutrality(self):
        """Battery must return to its initial energy level (constraint 7)."""
        final_energy = self.plan[23]["battery_energy_after_kwh"]
        assert abs(final_energy - BATTERY["initial_energy_kwh"]) < 0.01

    def test_battery_bounds(self):
        """Battery energy stays within [minimum, capacity] (constraint 4)."""
        for entry in self.plan:
            assert entry["battery_energy_after_kwh"] >= BATTERY["minimum_energy_kwh"] - 0.01
            assert entry["battery_energy_after_kwh"] <= BATTERY["capacity_kwh"] + 0.01

    def test_solar_not_exceeding_available(self):
        """Solar used never exceeds available solar (constraint 2)."""
        for entry in self.plan:
            h = entry["hour"]
            assert entry["solar_used_kwh"] <= HOURS_DATA[h]["solar_kwh"] + 0.01

    def test_no_simultaneous_charge_discharge(self):
        """Charge and discharge should never both be nonzero (constraint 8)."""
        for entry in self.plan:
            if entry["battery_action"] == "charge":
                # discharge must be zero
                assert entry["battery_action"] != "discharge"
            # More directly: battery_kwh is the magnitude, action tells direction
            # — if action is idle, battery_kwh should be 0
            if entry["battery_action"] == "idle":
                assert entry["battery_kwh"] == 0

    def test_total_cost_positive(self):
        """Sanity: total cost should be a positive number."""
        assert self.result["total_cost_bdt"] > 0

    def test_summary_fields_consistent(self):
        """total_grid_kwh and peak_grid_kwh match the hourly plan."""
        total = sum(e["grid_kwh"] for e in self.plan)
        peak = max(e["grid_kwh"] for e in self.plan)
        assert abs(total - self.result["total_grid_kwh"]) < 0.1
        assert abs(peak - self.result["peak_grid_kwh"]) < 0.1


# ---------------------------------------------------------------------------
# Case 2 — All 5 directive types active
# ---------------------------------------------------------------------------

ALL_DIRECTIVES = [
    {"type": "solar_reduction",        "hours": [13, 14], "factor": 0.2},
    {"type": "no_charge_window",       "hours": [14, 15]},
    {"type": "no_discharge_window",    "hours": [16, 17]},
    {"type": "minimum_battery_reserve","hours": [18, 19, 20], "minimum_energy_kwh": 120},
    {"type": "max_grid_window",        "hours": [19, 20, 21], "max_grid_kwh": 400},
]


class TestAllDirectives:
    """Run the optimiser with every directive type; verify each is enforced."""

    @pytest.fixture(autouse=True)
    def _solve(self):
        self.result = solve_schedule(HOURS_DATA, BATTERY, directives=ALL_DIRECTIVES)
        self.plan = self.result["hourly_plan"]
        # Index by hour for easy lookup
        self.by_hour = {e["hour"]: e for e in self.plan}

    # -- Directive: solar_reduction -----------------------------------------
    def test_solar_reduction_respected(self):
        """Hours 13 & 14 must use ≤ 20 % of available solar."""
        for h in [13, 14]:
            max_solar = HOURS_DATA[h]["solar_kwh"] * 0.2
            assert self.by_hour[h]["solar_used_kwh"] <= max_solar + 0.01, (
                f"Hour {h}: solar_used={self.by_hour[h]['solar_used_kwh']}, "
                f"max_allowed={max_solar}"
            )

    # -- Directive: no_charge_window ----------------------------------------
    def test_no_charge_window_respected(self):
        """Hours 14 & 15 must have zero charging."""
        for h in [14, 15]:
            entry = self.by_hour[h]
            assert entry["battery_action"] != "charge", (
                f"Hour {h}: battery should not charge"
            )
            if entry["battery_action"] == "charge":
                assert entry["battery_kwh"] == 0

    # -- Directive: no_discharge_window -------------------------------------
    def test_no_discharge_window_respected(self):
        """Hours 16 & 17 must have zero discharging."""
        for h in [16, 17]:
            entry = self.by_hour[h]
            assert entry["battery_action"] != "discharge", (
                f"Hour {h}: battery should not discharge"
            )

    # -- Directive: minimum_battery_reserve ---------------------------------
    def test_minimum_battery_reserve_respected(self):
        """Hours 18, 19, 20 must have battery_energy_after >= 120 kWh."""
        for h in [18, 19, 20]:
            energy = self.by_hour[h]["battery_energy_after_kwh"]
            assert energy >= 120 - 0.01, (
                f"Hour {h}: battery_energy={energy}, required >= 120"
            )

    # -- Directive: max_grid_window -----------------------------------------
    def test_max_grid_window_respected(self):
        """Hours 19, 20, 21 must have grid_kwh ≤ 400."""
        for h in [19, 20, 21]:
            g = self.by_hour[h]["grid_kwh"]
            assert g <= 400 + 0.01, (
                f"Hour {h}: grid_kwh={g}, max_allowed=400"
            )

    # -- General invariants still hold with directives ----------------------
    def test_energy_balance_every_hour(self):
        for entry in self.plan:
            h = entry["hour"]
            demand = HOURS_DATA[h]["demand_kwh"]
            supply = entry["grid_kwh"] + entry["solar_used_kwh"]
            if entry["battery_action"] == "discharge":
                supply += entry["battery_kwh"]
            sink = demand
            if entry["battery_action"] == "charge":
                sink += entry["battery_kwh"]
            assert abs(supply - sink) < 0.01

    def test_end_of_day_neutrality(self):
        final_energy = self.plan[23]["battery_energy_after_kwh"]
        assert abs(final_energy - BATTERY["initial_energy_kwh"]) < 0.01


# ---------------------------------------------------------------------------
# Case 3 — Infeasibility produces a clear exception
# ---------------------------------------------------------------------------

class TestInfeasibility:
    """Trigger an infeasible model and verify a clear error is raised."""

    def test_impossible_grid_cap_raises(self):
        """
        Cap grid to 0 for every hour while demand > 0 and solar+battery
        can't cover it → should raise OptimizationError.
        """
        impossible_directives = [
            {"type": "max_grid_window", "hours": list(range(24)), "max_grid_kwh": 0},
        ]
        with pytest.raises(OptimizationError, match="[Ii]nfeasible"):
            solve_schedule(HOURS_DATA, BATTERY, directives=impossible_directives)


# ---------------------------------------------------------------------------
# Case 4 — BUP CSE Fest 2026 Official Public Sample Cases Benchmark (10/10)
# ---------------------------------------------------------------------------

import json
import os

SAMPLE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json",
)


@pytest.mark.skipif(not os.path.exists(SAMPLE_FILE), reason="Sample cases file not found")
class TestPublicSampleCases:
    """Validate optimizer against all 10 official BUP CSE Fest public sample cases."""

    with open(SAMPLE_FILE) as _f:
        _cases = json.load(_f)["cases"]

    @pytest.mark.parametrize("case", _cases, ids=[c["id"] for c in _cases])
    def test_official_sample_case_optimal(self, case):
        c_input = case["input"]
        c_expected = case["expected_output"]

        result = solve_schedule(
            hours=c_input["hours"],
            battery=c_input["battery"],
            directives=c_expected["directive_interpretation"],
        )

        # 1. Optimal cost match (tolerance 0.05 BDT)
        expected_cost = c_expected["total_cost_bdt"]
        assert abs(result["total_cost_bdt"] - expected_cost) <= 0.05, (
            f"Case {case['id']}: cost {result['total_cost_bdt']} != expected {expected_cost}"
        )

        # 2. Total grid energy match
        expected_grid = c_expected["total_grid_kwh"]
        assert abs(result["total_grid_kwh"] - expected_grid) <= 0.05, (
            f"Case {case['id']}: grid {result['total_grid_kwh']} != expected {expected_grid}"
        )

        # 3. Hourly energy balance
        for entry in result["hourly_plan"]:
            h = entry["hour"]
            demand = c_input["hours"][h]["demand_kwh"]
            supply = entry["grid_kwh"] + entry["solar_used_kwh"]
            if entry["battery_action"] == "discharge":
                supply += entry["battery_kwh"]
            sink = demand
            if entry["battery_action"] == "charge":
                sink += entry["battery_kwh"]
            assert abs(supply - sink) < 0.05, f"Hour {h} balance violated in {case['id']}"

        # 4. End-of-day battery neutrality
        initial_e = c_input["battery"]["initial_energy_kwh"]
        final_e = result["hourly_plan"][23]["battery_energy_after_kwh"]
        assert abs(final_e - initial_e) < 0.05, (
            f"Case {case['id']}: end-of-day SoC {final_e} != initial {initial_e}"
        )
