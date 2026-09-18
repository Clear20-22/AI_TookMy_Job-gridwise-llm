from __future__ import annotations

from typing import Any

import pulp

# ---------------------------------------------------------------------------
# Custom exception for solver failures
# ---------------------------------------------------------------------------

class OptimizationError(Exception):
    """Raised when the LP solver cannot find a feasible / bounded solution."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPS = 1e-3  # threshold below which a value is treated as zero


def _clean(value: float) -> float:
    """Round tiny floating-point noise to zero and round to 4 decimal places."""
    if abs(value) < _EPS:
        return 0.0
    return round(value, 4)


def _battery_action(charge: float, discharge: float) -> str:
    """Derive a human-readable battery action label after cleaning."""
    if charge > 0:
        return "charge"
    if discharge > 0:
        return "discharge"
    return "idle"


# ---------------------------------------------------------------------------
# Pre-processing: build per-hour effective parameters from directives
# ---------------------------------------------------------------------------

def _build_effective_params(
    hours: list[dict[str, Any]],
    battery: dict[str, Any],
    directives: list[dict[str, Any]],
) -> tuple[
    dict[int, float],   # effective_solar
    dict[int, float],   # effective_min_battery
    set[int],           # no_charge_hours
    set[int],           # no_discharge_hours
    dict[int, float],   # max_grid_per_hour
]:
    """
    Walk the directive list once and produce per-hour overrides.
    """
    effective_solar: dict[int, float] = {
        entry["hour"]: entry["solar_kwh"] for entry in hours
    }
    effective_min_battery: dict[int, float] = {
        h: battery["minimum_energy_kwh"] for h in range(24)
    }
    no_charge_hours: set[int] = set()
    no_discharge_hours: set[int] = set()
    max_grid: dict[int, float] = {}  # only populated for constrained hours

    for d in directives:
        dtype = d["type"]

        if dtype == "solar_reduction":
            # Rule 2 — reduce available solar by a multiplicative factor
            for h in d["hours"]:
                effective_solar[h] = hours[h]["solar_kwh"] * d["factor"]

        elif dtype == "no_charge_window":
            # Rule 6a — battery charging forbidden
            for h in d["hours"]:
                no_charge_hours.add(h)

        elif dtype == "no_discharge_window":
            # Rule 6b — battery discharging forbidden
            for h in d["hours"]:
                no_discharge_hours.add(h)

        elif dtype == "minimum_battery_reserve":
            # Rule 4 — tighter lower bound on battery SoE
            for h in d["hours"]:
                effective_min_battery[h] = max(
                    effective_min_battery[h], d["minimum_energy_kwh"]
                )

        elif dtype == "max_grid_window":
            # Rule 6c — cap on grid purchase
            for h in d["hours"]:
                # If multiple directives apply, keep the tightest cap
                if h in max_grid:
                    max_grid[h] = min(max_grid[h], d["max_grid_kwh"])
                else:
                    max_grid[h] = d["max_grid_kwh"]

    return effective_solar, effective_min_battery, no_charge_hours, no_discharge_hours, max_grid


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_schedule(
    hours: list[dict[str, Any]],
    battery: dict[str, Any],
    directives: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Solve the 24-hour energy-cost minimisation LP.
    """

    # -- Pre-process directives into per-hour parameters --------------------
    (
        eff_solar,
        eff_min_bat,
        no_charge_hrs,
        no_discharge_hrs,
        max_grid_hrs,
    ) = _build_effective_params(hours, battery, directives)

    H = range(24)
    prob = pulp.LpProblem("GridWise_Schedule", pulp.LpMinimize)

    # -- Decision variables -------------------------------------------------
    grid      = [pulp.LpVariable(f"grid_{h}",      lowBound=0) for h in H]
    solar     = [pulp.LpVariable(f"solar_{h}",     lowBound=0) for h in H]
    charge    = [pulp.LpVariable(f"charge_{h}",    lowBound=0) for h in H]
    discharge = [pulp.LpVariable(f"discharge_{h}", lowBound=0) for h in H]
    bat_energy = [pulp.LpVariable(f"bat_e_{h}",    lowBound=0) for h in H]
    # Binary indicator for constraint 8 (see rate-limit block below): 1 means
    # the battery may charge this hour, 0 means it may discharge. Needed
    # because charge/discharge only appear as a net term (charge - discharge)
    # in the balance and transition constraints, so without this gate the LP
    # can add equal amounts to both with zero effect on cost or feasibility —
    # a degenerate optimum, not something the solver naturally avoids.
    is_charging = [pulp.LpVariable(f"is_charging_{h}", cat="Binary") for h in H]

    # -- Objective: minimise total grid electricity cost --------------------
    prob += pulp.lpSum(
        grid[h] * hours[h]["tariff_bdt_per_kwh"] for h in H
    ), "total_grid_cost"

    for h in H:
        demand_h = hours[h]["demand_kwh"]
        tariff_label = f"h{h}"

        # ---- Constraint 1: energy balance ---------------------------------
        prob += (
            grid[h] + solar[h] + discharge[h] == demand_h + charge[h],
            f"energy_balance_{tariff_label}",
        )

        # ---- Constraint 2: solar usage bound ------------------------------
        prob += (
            solar[h] <= eff_solar[h],
            f"solar_cap_{tariff_label}",
        )

        # ---- Constraint 3: battery state-of-energy transition -------------
        prev_energy = (
            battery["initial_energy_kwh"] if h == 0 else bat_energy[h - 1]
        )
        prob += (
            bat_energy[h] == prev_energy + charge[h] - discharge[h],
            f"bat_transition_{tariff_label}",
        )

        # ---- Constraint 4: battery energy bounds --------------------------
        prob += (
            bat_energy[h] <= battery["capacity_kwh"],
            f"bat_cap_upper_{tariff_label}",
        )
        prob += (
            bat_energy[h] >= eff_min_bat[h],
            f"bat_cap_lower_{tariff_label}",
        )

        # ---- Constraint 5 + 8: rate limits AND mutual exclusion ------------
        # Gating each bound by is_charging[h] (or its complement) enforces the
        # normal rate limit *and* rules out simultaneous charge+discharge in
        # one pair of constraints — see the is_charging comment above.
        prob += (
            charge[h] <= battery["max_charge_kwh_per_hour"] * is_charging[h],
            f"charge_rate_{tariff_label}",
        )
        prob += (
            discharge[h] <= battery["max_discharge_kwh_per_hour"] * (1 - is_charging[h]),
            f"discharge_rate_{tariff_label}",
        )

        # ---- Constraint 6a: no_charge_window directive --------------------
        if h in no_charge_hrs:
            prob += (charge[h] == 0, f"no_charge_{tariff_label}")

        # ---- Constraint 6b: no_discharge_window directive -----------------
        if h in no_discharge_hrs:
            prob += (discharge[h] == 0, f"no_discharge_{tariff_label}")

        # ---- Constraint 6c: max_grid_window directive ---------------------
        if h in max_grid_hrs:
            prob += (
                grid[h] <= max_grid_hrs[h],
                f"max_grid_{tariff_label}",
            )

    # ---- Constraint 7: end-of-day battery neutrality ----------------------
    prob += (
        bat_energy[23] == battery["initial_energy_kwh"],
        "end_of_day_neutrality",
    )

    # -- Solve --------------------------------------------------------------
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise OptimizationError(
            f"Solver returned status '{status}'. "
            "The problem may be infeasible given the current inputs and directives."
        )

    # -- Extract & clean results --------------------------------------------
    hourly_plan: list[dict[str, Any]] = []
    total_grid = 0.0
    total_cost = 0.0
    peak_grid = 0.0

    for h in H:
        g  = _clean(grid[h].varValue)
        s  = _clean(solar[h].varValue)
        ch = _clean(charge[h].varValue)
        dc = _clean(discharge[h].varValue)
        be = _clean(bat_energy[h].varValue)

        # Post-solve assertion for constraint 8
        assert not (ch > 0 and dc > 0), (
            f"Hour {h}: solver simultaneously charged ({ch}) and "
            f"discharged ({dc}) — this should never happen."
        )

        action = _battery_action(ch, dc)
        bat_kwh = ch if action == "charge" else dc  # magnitude of flow

        hourly_plan.append({
            "hour": h,
            "grid_kwh": g,
            "solar_used_kwh": s,
            "battery_action": action,
            "battery_kwh": bat_kwh,
            "battery_energy_after_kwh": be,
        })

        total_grid += g
        total_cost += g * hours[h]["tariff_bdt_per_kwh"]
        peak_grid = max(peak_grid, g)

    return {
        "hourly_plan": hourly_plan,
        "total_grid_kwh": round(total_grid, 4),
        "total_cost_bdt": round(total_cost, 4),
        "peak_grid_kwh": round(peak_grid, 4),
    }