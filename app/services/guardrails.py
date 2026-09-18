"""
Deterministic guardrail validator for LLM-parsed directives.

Every directive produced by the LLM is validated and normalised here
*before* it touches the LP solver.  Malformed directives degrade to
``no_op`` with a logged warning — the service never crashes on bad LLM output.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---- Allowed enums --------------------------------------------------------

ALLOWED_DIRECTIVE_TYPES = frozenset({
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
})

ALLOWED_BATTERY_ACTIONS = frozenset({"charge", "discharge", "idle"})


# ---- Internal helpers -----------------------------------------------------

def _make_no_op(note_index: int, reason: str) -> dict[str, Any]:
    """Return a safe no_op directive with an explanation of why it was demoted."""
    logger.warning("Directive for note %d demoted to no_op: %s", note_index, reason)
    return {
        "note_index": note_index,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": f"Guardrail override: {reason}",
    }


def _validate_hours(hours: Any) -> list[int] | None:
    """Return a sorted, deduplicated list of ints in [0, 23], or None on failure."""
    if not isinstance(hours, list):
        return None
    clean: list[int] = []
    for h in hours:
        if not isinstance(h, (int, float)):
            return None
        h_int = int(h)
        if h_int < 0 or h_int > 23:
            return None
        clean.append(h_int)
    # De-duplicate and sort
    return sorted(set(clean))


# ---- Public API -----------------------------------------------------------

def validate_directives(
    raw_directives: list[dict[str, Any]],
    battery_capacity_kwh: float,
    battery_minimum_kwh: float,
) -> list[dict[str, Any]]:
    """
    Validate and normalise a list of raw LLM-emitted directives.

    Parameters
    ----------
    raw_directives : list[dict]
        One dict per operator note, as returned by the LLM parser.
    battery_capacity_kwh : float
        Battery capacity (for clamping minimum_energy_kwh).
    battery_minimum_kwh : float
        Battery physical minimum (for clamping minimum_energy_kwh).

    Returns
    -------
    list[dict]
        Validated directives, safe to feed into the LP solver.
    """
    validated: list[dict[str, Any]] = []

    for i, raw in enumerate(raw_directives):
        note_index = raw.get("note_index", i)

        # ---- Check directive_type is known --------------------------------
        dtype = raw.get("directive_type")
        if dtype not in ALLOWED_DIRECTIVE_TYPES:
            validated.append(
                _make_no_op(note_index, f"Unknown directive_type '{dtype}'")
            )
            continue

        # ---- Handle no_op -------------------------------------------------
        if dtype == "no_op":
            validated.append({
                "note_index": note_index,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": raw.get("explanation", "No operational impact."),
            })
            continue

        # ---- All other types require structured_adjustment ----------------
        adj = raw.get("structured_adjustment")
        if not isinstance(adj, dict):
            validated.append(
                _make_no_op(note_index, "Missing or invalid structured_adjustment")
            )
            continue

        # ---- All non-no_op types require valid hours ----------------------
        hours = _validate_hours(adj.get("hours"))
        if hours is None or len(hours) == 0:
            validated.append(
                _make_no_op(note_index, "Invalid or empty hours array")
            )
            continue

        # ---- Type-specific validation -------------------------------------
        if dtype == "solar_reduction":
            factor = adj.get("factor")
            if not isinstance(factor, (int, float)):
                validated.append(
                    _make_no_op(note_index, "solar_reduction missing numeric factor")
                )
                continue
            # Clamp factor to [0.0, 1.0]
            factor = max(0.0, min(1.0, float(factor)))
            validated.append({
                "note_index": note_index,
                "applies": True,
                "directive_type": "solar_reduction",
                "structured_adjustment": {"hours": hours, "factor": factor},
                "explanation": raw.get("explanation", ""),
            })

        elif dtype == "minimum_battery_reserve":
            min_e = adj.get("minimum_energy_kwh")
            if not isinstance(min_e, (int, float)):
                validated.append(
                    _make_no_op(
                        note_index,
                        "minimum_battery_reserve missing numeric minimum_energy_kwh",
                    )
                )
                continue
            # Clamp to [battery_minimum, capacity]
            min_e = max(battery_minimum_kwh, min(battery_capacity_kwh, float(min_e)))
            validated.append({
                "note_index": note_index,
                "applies": True,
                "directive_type": "minimum_battery_reserve",
                "structured_adjustment": {
                    "hours": hours,
                    "minimum_energy_kwh": min_e,
                },
                "explanation": raw.get("explanation", ""),
            })

        elif dtype == "max_grid_window":
            max_g = adj.get("max_grid_kwh")
            if not isinstance(max_g, (int, float)):
                validated.append(
                    _make_no_op(
                        note_index,
                        "max_grid_window missing numeric max_grid_kwh",
                    )
                )
                continue
            max_g = max(0.0, float(max_g))
            validated.append({
                "note_index": note_index,
                "applies": True,
                "directive_type": "max_grid_window",
                "structured_adjustment": {"hours": hours, "max_grid_kwh": max_g},
                "explanation": raw.get("explanation", ""),
            })

        elif dtype in ("no_charge_window", "no_discharge_window"):
            # Only hours needed — already validated above
            validated.append({
                "note_index": note_index,
                "applies": True,
                "directive_type": dtype,
                "structured_adjustment": {"hours": hours},
                "explanation": raw.get("explanation", ""),
            })

    return validated


def directives_to_optimizer_format(
    validated_directives: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert validated directive dicts (with structured_adjustment nesting)
    into the flat dict format the optimizer's ``solve_schedule`` expects.

    The optimizer expects::

        {"type": "solar_reduction", "hours": [...], "factor": 0.25}

    while the API response uses::

        {"directive_type": "solar_reduction",
         "structured_adjustment": {"hours": [...], "factor": 0.25}}
    """
    opt_list: list[dict[str, Any]] = []

    for d in validated_directives:
        if d["directive_type"] == "no_op":
            continue  # no_op has no effect on the optimizer
        adj = d["structured_adjustment"]
        flat: dict[str, Any] = {"type": d["directive_type"], **adj}
        opt_list.append(flat)

    return opt_list
