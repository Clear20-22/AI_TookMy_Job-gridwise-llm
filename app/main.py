"""HTTP entrypoint for the GridWise LLM service."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from app.model.request import ScenarioRequest
from app.model.response import (
    DirectiveInterpretation,
    HourlyPlanEntry,
    OptimizeResponse,
)
from app.optimizer import OptimizationError, solve_schedule
from app.services.guardrails import directives_to_optimizer_format, validate_directives
from app.services.llm_parser import get_generative_model, interpret_notes

# Load .env file if present
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the Gemini model singleton at startup to eliminate cold-start latency."""
    try:
        model = get_generative_model()
        if model is not None:
            logger.info("Gemini model singleton pre-warmed successfully")
    except Exception as exc:
        logger.warning("Could not pre-warm Gemini model at startup: %s", exc)
    yield


app = FastAPI(
    title="GridWise LLM",
    description="Smart campus energy optimization microservice.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", tags=["system"])
def read_root() -> dict[str, str]:
    """Return a small service descriptor."""
    return {"service": "gridwise-llm", "status": "ok"}


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    """Liveness check for local and container orchestration."""
    return {"status": "ok"}


@app.post("/optimize-energy", tags=["core"], response_model=OptimizeResponse)
async def optimize_energy(request: ScenarioRequest) -> OptimizeResponse:
    """
    Accept a scenario and return the optimal 24-hour energy schedule.

    Pipeline:
        1. LLM interprets operator notes → raw directives  (async via thread)
        2. Guardrails validate and normalise directives     (sync, fast)
        3. LP solver optimises the schedule                (async via thread)
        4. Response is assembled and returned
    """
    logger.info("Processing scenario %s with %d notes", request.scenario_id, len(request.operator_notes))

    # ---- Stage 1: LLM Interpretation (runs in thread pool) ---------------
    try:
        raw_directives = await asyncio.to_thread(
            interpret_notes,
            request.operator_notes,
            request.battery.capacity_kwh,
        )
    except Exception as exc:
        logger.exception("LLM interpretation failed for scenario %s", request.scenario_id)
        # Graceful fallback: treat all notes as no_op
        raw_directives = [
            {
                "note_index": i,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "LLM processing unavailable; safely demoted to no-op.",
            }
            for i in range(len(request.operator_notes))
        ]

    # ---- Stage 2: Guardrail Validation ------------------------------------
    validated_directives = validate_directives(
        raw_directives=raw_directives,
        battery_capacity_kwh=request.battery.capacity_kwh,
        battery_minimum_kwh=request.battery.minimum_energy_kwh,
    )

    # ---- Stage 3: Convert to optimizer format & solve ---------------------
    optimizer_directives = directives_to_optimizer_format(validated_directives)

    # Build hours list as plain dicts for the solver
    hours_data: list[dict[str, Any]] = [
        {
            "hour": entry.hour,
            "demand_kwh": entry.demand_kwh,
            "solar_kwh": entry.solar_kwh,
            "tariff_bdt_per_kwh": entry.tariff_bdt_per_kwh,
        }
        for entry in request.hours
    ]
    battery_data: dict[str, Any] = {
        "capacity_kwh": request.battery.capacity_kwh,
        "initial_energy_kwh": request.battery.initial_energy_kwh,
        "minimum_energy_kwh": request.battery.minimum_energy_kwh,
        "max_charge_kwh_per_hour": request.battery.max_charge_kwh_per_hour,
        "max_discharge_kwh_per_hour": request.battery.max_discharge_kwh_per_hour,
    }

    try:
        solver_result = await asyncio.to_thread(
            solve_schedule, hours_data, battery_data, optimizer_directives
        )
    except OptimizationError as exc:
        logger.error("Solver failed for scenario %s: %s", request.scenario_id, exc)
        raise HTTPException(
            status_code=500,
            detail="Optimization failed: problem is infeasible under provided constraints.",
        )

    # ---- Stage 4: Assemble response --------------------------------------
    directive_interpretations = [
        DirectiveInterpretation(**d) for d in validated_directives
    ]

    hourly_plan = [
        HourlyPlanEntry(**entry) for entry in solver_result["hourly_plan"]
    ]

    # Build a brief plan summary
    active_types = [
        d.directive_type
        for d in directive_interpretations
        if d.applies
    ]
    if active_types:
        summary_directives = ", ".join(sorted(set(active_types)))
        plan_summary = (
            f"Optimal 24-hour schedule honouring {summary_directives} "
            f"directive(s) while maintaining end-of-day battery neutrality. "
            f"Total cost: {solver_result['total_cost_bdt']:.2f} BDT."
        )
    else:
        plan_summary = (
            f"Optimal 24-hour schedule with no active operational directives. "
            f"Total cost: {solver_result['total_cost_bdt']:.2f} BDT."
        )

    return OptimizeResponse(
        scenario_id=request.scenario_id,
        directive_interpretation=directive_interpretations,
        hourly_plan=hourly_plan,
        total_grid_kwh=solver_result["total_grid_kwh"],
        total_cost_bdt=solver_result["total_cost_bdt"],
        peak_grid_kwh=solver_result["peak_grid_kwh"],
        plan_summary=plan_summary,
    )
