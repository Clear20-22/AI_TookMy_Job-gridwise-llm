"""Pydantic v2 request schemas for the /optimize-energy endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class HourEntry(BaseModel):
    """Hourly campus data: demand, solar forecast, and tariff."""

    hour: int = Field(..., ge=0, le=23)
    demand_kwh: float = Field(..., ge=0)
    solar_kwh: float = Field(..., ge=0)
    tariff_bdt_per_kwh: float = Field(..., ge=0)


class BatterySpec(BaseModel):
    """Battery storage parameters."""

    capacity_kwh: float = Field(..., gt=0)
    initial_energy_kwh: float = Field(..., ge=0)
    minimum_energy_kwh: float = Field(..., ge=0)
    max_charge_kwh_per_hour: float = Field(..., ge=0)
    max_discharge_kwh_per_hour: float = Field(..., ge=0)


class ScenarioRequest(BaseModel):
    """Top-level request body for POST /optimize-energy."""

    scenario_id: str
    operator_notes: list[str] = Field(..., min_length=1, max_length=3)
    hours: list[HourEntry] = Field(..., min_length=24, max_length=24)
    battery: BatterySpec

    @field_validator("operator_notes")
    @classmethod
    def validate_notes_not_blank(cls, v: list[str]) -> list[str]:
        """Ensure no note is an empty or whitespace-only string."""
        for i, note in enumerate(v):
            if not note.strip():
                raise ValueError(
                    f"operator_notes[{i}] is blank or whitespace-only. "
                    "All notes must contain meaningful text."
                )
        return v

    @field_validator("hours")
    @classmethod
    def validate_hours_coverage(cls, v: list[HourEntry]) -> list[HourEntry]:
        """Ensure exactly hours 0-23 are present, each exactly once."""
        hour_set = {entry.hour for entry in v}
        expected = set(range(24))
        if hour_set != expected:
            missing = expected - hour_set
            extra = hour_set - expected
            raise ValueError(
                f"hours must contain exactly hours 0-23. "
                f"Missing: {sorted(missing)}, Extra: {sorted(extra)}"
            )
        # Sort by hour for consistent processing
        return sorted(v, key=lambda e: e.hour)

