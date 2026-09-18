"""Forwarding module for app.services.optimizer to app.optimizer."""

from app.optimizer.optimizer import (
    OptimizationError,
    solve_schedule,
)

__all__ = ["OptimizationError", "solve_schedule"]
