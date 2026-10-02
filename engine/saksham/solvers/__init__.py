"""Solver abstraction, HiGHS backend, and explicit solver registry.

Phase 6 provides one LP/MILP backend. It does not add advanced solver tuning,
automatic benchmarking, portfolio or distributed solving, or decomposition.
"""
from .base import AbstractSolverBackend, SolveResult
from .highs_backend import HighsSolverBackend, SolverBackendError, SolverInputError
from .registry import (
	SolverRegistry,
	SolverUnavailableError,
	UnsupportedStrategyError,
)

__all__ = [
	"AbstractSolverBackend",
	"HighsSolverBackend",
	"SolveResult",
	"SolverBackendError",
	"SolverInputError",
	"SolverRegistry",
	"SolverUnavailableError",
	"UnsupportedStrategyError",
]
