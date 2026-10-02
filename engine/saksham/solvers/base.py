"""
Abstract Solver Backend Interface.
Defines contracts for numerical solvers (HiGHS, SciPy, future C++/Rust kernel).
Full integration scheduled for Phase 6.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any
import numpy as np
from ..canonical.model import CanonicalModel
from ..strategy.recommender import StrategyRecommendation


@dataclass
class SolveResult:
    """
    Real solver results data structure.
    Only real metrics available from the underlying solver will be populated.
    """
    status: str = "NOT_SOLVED"
    objective_value: Optional[float] = None
    primal_solution: Optional[np.ndarray] = None
    dual_solution: Optional[np.ndarray] = None
    wall_time_seconds: float = 0.0
    iteration_count: int = 0
    solver_name: str = "generic_backend"
    details: Dict[str, Any] = field(default_factory=dict)


class AbstractSolverBackend(ABC):
    """Base interface for all mathematical solver backends."""

    @abstractmethod
    def load_model(self, model: CanonicalModel) -> None:
        """Load CanonicalModel into solver."""
        pass

    @abstractmethod
    def apply_strategy(self, strategy: StrategyRecommendation) -> None:
        """Apply selected strategy parameters."""
        pass

    @abstractmethod
    def solve(self, telemetry_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> SolveResult:
        """Execute numerical solve."""
        pass
