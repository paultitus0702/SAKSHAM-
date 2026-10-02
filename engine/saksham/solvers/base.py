"""Solver-independent contracts shared by concrete optimization backends.

The base interface accepts a validated CanonicalModel and returns normalized
result metadata. Concrete backends own model conversion and never expose
solver-specific objects through SolveResult.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional
import numpy as np
from ..canonical.model import CanonicalModel
from ..strategy.recommender import StrategyRecommendation


@dataclass
class SolveResult:
    """Solver-independent solution and measured solver metadata.

    Statistics remain ``None`` when the backend does not provide them. The
    existing ``primal_solution`` and ``details`` fields remain the storage
    contract for variable values and backend-neutral statistics.
    """
    status: str = "NOT_SOLVED"
    objective_value: Optional[float] = None
    primal_solution: Optional[np.ndarray] = None
    dual_solution: Optional[np.ndarray] = None
    wall_time_seconds: Optional[float] = None
    iteration_count: Optional[int] = None
    solver_name: str = "generic_backend"
    details: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    message: str = ""
    termination_reason: Optional[str] = None
    error_message: Optional[str] = None

    @property
    def variable_values(self) -> Optional[np.ndarray]:
        """Alias for the primal variable solution."""
        return self.primal_solution

    @property
    def solver_statistics(self) -> Dict[str, Any]:
        """Alias for reported backend-neutral statistics."""
        return self.details


class AbstractSolverBackend(ABC):
    """Base lifecycle for solver-independent optimization backends."""

    @property
    def solver_name(self) -> str:
        """Stable backend identifier used in results and the registry."""
        return self.__class__.__name__

    def is_available(self) -> bool:
        """Return whether the concrete backend can currently initialize."""
        return True

    @property
    def supported_variable_types(self) -> tuple[str, ...]:
        """Variable types supported by the concrete backend."""
        return ("continuous",)

    @property
    def supports_mip(self) -> bool:
        """Whether the backend can solve integer and binary models."""
        return False

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
        """Execute numerical solve and optionally emit final solve telemetry."""
        pass
