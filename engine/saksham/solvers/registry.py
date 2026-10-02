"""Small explicit solver registry and strategy-to-backend selection."""
from typing import Any, Callable, Dict, List, Optional

from ..canonical.model import CanonicalModel
from ..strategy.recommender import StrategyRecommendation, StrategyRecommender
from .base import AbstractSolverBackend, SolveResult
from .highs_backend import HighsSolverBackend, SolverBackendError


class SolverUnavailableError(RuntimeError):
    """Raised when a requested solver is unknown or unavailable."""


class UnsupportedStrategyError(ValueError):
    """Raised when no registered solver supports a recommendation category."""


class SolverRegistry:
    """Register backends and route recommendations without silent fallback."""

    _SUPPORTED_STRATEGIES = {
        "DIRECT_LP",
        "GENERAL_SPARSE",
        "GENERAL_MIP",
        "NETWORK_OR_FLOW_CANDIDATE",
    }

    def __init__(self, register_defaults: bool = True) -> None:
        self._factories: Dict[str, Callable[[], AbstractSolverBackend]] = {}
        if register_defaults:
            self.register("highs", HighsSolverBackend)

    def register(
        self,
        name: str,
        factory: Callable[[], AbstractSolverBackend],
    ) -> None:
        """Register a backend factory under a deterministic lowercase name."""
        normalized_name = name.strip().lower()
        if not normalized_name:
            raise ValueError("Solver name must not be empty.")
        if not callable(factory):
            raise TypeError("Solver factory must be callable.")
        self._factories[normalized_name] = factory

    def get(self, name: str) -> AbstractSolverBackend:
        """Create a named backend or raise an explicit availability error."""
        normalized_name = name.strip().lower()
        factory = self._factories.get(normalized_name)
        if factory is None:
            raise SolverUnavailableError(f"Solver '{normalized_name}' is not registered.")
        backend = factory()
        if not isinstance(backend, AbstractSolverBackend):
            raise TypeError(f"Factory for '{normalized_name}' did not return a solver backend.")
        if not backend.is_available():
            raise SolverUnavailableError(
                f"Solver '{normalized_name}' is unavailable on this system."
            )
        return backend

    def available(self) -> Dict[str, bool]:
        """Return registered solver availability without fallback behavior."""
        result: Dict[str, bool] = {}
        for name, factory in self._factories.items():
            try:
                backend = factory()
                result[name] = isinstance(backend, AbstractSolverBackend) and backend.is_available()
            except Exception:
                result[name] = False
        return result

    def supported_solvers(self) -> List[Dict[str, object]]:
        """Describe registered solver capabilities using backend-neutral data."""
        descriptions = []
        for name, factory in self._factories.items():
            try:
                backend = factory()
                is_available = backend.is_available()
                supported_types = list(backend.supported_variable_types)
                supports_mip = backend.supports_mip
            except Exception:
                is_available = False
                supported_types = []
                supports_mip = False
            descriptions.append(
                {
                    "name": name,
                    "available": is_available,
                    "supported_variable_types": supported_types,
                    "supports_mip": supports_mip,
                }
            )
        return descriptions

    def select_for_strategy(
        self,
        strategy: StrategyRecommendation,
    ) -> AbstractSolverBackend:
        """Select a compatible registered backend for a valid recommendation."""
        if not isinstance(strategy, StrategyRecommendation):
            raise TypeError("strategy must be a StrategyRecommendation instance.")
        if strategy.status == "INVALID" or strategy.primary_strategy == "INVALID_MODEL":
            raise UnsupportedStrategyError("Invalid models cannot be sent to a solver.")
        if strategy.primary_strategy not in self._SUPPORTED_STRATEGIES:
            raise UnsupportedStrategyError(
                f"Strategy '{strategy.primary_strategy}' is not supported by this registry."
            )
        backend = self.get("highs")
        if strategy.primary_strategy == "GENERAL_MIP" and not backend.supports_mip:
            raise UnsupportedStrategyError("No registered solver supports mixed-integer models.")
        return backend

    def solve(
        self,
        model: CanonicalModel,
        strategy: Optional[StrategyRecommendation] = None,
        telemetry_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> SolveResult:
        """Recommend if needed, select one backend, and return its normalized result."""
        if not isinstance(model, CanonicalModel):
            raise TypeError("model must be a CanonicalModel instance.")
        if strategy is None:
            strategy = StrategyRecommender().recommend(model)
        if strategy.status == "INVALID" or strategy.primary_strategy == "INVALID_MODEL":
            reason = "; ".join(strategy.warnings) or "Model was marked invalid."
            return SolveResult(
                status="ERROR",
                solver_name="unselected",
                message="Solve was not started because pre-solve analysis marked the model invalid.",
                termination_reason="INVALID_MODEL",
                error_message=reason,
            )

        try:
            backend = self.select_for_strategy(strategy)
            backend.load_model(model)
            backend.apply_strategy(strategy)
            return backend.solve(telemetry_callback=telemetry_callback)
        except UnsupportedStrategyError as error:
            return SolveResult(
                status="UNSUPPORTED",
                solver_name="unselected",
                message=str(error),
                termination_reason="UNSUPPORTED_STRATEGY",
                error_message=str(error),
            )
        except SolverUnavailableError as error:
            return SolveResult(
                status="ERROR",
                solver_name="unselected",
                message=str(error),
                termination_reason="SOLVER_UNAVAILABLE",
                error_message=str(error),
            )
        except (ModelValidationError, SolverBackendError, ValueError) as error:
            return SolveResult(
                status="ERROR",
                solver_name="highs",
                message=str(error),
                termination_reason="MODEL_OR_BACKEND_ERROR",
                error_message=str(error),
            )
        except Exception as error:
            return SolveResult(
                status="ERROR",
                solver_name="highs",
                message=f"Solver runtime error: {error}",
                termination_reason="BACKEND_ERROR",
                error_message=str(error),
            )