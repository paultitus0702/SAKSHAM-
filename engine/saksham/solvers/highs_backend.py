"""HiGHS implementation of the solver-independent backend contract.

Canonical CSR models are copied to HiGHS's sparse column-wise LP structure.
Objective sense and offset are carried natively, while integer and binary
variables are passed as integral columns with their canonical bounds.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
from scipy import sparse

try:
    import highspy
except ImportError as import_error:  # pragma: no cover - dependency is declared
    highspy = None
    _HIGHS_IMPORT_ERROR = str(import_error)
else:
    _HIGHS_IMPORT_ERROR = None

from ..canonical.model import CanonicalModel, ObjectiveSense, VarType
from ..canonical.validator import ModelValidationError, ModelValidator
from ..strategy.recommender import StrategyRecommendation
from .base import AbstractSolverBackend, SolveResult


class SolverInputError(ValueError):
    """Raised when a valid CIR contains values unsupported by the backend."""


class SolverBackendError(RuntimeError):
    """Raised when HiGHS rejects a converted model or option."""


class HighsSolverBackend(AbstractSolverBackend):
    """Sparse LP/MILP backend using the existing ``highspy`` dependency."""

    @property
    def solver_name(self) -> str:
        return "highs"

    @property
    def supported_variable_types(self) -> tuple[str, ...]:
        return ("continuous", "integer", "binary")

    @property
    def supports_mip(self) -> bool:
        return True

    def __init__(self, time_limit_seconds: Optional[float] = None) -> None:
        self._highs = None
        self._model: Optional[CanonicalModel] = None
        self._strategy: Optional[StrategyRecommendation] = None
        self._initialization_error: Optional[str] = None

        if highspy is None:
            self._initialization_error = _HIGHS_IMPORT_ERROR or "highspy is unavailable."
            return

        try:
            self._highs = highspy.Highs()
            self._set_option("output_flag", False)
            if time_limit_seconds is not None:
                time_limit = float(time_limit_seconds)
                if not np.isfinite(time_limit) or time_limit <= 0:
                    raise ValueError("time_limit_seconds must be finite and positive.")
                self._set_option("time_limit", time_limit)
        except Exception as error:
            self._initialization_error = str(error)
            self._highs = None

    def is_available(self) -> bool:
        """Report whether the HiGHS wrapper initialized successfully."""
        return self._highs is not None

    @staticmethod
    def convert_model(model: CanonicalModel):
        """Convert a validated CIR into a sparse ``highspy.HighsLp`` object."""
        if not isinstance(model, CanonicalModel):
            raise TypeError("model must be a CanonicalModel instance.")
        report = ModelValidator.validate(model)
        if not report.is_valid:
            raise ModelValidationError(
                f"Model '{model.name}' failed validation before solver conversion.",
                errors=report.errors,
            )
        if highspy is None:
            raise SolverBackendError(_HIGHS_IMPORT_ERROR or "highspy is unavailable.")
        if not np.all(np.isfinite(model.c)):
            raise SolverInputError("Objective coefficients must be finite for HiGHS.")
        if not np.isfinite(model.c0):
            raise SolverInputError("Objective constant must be finite for HiGHS.")
        if not np.all(np.isfinite(model.rhs)):
            raise SolverInputError("Constraint RHS values must be finite for HiGHS.")
        if np.any(np.isposinf(model.l_var)) or np.any(np.isneginf(model.u_var)):
            raise SolverInputError(
                "Variable lower bounds may be -infinity but not +infinity; "
                "upper bounds may be +infinity but not -infinity."
            )
        if model.sense not in (ObjectiveSense.MINIMIZE, ObjectiveSense.MAXIMIZE):
            raise SolverInputError(f"Unsupported objective sense: {model.sense!r}.")

        row_lower, row_upper = model.compute_row_bounds()
        matrix = model.A.tocsc(copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()

        highs_model = highspy.HighsLp()
        highs_model.num_col_ = model.num_vars
        highs_model.num_row_ = model.num_constrs
        highs_model.col_cost_ = model.c.copy()
        highs_model.col_lower_ = model.l_var.copy()
        highs_model.col_upper_ = model.u_var.copy()
        highs_model.row_lower_ = row_lower
        highs_model.row_upper_ = row_upper
        highs_model.offset_ = float(model.c0)
        highs_model.sense_ = (
            highspy.ObjSense.kMinimize
            if model.sense == ObjectiveSense.MINIMIZE
            else highspy.ObjSense.kMaximize
        )
        highs_model.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        highs_model.a_matrix_.start_ = matrix.indptr.astype(np.int32, copy=True)
        highs_model.a_matrix_.index_ = matrix.indices.astype(np.int32, copy=True)
        highs_model.a_matrix_.value_ = matrix.data.astype(np.float64, copy=True)
        if model.is_mip:
            highs_model.integrality_ = [
                highspy.HighsVarType.kContinuous
                if var_type == VarType.CONTINUOUS
                else highspy.HighsVarType.kInteger
                for var_type in model.var_types
            ]
        return highs_model

    def load_model(self, model: CanonicalModel) -> None:
        """Validate and load a copied sparse representation into HiGHS."""
        if not isinstance(model, CanonicalModel):
            raise TypeError("model must be a CanonicalModel instance.")
        if not self.is_available():
            raise SolverBackendError(
                f"HiGHS is unavailable: {self._initialization_error or 'unknown error'}"
            )
        highs_model = self.convert_model(model)
        status = self._highs.passModel(highs_model)
        if status == highspy.HighsStatus.kError:
            raise SolverBackendError("HiGHS rejected the converted CanonicalModel.")
        self._model = model

    def apply_strategy(self, strategy: StrategyRecommendation) -> None:
        """Apply supported generic hints; unsupported hints keep HiGHS defaults."""
        if not isinstance(strategy, StrategyRecommendation):
            raise TypeError("strategy must be a StrategyRecommendation instance.")
        if strategy.status == "INVALID" or strategy.primary_strategy == "INVALID_MODEL":
            raise ValueError("An invalid-model recommendation cannot be solved.")
        if not self.is_available():
            raise SolverBackendError(
                f"HiGHS is unavailable: {self._initialization_error or 'unknown error'}"
            )

        if strategy.presolve.lower() in {"on", "off", "choose"}:
            self._set_option("presolve", strategy.presolve.lower())
        if strategy.primary_algorithm == "dual_simplex":
            self._set_option("solver", "simplex")
        self._strategy = strategy

    def solve(
        self,
        telemetry_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> SolveResult:
        """Solve the loaded LP/MILP and normalize HiGHS status and statistics."""
        if self._model is None or self._highs is None:
            return SolveResult(
                status="ERROR",
                solver_name=self.solver_name,
                message="No model is loaded or the HiGHS backend is unavailable.",
                termination_reason="MODEL_NOT_LOADED",
                error_message=self._initialization_error,
            )

        try:
            run_status = self._highs.run()
            model_status = self._highs.getModelStatus()
            model_status_text = self._highs.modelStatusToString(model_status)
            solution = self._highs.getSolution()
            info = self._highs.getInfo()
            status, termination_reason = self._normalize_status(
                model_status, solution.value_valid
            )
            primal = None
            objective_value = None
            dual = None
            if status in {"OPTIMAL", "FEASIBLE"} and solution.value_valid:
                candidate = np.asarray(solution.col_value, dtype=np.float64)
                if candidate.shape == (self._model.num_vars,) and np.all(np.isfinite(candidate)):
                    primal = candidate.copy()
                    value = float(self._highs.getObjectiveValue())
                    if np.isfinite(value):
                        objective_value = value
                if solution.dual_valid:
                    row_duals = np.asarray(solution.row_dual, dtype=np.float64)
                    if row_duals.shape == (self._model.num_constrs,):
                        dual = row_duals.copy()

            statistics: Dict[str, Any] = {
                "highs_model_status": model_status_text,
                "highs_run_status": str(run_status),
            }
            iteration_count = None
            if info.valid:
                simplex_iterations = int(info.simplex_iteration_count)
                ipm_iterations = int(info.ipm_iteration_count)
                pdlp_iterations = int(info.pdlp_iteration_count)
                iteration_count = simplex_iterations + ipm_iterations + pdlp_iterations
                statistics.update(
                    {
                        "simplex_iteration_count": simplex_iterations,
                        "ipm_iteration_count": ipm_iterations,
                        "pdlp_iteration_count": pdlp_iterations,
                    }
                )
                if self._model.is_mip:
                    statistics["mip_node_count"] = int(info.mip_node_count)
                    if np.isfinite(info.mip_gap):
                        statistics["mip_gap"] = float(info.mip_gap)

            runtime = float(self._highs.getRunTime())
            wall_time = runtime if np.isfinite(runtime) else None
            message = model_status_text
            error_message = None
            if status == "ERROR":
                error_message = model_status_text
            result = SolveResult(
                status=status,
                objective_value=objective_value,
                primal_solution=primal,
                dual_solution=dual,
                wall_time_seconds=wall_time,
                iteration_count=iteration_count,
                solver_name=self.solver_name,
                details=statistics,
                success=status in {"OPTIMAL", "FEASIBLE"},
                message=message,
                termination_reason=termination_reason,
                error_message=error_message,
            )
        except Exception as error:
            result = SolveResult(
                status="ERROR",
                solver_name=self.solver_name,
                message=f"HiGHS solve failed: {error}",
                termination_reason="BACKEND_ERROR",
                error_message=str(error),
            )

        if telemetry_callback is not None:
            telemetry_callback(
                {
                    "event": "solve_finished",
                    "status": result.status,
                    "objective_value": result.objective_value,
                    "iteration_count": result.iteration_count,
                    "wall_time_seconds": result.wall_time_seconds,
                }
            )
        return result

    def _set_option(self, name: str, value: Any) -> None:
        status = self._highs.setOptionValue(name, value)
        if status == highspy.HighsStatus.kError:
            raise SolverBackendError(f"HiGHS rejected option {name}={value!r}.")

    @staticmethod
    def _normalize_status(model_status, has_solution: bool) -> tuple[str, str]:
        exact_statuses = {
            highspy.HighsModelStatus.kOptimal: "OPTIMAL",
            highspy.HighsModelStatus.kInfeasible: "INFEASIBLE",
            highspy.HighsModelStatus.kUnbounded: "UNBOUNDED",
            highspy.HighsModelStatus.kTimeLimit: "TIME_LIMIT",
        }
        if model_status in exact_statuses:
            normalized = exact_statuses[model_status]
            if normalized == "TIME_LIMIT" and has_solution:
                return "FEASIBLE", "TIME_LIMIT"
            return normalized, normalized

        feasible_stop_statuses = {
            highspy.HighsModelStatus.kObjectiveBound,
            highspy.HighsModelStatus.kObjectiveTarget,
            highspy.HighsModelStatus.kIterationLimit,
            highspy.HighsModelStatus.kSolutionLimit,
            highspy.HighsModelStatus.kInterrupt,
            highspy.HighsModelStatus.kHighsInterrupt,
            highspy.HighsModelStatus.kMemoryLimit,
        }
        if model_status in feasible_stop_statuses:
            if has_solution:
                return "FEASIBLE", model_status.name.removeprefix("k").upper()
            if model_status == highspy.HighsModelStatus.kIterationLimit:
                return "TIME_LIMIT", "ITERATION_LIMIT"
            return "ERROR", model_status.name.removeprefix("k").upper()
        if model_status == highspy.HighsModelStatus.kUnboundedOrInfeasible:
            return "ERROR", "UNBOUNDED_OR_INFEASIBLE"
        return "ERROR", model_status.name.removeprefix("k").upper()