"""Independent feasibility and objective checks for solver results.

SAKSHI recomputes constraints, bounds, integrality, and the objective from the
CanonicalModel. It does not trust solver-internal residuals and does not prove
global optimality.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..canonical.model import CanonicalModel, ConstraintSense, ObjectiveSense, VarType
from ..canonical.validator import ModelValidator
from ..solvers.base import SolveResult


@dataclass
class SakshiVerificationResult:
    is_valid: bool = False
    max_primal_violation: Optional[float] = None
    max_bound_violation: Optional[float] = None
    verdict: str = "UNAVAILABLE"
    details: Dict[str, Any] = field(default_factory=dict)
    status: str = "UNAVAILABLE"
    solver_status: Optional[str] = None
    bounds_satisfied: Optional[bool] = None
    constraints_satisfied: Optional[bool] = None
    integrality_satisfied: Optional[bool] = None
    objective_consistent: Optional[bool] = None
    max_constraint_violation: Optional[float] = None
    max_integrality_violation: Optional[float] = None
    objective_discrepancy: Optional[float] = None
    independently_calculated_objective: Optional[float] = None
    solver_reported_objective: Optional[float] = None
    tolerance: float = 1e-7
    feasibility_tolerance: float = 1e-7
    integrality_tolerance: float = 1e-6
    warnings: List[str] = field(default_factory=list)
    unavailable_checks: List[str] = field(default_factory=list)


class SakshiVerifier:
    """Verify solver output against the original canonical model."""

    def __init__(
        self,
        feasibility_tolerance: float = 1e-7,
        integrality_tolerance: float = 1e-6,
    ) -> None:
        self.feasibility_tolerance = self._validate_tolerance(
            feasibility_tolerance, "feasibility_tolerance"
        )
        self.integrality_tolerance = self._validate_tolerance(
            integrality_tolerance, "integrality_tolerance"
        )

    def verify(
        self,
        model: CanonicalModel,
        result: SolveResult,
    ) -> SakshiVerificationResult:
        """Independently check feasibility, variable domains, and objective value."""
        solver_status = getattr(result, "status", None) if result is not None else None
        base = SakshiVerificationResult(
            solver_status=solver_status,
            tolerance=self.feasibility_tolerance,
            feasibility_tolerance=self.feasibility_tolerance,
            integrality_tolerance=self.integrality_tolerance,
            details={"optimality_verified": False},
        )

        if result is None:
            base.status = "UNAVAILABLE"
            base.verdict = base.status
            base.unavailable_checks = ["solver_result", "all_solution_checks"]
            base.warnings.append("No solver result was supplied.")
            return base
        if not isinstance(result, SolveResult):
            return self._invalid(base, "Solver result has an unsupported type.")
        if not isinstance(model, CanonicalModel):
            return self._invalid(base, "Model is not a CanonicalModel instance.")

        try:
            validation = ModelValidator.validate(model)
        except Exception as error:
            return self._invalid(base, f"Model validation could not be completed: {error}")
        if not validation.is_valid:
            base.warnings.extend(validation.errors)
            return self._invalid(base, "CanonicalModel failed validation.")
        base.warnings.extend(validation.warnings)

        if not self._model_numbers_are_finite(model):
            return self._invalid(
                base,
                "Model contains non-finite objective, RHS, or matrix data required for verification.",
            )

        if not isinstance(solver_status, str) or not solver_status:
            base.status = "UNAVAILABLE"
            base.verdict = base.status
            base.unavailable_checks = ["solver_status", "all_solution_checks"]
            base.warnings.append("Solver result does not contain a usable status.")
            return base

        if solver_status.upper() not in {"OPTIMAL", "FEASIBLE"}:
            base.status = "UNAVAILABLE"
            base.verdict = base.status
            base.unavailable_checks = ["primal_solution", "bounds", "constraints", "integrality", "objective"]
            base.warnings.append(
                f"Solver status '{solver_status}' does not provide a solution eligible for verification."
            )
            return base

        if result.primal_solution is None:
            base.status = "UNAVAILABLE"
            base.verdict = base.status
            base.unavailable_checks = ["primal_solution", "bounds", "constraints", "integrality", "objective"]
            base.warnings.append("Solver result has no primal solution vector.")
            return base

        try:
            solution = np.asarray(result.primal_solution, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            return self._invalid(base, f"Primal solution cannot be interpreted numerically: {error}")

        if solution.ndim != 1 or solution.shape != (model.num_vars,):
            return self._invalid(
                base,
                f"Primal solution dimension {solution.shape} does not match {model.num_vars} variables.",
            )
        if not np.all(np.isfinite(solution)):
            return self._invalid(base, "Primal solution contains NaN or infinite values.")

        lower_violation = np.maximum(model.l_var - solution, 0.0)
        upper_violation = np.maximum(solution - model.u_var, 0.0)
        bound_violations = np.maximum(lower_violation, upper_violation)
        base.max_bound_violation = float(bound_violations.max(initial=0.0))
        base.bounds_satisfied = base.max_bound_violation <= self.feasibility_tolerance

        try:
            activities = np.asarray(model.A @ solution, dtype=np.float64).reshape(-1)
        except Exception as error:
            return self._invalid(base, f"Constraint activities could not be computed: {error}")
        if activities.shape != (model.num_constrs,) or not np.all(np.isfinite(activities)):
            return self._invalid(base, "Constraint activities are malformed or non-finite.")

        constraint_violations = np.zeros(model.num_constrs, dtype=np.float64)
        signed_residuals = activities - model.rhs
        for index, sense in enumerate(model.constraint_senses):
            if sense == ConstraintSense.LESS_EQUAL:
                constraint_violations[index] = max(signed_residuals[index], 0.0)
            elif sense == ConstraintSense.EQUAL:
                constraint_violations[index] = abs(signed_residuals[index])
            elif sense == ConstraintSense.GREATER_EQUAL:
                constraint_violations[index] = max(-signed_residuals[index], 0.0)
            else:
                return self._invalid(base, f"Unsupported constraint sense at row {index}.")
        if not np.all(np.isfinite(constraint_violations)):
            return self._invalid(base, "Constraint violations are non-finite.")
        base.max_constraint_violation = float(constraint_violations.max(initial=0.0))
        base.max_primal_violation = base.max_constraint_violation
        base.constraints_satisfied = (
            base.max_constraint_violation <= self.feasibility_tolerance
        )

        integrality_violations = np.zeros(model.num_vars, dtype=np.float64)
        for index, var_type in enumerate(model.var_types):
            if var_type == VarType.INTEGER:
                integrality_violations[index] = abs(solution[index] - np.rint(solution[index]))
            elif var_type == VarType.BINARY:
                integrality_violations[index] = min(
                    abs(solution[index]), abs(solution[index] - 1.0)
                )
        base.max_integrality_violation = float(integrality_violations.max(initial=0.0))
        base.integrality_satisfied = (
            base.max_integrality_violation <= self.integrality_tolerance
        )

        with np.errstate(over="ignore", invalid="ignore"):
            independently_calculated = float(model.c0 + model.c @ solution)
        if not np.isfinite(independently_calculated):
            return self._invalid(base, "Independently calculated objective is non-finite.")
        base.independently_calculated_objective = independently_calculated

        reported_objective = result.objective_value
        if reported_objective is None:
            base.objective_consistent = None
            base.unavailable_checks.append("solver_reported_objective")
            base.warnings.append("Solver-reported objective is unavailable for comparison.")
        else:
            try:
                reported_objective = float(reported_objective)
            except (TypeError, ValueError, OverflowError):
                return self._invalid(base, "Solver-reported objective is not numeric.")
            if not np.isfinite(reported_objective):
                return self._invalid(base, "Solver-reported objective is non-finite.")
            base.solver_reported_objective = reported_objective
            base.objective_discrepancy = abs(
                independently_calculated - reported_objective
            )
            objective_tolerance = self.feasibility_tolerance * max(
                1.0, abs(independently_calculated), abs(reported_objective)
            )
            base.objective_consistent = (
                base.objective_discrepancy <= objective_tolerance
            )

        base.details.update(
            {
                "objective_sense": model.sense.value,
                "constraint_residual_norm": float(np.linalg.norm(constraint_violations)),
                "constraint_violations": constraint_violations.tolist(),
                "optimality_verified": False,
            }
        )
        checks = [
            base.bounds_satisfied,
            base.constraints_satisfied,
            base.integrality_satisfied,
            base.objective_consistent,
        ]
        failed = any(check is False for check in checks)
        base.status = "FAILED" if failed else "VERIFIED"
        base.verdict = base.status
        base.is_valid = base.status == "VERIFIED"
        if not base.is_valid:
            base.warnings.append("One or more independently checked solution conditions failed.")
        return base

    @staticmethod
    def _validate_tolerance(value: float, name: str) -> float:
        tolerance = float(value)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
        return tolerance

    @staticmethod
    def _model_numbers_are_finite(model: CanonicalModel) -> bool:
        return bool(
            np.all(np.isfinite(model.A.data))
            and np.all(np.isfinite(model.c))
            and np.all(np.isfinite(model.rhs))
            and np.isfinite(model.c0)
        )

    @staticmethod
    def _invalid(
        result: SakshiVerificationResult,
        warning: str,
    ) -> SakshiVerificationResult:
        result.status = "INVALID"
        result.verdict = result.status
        result.is_valid = False
        result.warnings.append(warning)
        result.unavailable_checks.extend(
            check
            for check, value in (
                ("bounds", result.bounds_satisfied),
                ("constraints", result.constraints_satisfied),
                ("integrality", result.integrality_satisfied),
                ("objective", result.objective_consistent),
            )
            if value is None and check not in result.unavailable_checks
        )
        return result
