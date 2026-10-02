"""
Mathematical Model Validator for SAKSHAM Canonical Representation.

Performs strict, zero-tolerance structural and dimensional audits.
Does NOT silently repair invalid mathematical models — every inconsistency
is reported as a named error so the caller can decide what to do.

Design principles
-----------------
- Collect ALL errors in a single pass (do not short-circuit on first error).
- Return a structured ``ValidationReport`` that is easy to inspect in code
  and easy to serialise for the API layer.
- Raise ``ModelValidationError`` only when the caller explicitly requests it
  via ``raise_on_error=True``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import sparse

from .model import CanonicalModel, ConstraintSense, VarType


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class ModelValidationError(ValueError):
    """
    Raised when a CanonicalModel fails mathematical validation.

    Attributes
    ----------
    errors : list of str
        Individual error messages collected during validation.
    """

    def __init__(self, message: str, errors: Optional[List[str]] = None) -> None:
        super().__init__(message)
        self.errors: List[str] = errors or [message]


# ---------------------------------------------------------------------------
# Structured result
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """
    Detailed audit report returned by ``ModelValidator.validate()``.

    Attributes
    ----------
    is_valid : bool
        True only when the error list is empty.
    errors : list of str
        Critical problems that make the model mathematically unusable.
    warnings : list of str
        Non-fatal observations (e.g. empty rows, redundant entries).
    summary : dict
        A copy of ``CanonicalModel.summary()`` — only populated when
        ``is_valid`` is True.
    """

    is_valid: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the report to a plain dictionary."""
        return {
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class ModelValidator:
    """
    Audits a ``CanonicalModel`` for mathematical consistency and
    dimensional integrity.

    All checks are class methods so the validator is stateless and
    thread-safe.
    """

    # Maximum number of individual bound-violation messages before truncation
    _MAX_BOUND_ERRORS = 5

    @classmethod
    def validate(
        cls,
        model: CanonicalModel,
        raise_on_error: bool = False,
    ) -> ValidationReport:
        """
        Run the full validation suite against *model*.

        Parameters
        ----------
        model : CanonicalModel
            The instance to inspect.
        raise_on_error : bool
            If True and any errors are found, raise ``ModelValidationError``
            with the full error list attached.

        Returns
        -------
        ValidationReport
            Structured result containing errors, warnings, and (on success)
            model summary.

        Raises
        ------
        ModelValidationError
            Only when ``raise_on_error=True`` and errors exist.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Snapshot counts once; individual checks must not call model.num_vars
        # again because var_names may be inconsistent with other arrays.
        n = len(model.var_names)
        m = len(model.constr_names)

        cls._check_variable_count(n, errors)
        cls._check_objective_vector(model, n, errors)
        cls._check_variable_bounds_vectors(model, n, errors)
        cls._check_variable_types_list(model, n, errors)
        cls._check_variable_name_uniqueness(model, n, errors)
        cls._check_constraint_matrix(model, m, n, errors, warnings)
        cls._check_rhs_vector(model, m, errors)
        cls._check_constraint_senses_list(model, m, errors)
        cls._check_constraint_name_uniqueness(model, m, errors)
        cls._check_bound_consistency(model, n, errors)
        cls._check_binary_variable_bounds(model, n, errors)
        cls._check_empty_constraint_rows(model, m, n, warnings)

        is_valid = len(errors) == 0
        report = ValidationReport(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            summary=model.summary() if is_valid else {},
        )

        if raise_on_error and not is_valid:
            joined = "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(errors))
            raise ModelValidationError(
                f"Model '{model.name}' failed validation with "
                f"{len(errors)} error(s):\n{joined}",
                errors=errors,
            )

        return report

    # ------------------------------------------------------------------
    # Individual check methods — each appends to *errors* or *warnings*
    # ------------------------------------------------------------------

    @classmethod
    def _check_variable_count(cls, n: int, errors: List[str]) -> None:
        if n <= 0:
            errors.append(
                f"Model has {n} variable(s). At least 1 variable is required."
            )

    @classmethod
    def _check_objective_vector(
        cls, model: CanonicalModel, n: int, errors: List[str]
    ) -> None:
        if model.c is None:
            errors.append("Objective coefficient vector c is None.")
            return
        if model.c.ndim != 1:
            errors.append(
                f"Objective vector c must be 1-dimensional; got shape {model.c.shape}."
            )
            return
        if len(model.c) != n:
            errors.append(
                f"Objective vector c has length {len(model.c)}, "
                f"expected {n} (number of variables)."
            )
        # NaN check (Inf in objective is technically legal — no constraint there)
        if model.c is not None and np.isnan(model.c).any():
            bad = np.where(np.isnan(model.c))[0].tolist()
            errors.append(
                f"Objective vector c contains NaN at variable indices "
                f"{bad[:10]}{'...' if len(bad) > 10 else ''}."
            )

    @classmethod
    def _check_variable_bounds_vectors(
        cls, model: CanonicalModel, n: int, errors: List[str]
    ) -> None:
        for attr, label in (("l_var", "lower-bound"), ("u_var", "upper-bound")):
            vec = getattr(model, attr)
            if vec is None:
                errors.append(f"Variable {label} vector {attr} is None.")
                continue
            if vec.ndim != 1:
                errors.append(
                    f"Variable {label} vector {attr} must be 1-D; got shape {vec.shape}."
                )
                continue
            if len(vec) != n:
                errors.append(
                    f"Variable {label} vector {attr} has length {len(vec)}, "
                    f"expected {n}."
                )
            if np.isnan(vec).any():
                bad = np.where(np.isnan(vec))[0].tolist()
                errors.append(
                    f"Variable {label} vector {attr} contains NaN at indices "
                    f"{bad[:10]}{'...' if len(bad) > 10 else ''}."
                )

    @classmethod
    def _check_variable_types_list(
        cls, model: CanonicalModel, n: int, errors: List[str]
    ) -> None:
        if len(model.var_types) != n:
            errors.append(
                f"var_types has {len(model.var_types)} entries, expected {n}."
            )
            return
        # Verify every entry is a valid VarType member
        valid_types = set(VarType)
        for idx, vt in enumerate(model.var_types):
            if vt not in valid_types:
                vname = model.var_names[idx] if idx < len(model.var_names) else f"x_{idx}"
                errors.append(
                    f"Variable '{vname}' (index {idx}) has invalid var_type '{vt}'. "
                    f"Must be one of {[v.value for v in VarType]}."
                )

    @classmethod
    def _check_variable_name_uniqueness(
        cls, model: CanonicalModel, n: int, errors: List[str]
    ) -> None:
        if len(model.var_names) != n:
            # Dimension mismatch already caught by num_vars; skip
            return
        seen: Dict[str, int] = {}
        for idx, name in enumerate(model.var_names):
            if not name or not isinstance(name, str):
                errors.append(
                    f"Variable at index {idx} has an invalid name: {name!r}."
                )
            elif name in seen:
                errors.append(
                    f"Duplicate variable name '{name}' at indices "
                    f"{seen[name]} and {idx}."
                )
            else:
                seen[name] = idx

    @classmethod
    def _check_constraint_matrix(
        cls,
        model: CanonicalModel,
        m: int,
        n: int,
        errors: List[str],
        warnings: List[str],
    ) -> None:
        if model.A is None:
            errors.append("Constraint matrix A is None.")
            return

        expected_shape = (m, n)
        if model.A.shape != expected_shape:
            errors.append(
                f"Constraint matrix A has shape {model.A.shape}, "
                f"expected {expected_shape} (m={m}, n={n})."
            )

        if not sparse.isspmatrix_csr(model.A):
            warnings.append(
                "Constraint matrix A is not in CSR format. "
                "Consider converting with A.tocsr() for optimal row-access performance."
            )

        if model.A.data is not None and len(model.A.data) > 0:
            if np.isnan(model.A.data).any():
                errors.append("Constraint matrix A contains NaN coefficient values.")
            if np.isinf(model.A.data).any():
                errors.append(
                    "Constraint matrix A contains infinite coefficient values."
                )

    @classmethod
    def _check_rhs_vector(
        cls, model: CanonicalModel, m: int, errors: List[str]
    ) -> None:
        if model.rhs is None:
            errors.append("RHS vector rhs is None.")
            return
        if model.rhs.ndim != 1:
            errors.append(
                f"RHS vector rhs must be 1-D; got shape {model.rhs.shape}."
            )
            return
        if len(model.rhs) != m:
            errors.append(
                f"RHS vector rhs has length {len(model.rhs)}, "
                f"expected {m} (number of constraints)."
            )
        if np.isnan(model.rhs).any():
            bad = np.where(np.isnan(model.rhs))[0].tolist()
            errors.append(
                f"RHS vector contains NaN at constraint indices "
                f"{bad[:10]}{'...' if len(bad) > 10 else ''}."
            )

    @classmethod
    def _check_constraint_senses_list(
        cls, model: CanonicalModel, m: int, errors: List[str]
    ) -> None:
        if len(model.constraint_senses) != m:
            errors.append(
                f"constraint_senses has {len(model.constraint_senses)} entries, "
                f"expected {m}."
            )
            return
        valid_senses = set(ConstraintSense)
        for idx, sense in enumerate(model.constraint_senses):
            if sense not in valid_senses:
                cname = (
                    model.constr_names[idx]
                    if idx < len(model.constr_names)
                    else f"c_{idx}"
                )
                errors.append(
                    f"Constraint '{cname}' (index {idx}) has invalid sense '{sense}'. "
                    f"Must be one of {[s.value for s in ConstraintSense]}."
                )

    @classmethod
    def _check_constraint_name_uniqueness(
        cls, model: CanonicalModel, m: int, errors: List[str]
    ) -> None:
        if len(model.constr_names) != m:
            return
        seen: Dict[str, int] = {}
        for idx, name in enumerate(model.constr_names):
            if not name or not isinstance(name, str):
                errors.append(
                    f"Constraint at index {idx} has an invalid name: {name!r}."
                )
            elif name in seen:
                errors.append(
                    f"Duplicate constraint name '{name}' at indices "
                    f"{seen[name]} and {idx}."
                )
            else:
                seen[name] = idx

    @classmethod
    def _check_bound_consistency(
        cls, model: CanonicalModel, n: int, errors: List[str]
    ) -> None:
        """Verify l_var[j] <= u_var[j] for every variable j."""
        if (
            model.l_var is None
            or model.u_var is None
            or len(model.l_var) != n
            or len(model.u_var) != n
        ):
            return  # Dimension mismatch already reported

        bad_indices = np.where(model.l_var > model.u_var)[0]
        for count, idx in enumerate(bad_indices):
            if count >= cls._MAX_BOUND_ERRORS:
                remaining = len(bad_indices) - cls._MAX_BOUND_ERRORS
                errors.append(
                    f"... and {remaining} more variable(s) with lower bound > upper bound."
                )
                break
            vname = (
                model.var_names[idx] if idx < len(model.var_names) else f"x_{idx}"
            )
            errors.append(
                f"Variable '{vname}' (index {idx}) has inconsistent bounds: "
                f"l_var={model.l_var[idx]} > u_var={model.u_var[idx]}."
            )

    @classmethod
    def _check_binary_variable_bounds(
        cls, model: CanonicalModel, n: int, errors: List[str]
    ) -> None:
        """Binary variables must have bounds within [0, 1]."""
        if (
            len(model.var_types) != n
            or len(model.l_var) != n
            or len(model.u_var) != n
        ):
            return

        from .model import VarType as VT  # local import avoids circular risk
        for idx, vt in enumerate(model.var_types):
            if vt != VT.BINARY:
                continue
            lb, ub = model.l_var[idx], model.u_var[idx]
            vname = (
                model.var_names[idx] if idx < len(model.var_names) else f"x_{idx}"
            )
            if lb < 0.0 or ub > 1.0:
                errors.append(
                    f"Binary variable '{vname}' (index {idx}) has bounds "
                    f"[{lb}, {ub}] outside [0, 1]."
                )

    @classmethod
    def _check_empty_constraint_rows(
        cls,
        model: CanonicalModel,
        m: int,
        n: int,
        warnings: List[str],
    ) -> None:
        """Warn about rows that are entirely zero (no variable appears)."""
        if (
            model.A is None
            or model.A.shape != (m, n)
            or not sparse.isspmatrix_csr(model.A)
        ):
            return
        row_nnz = np.diff(model.A.indptr)
        empty_row_indices = np.where(row_nnz == 0)[0]
        if len(empty_row_indices) > 0:
            names = [
                model.constr_names[i]
                if i < len(model.constr_names)
                else f"c_{i}"
                for i in empty_row_indices[:5]
            ]
            suffix = (
                f" (showing first 5 of {len(empty_row_indices)})"
                if len(empty_row_indices) > 5
                else ""
            )
            warnings.append(
                f"Model contains {len(empty_row_indices)} empty constraint row(s) "
                f"(all-zero coefficients){suffix}: {names}."
            )
