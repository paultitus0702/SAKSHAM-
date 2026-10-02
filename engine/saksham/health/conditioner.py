"""Sparse-safe numerical diagnostics for canonical optimization models."""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy import sparse

from ..canonical.model import CanonicalModel
from ..canonical.validator import ModelValidator


DEFAULT_DYNAMIC_RANGE_WARNING_THRESHOLD = 1e7


@dataclass
class HealthReport:
    dynamic_range: Optional[float] = None
    estimated_condition_number: Optional[float] = None
    numerical_warning: bool = False
    details: str = "No numerical health evaluation has been performed."
    status: str = "UNAVAILABLE"
    coefficient_min_abs: Optional[float] = None
    coefficient_max_abs: Optional[float] = None
    nonzero_coefficient_count: int = 0
    stored_coefficient_count: int = 0
    explicit_zero_count: int = 0
    zero_row_count: int = 0
    zero_column_count: int = 0
    has_non_finite_coefficients: bool = False
    dynamic_range_warning_threshold: float = DEFAULT_DYNAMIC_RANGE_WARNING_THRESHOLD
    validation_errors: List[str] = field(default_factory=list)


class NumericalHealthMonitor:
    """Monitor coefficient scaling and sparse matrix degeneracies."""

    def __init__(
        self,
        dynamic_range_warning_threshold: float = DEFAULT_DYNAMIC_RANGE_WARNING_THRESHOLD,
    ) -> None:
        threshold = float(dynamic_range_warning_threshold)
        if not np.isfinite(threshold) or threshold < 1.0:
            raise ValueError("Dynamic-range warning threshold must be finite and >= 1.")
        self.dynamic_range_warning_threshold = threshold

    def evaluate(self, model: CanonicalModel) -> HealthReport:
        """Validate and evaluate the model's sparse constraint coefficients."""
        validation = ModelValidator.validate(model)
        matrix = model.A if sparse.issparse(model.A) else None
        stored_data = matrix.data if matrix is not None else np.array([], dtype=float)
        has_non_finite = bool(np.any(~np.isfinite(stored_data)))
        report = HealthReport(
            status="INVALID",
            numerical_warning=True,
            details="Model validation failed; numerical metrics are unavailable.",
            stored_coefficient_count=int(matrix.nnz) if matrix is not None else 0,
            explicit_zero_count=int(np.count_nonzero(stored_data == 0)),
            has_non_finite_coefficients=has_non_finite,
            dynamic_range_warning_threshold=self.dynamic_range_warning_threshold,
            validation_errors=list(validation.errors),
        )
        if not validation.is_valid:
            report.details = " ".join(validation.errors)
            return report

        canonical_matrix = model.A.tocsr(copy=True)
        canonical_matrix.sum_duplicates()
        canonical_matrix.eliminate_zeros()
        row_nnz = np.diff(canonical_matrix.indptr)
        column_nnz = np.bincount(
            canonical_matrix.indices, minlength=model.num_vars
        )
        coefficient_abs = np.abs(canonical_matrix.data)

        report.nonzero_coefficient_count = int(canonical_matrix.nnz)
        report.zero_row_count = int(np.count_nonzero(row_nnz == 0))
        report.zero_column_count = int(np.count_nonzero(column_nnz == 0))

        if coefficient_abs.size == 0:
            report.status = "UNAVAILABLE"
            report.numerical_warning = False
            report.details = (
                "No nonzero constraint coefficients are available for scale analysis."
            )
            return report

        report.coefficient_min_abs = float(coefficient_abs.min())
        report.coefficient_max_abs = float(coefficient_abs.max())
        report.dynamic_range = float(
            report.coefficient_max_abs / report.coefficient_min_abs
        )

        warnings: List[str] = []
        if report.dynamic_range > self.dynamic_range_warning_threshold:
            warnings.append(
                "Coefficient dynamic range "
                f"{report.dynamic_range:.6g} exceeds configured threshold "
                f"{self.dynamic_range_warning_threshold:.6g}."
            )
        if report.zero_row_count:
            warnings.append(f"Matrix contains {report.zero_row_count} zero row(s).")
        if report.zero_column_count:
            warnings.append(
                f"Matrix contains {report.zero_column_count} zero column(s)."
            )

        if warnings:
            report.status = "WARNING"
            report.numerical_warning = True
            report.details = " ".join(warnings)
        else:
            report.status = "HEALTHY"
            report.numerical_warning = False
            report.details = (
                "Coefficient scale is within threshold; no zero rows or columns found. "
                "Condition number was not estimated."
            )
        return report
