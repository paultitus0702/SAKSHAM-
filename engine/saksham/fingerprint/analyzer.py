"""Deterministic structural metrics for canonical optimization models."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..canonical.model import CanonicalModel
from ..canonical.model import ConstraintSense, VarType
from ..canonical.validator import ModelValidator


_SPARSE_DENSITY_THRESHOLD = 0.2
_HEAVY_SHARE_THRESHOLD = 0.5


@dataclass
class FingerprintResult:
    """Stores measured structural properties and conservative candidate signals."""
    num_vars: int = 0
    num_constrs: int = 0
    num_nonzeros: int = 0
    density: float = 0.0
    conservation_rows_detected: int = 0
    knapsack_rows_detected: int = 0
    block_angular_candidate: bool = False
    tags: List[str] = field(default_factory=list)
    raw_metrics: Dict[str, Any] = field(default_factory=dict)
    status: str = "UNAVAILABLE"


class StructuralFingerprinter:
    """Extract deterministic structure metrics from a CanonicalModel."""

    def analyze(self, model: CanonicalModel) -> FingerprintResult:
        """Analyze matrix sparsity and conservative row/column structure."""
        validation = ModelValidator.validate(model)
        if not validation.is_valid:
            return FingerprintResult(
                num_vars=model.num_vars,
                num_constrs=model.num_constrs,
                num_nonzeros=model.num_nonzeros,
                density=model.density,
                tags=["invalid_model"],
                raw_metrics={"validation_errors": validation.errors},
                status="INVALID",
            )

        matrix = model.A.tocsr(copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()

        num_vars = model.num_vars
        num_constrs = model.num_constrs
        row_nnz = np.diff(matrix.indptr)
        column_nnz = np.bincount(matrix.indices, minlength=num_vars)
        num_nonzeros = int(matrix.nnz)
        total_entries = num_vars * num_constrs
        density = float(num_nonzeros / total_entries) if total_entries else 0.0

        sense_counts = {
            "equal": sum(sense == ConstraintSense.EQUAL for sense in model.constraint_senses),
            "less_equal": sum(
                sense == ConstraintSense.LESS_EQUAL for sense in model.constraint_senses
            ),
            "greater_equal": sum(
                sense == ConstraintSense.GREATER_EQUAL
                for sense in model.constraint_senses
            ),
        }
        type_counts = {
            "continuous": sum(var_type == VarType.CONTINUOUS for var_type in model.var_types),
            "integer": sum(var_type == VarType.INTEGER for var_type in model.var_types),
            "binary": sum(var_type == VarType.BINARY for var_type in model.var_types),
        }

        coefficient_abs = np.abs(matrix.data)
        coefficient_min = float(coefficient_abs.min()) if coefficient_abs.size else None
        coefficient_max = float(coefficient_abs.max()) if coefficient_abs.size else None
        coefficient_mean = float(coefficient_abs.mean()) if coefficient_abs.size else None
        coefficient_std = float(coefficient_abs.std()) if coefficient_abs.size else None
        coefficient_dynamic_range = (
            float(coefficient_max / coefficient_min)
            if coefficient_min is not None
            else None
        )

        conservation_rows, knapsack_rows = self._classify_rows(model, matrix)
        tags = self._build_tags(model, density, sense_counts, type_counts, total_entries)
        network_candidate = self._is_network_candidate(matrix, num_vars)
        if network_candidate:
            tags.append("potentially_network_like")

        row_sparsity = (
            (1.0 - row_nnz / num_vars).tolist() if num_vars else []
        )
        column_sparsity: List[Optional[float]] = (
            (1.0 - column_nnz / num_constrs).tolist()
            if num_constrs
            else [None] * num_vars
        )
        raw_metrics: Dict[str, Any] = {
            "row_nnz": row_nnz.astype(int).tolist(),
            "column_nnz": column_nnz.astype(int).tolist(),
            "row_sparsity": row_sparsity,
            "column_sparsity": column_sparsity,
            "zero_row_count": int(np.count_nonzero(row_nnz == 0)),
            "zero_column_count": int(np.count_nonzero(column_nnz == 0)),
            "stored_coefficient_count": int(model.A.nnz),
            "explicit_zero_count": int(np.count_nonzero(model.A.data == 0)),
            "coefficient_min_abs": coefficient_min,
            "coefficient_max_abs": coefficient_max,
            "coefficient_mean_abs": coefficient_mean,
            "coefficient_std_abs": coefficient_std,
            "coefficient_dynamic_range": coefficient_dynamic_range,
            "constraint_sense_counts": sense_counts,
            "variable_type_counts": type_counts,
            "block_structure_status": "unknown",
            "network_structure_status": "candidate" if network_candidate else "unknown",
        }

        return FingerprintResult(
            num_vars=num_vars,
            num_constrs=num_constrs,
            num_nonzeros=num_nonzeros,
            density=density,
            conservation_rows_detected=conservation_rows,
            knapsack_rows_detected=knapsack_rows,
            block_angular_candidate=False,
            tags=tags,
            raw_metrics=raw_metrics,
            status="OK",
        )

    @staticmethod
    def _build_tags(
        model: CanonicalModel,
        density: float,
        sense_counts: Dict[str, int],
        type_counts: Dict[str, int],
        total_entries: int,
    ) -> List[str]:
        tags: List[str] = []
        if total_entries:
            tags.append(
                "sparse" if density <= _SPARSE_DENSITY_THRESHOLD else "dense"
            )
        else:
            tags.append("density_unknown")

        if model.num_vars:
            if type_counts["continuous"] == model.num_vars:
                tags.append("continuous")
            else:
                tags.append("mixed_integer")
            if type_counts["binary"] / model.num_vars >= _HEAVY_SHARE_THRESHOLD:
                tags.append("binary_heavy")
            if type_counts["integer"] / model.num_vars >= _HEAVY_SHARE_THRESHOLD:
                tags.append("integer_heavy")

        if model.num_constrs:
            equal_share = sense_counts["equal"] / model.num_constrs
            inequality_count = (
                sense_counts["less_equal"] + sense_counts["greater_equal"]
            )
            inequality_share = inequality_count / model.num_constrs
            if equal_share >= _HEAVY_SHARE_THRESHOLD:
                tags.append("equality_heavy")
            if inequality_share > _HEAVY_SHARE_THRESHOLD:
                tags.append("inequality_heavy")
        return tags

    @staticmethod
    def _classify_rows(model: CanonicalModel, matrix) -> tuple[int, int]:
        conservation_rows = 0
        knapsack_rows = 0
        for row_index, sense in enumerate(model.constraint_senses):
            start, end = matrix.indptr[row_index : row_index + 2]
            values = matrix.data[start:end]
            columns = matrix.indices[start:end]
            if values.size < 2:
                continue

            if sense == ConstraintSense.EQUAL:
                scale = max(1.0, float(np.abs(values).sum()))
                balanced = abs(float(values.sum())) <= 1e-12 * scale
                has_both_signs = np.any(values > 0) and np.any(values < 0)
                if balanced and has_both_signs:
                    conservation_rows += 1

            if (
                sense == ConstraintSense.LESS_EQUAL
                and np.all(values > 0)
                and np.isfinite(model.rhs[row_index])
                and model.rhs[row_index] >= 0
                and all(model.var_types[column] == VarType.BINARY for column in columns)
            ):
                knapsack_rows += 1
        return conservation_rows, knapsack_rows

    @staticmethod
    def _is_network_candidate(matrix, num_vars: int) -> bool:
        if num_vars == 0:
            return False
        column_matrix = matrix.tocsc()
        if np.any(np.diff(column_matrix.indptr) != 2):
            return False
        for column_index in range(num_vars):
            start, end = column_matrix.indptr[column_index : column_index + 2]
            values = column_matrix.data[start:end]
            if not (values[0] * values[1] < 0):
                return False
            if not np.isclose(abs(values[0]), abs(values[1]), rtol=0.0, atol=1e-12):
                return False
        return True
