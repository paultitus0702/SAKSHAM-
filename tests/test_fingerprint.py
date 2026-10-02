"""Phase 3 structural fingerprint and numerical health tests."""

import numpy as np
import pytest
from scipy import sparse

from saksham.canonical.model import (
    CanonicalModel,
    ConstraintSense,
    VarType,
)
from saksham.fingerprint.analyzer import StructuralFingerprinter
from saksham.health.conditioner import NumericalHealthMonitor


def make_model(
    matrix,
    constraint_senses=None,
    var_types=None,
    name="phase3_test",
):
    matrix = sparse.csr_matrix(matrix, dtype=np.float64)
    num_constraints, num_variables = matrix.shape
    if constraint_senses is None:
        constraint_senses = [ConstraintSense.LESS_EQUAL] * num_constraints
    if var_types is None:
        var_types = [VarType.CONTINUOUS] * num_variables
    lower_bounds = np.zeros(num_variables, dtype=np.float64)
    upper_bounds = np.full(num_variables, np.inf, dtype=np.float64)
    for index, var_type in enumerate(var_types):
        if var_type == VarType.BINARY:
            upper_bounds[index] = 1.0

    return CanonicalModel(
        name=name,
        c=np.ones(num_variables, dtype=np.float64),
        A=matrix,
        rhs=np.ones(num_constraints, dtype=np.float64),
        constraint_senses=constraint_senses,
        l_var=lower_bounds,
        u_var=upper_bounds,
        var_types=var_types,
        var_names=[f"x{index}" for index in range(num_variables)],
        constr_names=[f"r{index}" for index in range(num_constraints)],
    )


def test_fingerprint_reports_sparse_matrix_and_coefficient_metrics():
    model = make_model([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, -2.0, 0.0, 0.0, 0.0]])

    result = StructuralFingerprinter().analyze(model)

    assert result.num_vars == 5
    assert result.num_constrs == 2
    assert result.num_nonzeros == 2
    assert result.density == pytest.approx(0.2)
    assert result.raw_metrics["row_nnz"] == [1, 1]
    assert result.raw_metrics["column_nnz"] == [1, 1, 0, 0, 0]
    assert result.raw_metrics["row_sparsity"] == pytest.approx([0.8, 0.8])
    assert result.raw_metrics["column_sparsity"] == pytest.approx(
        [0.5, 0.5, 1.0, 1.0, 1.0]
    )
    assert result.raw_metrics["coefficient_min_abs"] == 1.0
    assert result.raw_metrics["coefficient_max_abs"] == 2.0
    assert result.raw_metrics["coefficient_mean_abs"] == 1.5
    assert result.raw_metrics["coefficient_std_abs"] == 0.5
    assert result.raw_metrics["coefficient_dynamic_range"] == 2.0
    assert "sparse" in result.tags
    assert result.status == "OK"


def test_fingerprint_detects_dense_matrix():
    result = StructuralFingerprinter().analyze(make_model([[1.0, 2.0], [3.0, 4.0]]))

    assert result.density == 1.0
    assert result.raw_metrics["row_nnz"] == [2, 2]
    assert result.raw_metrics["column_nnz"] == [2, 2]
    assert "dense" in result.tags


def test_fingerprint_reports_variable_and_constraint_types():
    model = make_model(
        [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
        constraint_senses=[ConstraintSense.EQUAL, ConstraintSense.EQUAL],
        var_types=[VarType.CONTINUOUS, VarType.INTEGER, VarType.BINARY],
    )

    result = StructuralFingerprinter().analyze(model)

    assert result.raw_metrics["constraint_sense_counts"] == {
        "equal": 2,
        "less_equal": 0,
        "greater_equal": 0,
    }
    assert result.raw_metrics["variable_type_counts"] == {
        "continuous": 1,
        "integer": 1,
        "binary": 1,
    }
    assert "mixed_integer" in result.tags
    assert "equality_heavy" in result.tags


def test_fingerprint_counts_each_constraint_sense():
    model = make_model(
        [[1.0], [2.0], [3.0]],
        constraint_senses=[
            ConstraintSense.EQUAL,
            ConstraintSense.LESS_EQUAL,
            ConstraintSense.GREATER_EQUAL,
        ],
    )

    result = StructuralFingerprinter().analyze(model)

    assert result.raw_metrics["constraint_sense_counts"] == {
        "equal": 1,
        "less_equal": 1,
        "greater_equal": 1,
    }


def test_fingerprint_counts_conservation_and_binary_knapsack_candidates():
    model = make_model(
        [[1.0, -1.0, 0.0], [2.0, 3.0, 0.0]],
        constraint_senses=[ConstraintSense.EQUAL, ConstraintSense.LESS_EQUAL],
        var_types=[VarType.BINARY, VarType.BINARY, VarType.CONTINUOUS],
    )

    result = StructuralFingerprinter().analyze(model)

    assert result.conservation_rows_detected == 1
    assert result.knapsack_rows_detected == 1
    assert "potentially_network_like" not in result.tags


def test_fingerprint_marks_network_pattern_as_candidate():
    model = make_model(
        [[1.0, 2.0], [-1.0, -2.0]],
        constraint_senses=[ConstraintSense.EQUAL, ConstraintSense.EQUAL],
    )

    result = StructuralFingerprinter().analyze(model)

    assert "potentially_network_like" in result.tags
    assert result.raw_metrics["block_structure_status"] == "unknown"
    assert result.block_angular_candidate is False


def test_fingerprint_reports_zero_rows_and_columns():
    model = make_model([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    result = StructuralFingerprinter().analyze(model)

    assert result.raw_metrics["zero_row_count"] == 1
    assert result.raw_metrics["zero_column_count"] == 2


def test_fingerprint_ignores_explicit_sparse_zeros_without_mutating_model():
    matrix = sparse.csr_matrix(
        (np.array([0.0, 2.0]), np.array([0, 1]), np.array([0, 2])),
        shape=(1, 2),
    )
    model = make_model(matrix)
    stored_nonzeros = model.A.nnz

    result = StructuralFingerprinter().analyze(model)

    assert result.num_nonzeros == 1
    assert result.raw_metrics["stored_coefficient_count"] == stored_nonzeros
    assert model.A.nnz == stored_nonzeros


def test_health_reports_well_scaled_coefficients():
    report = NumericalHealthMonitor().evaluate(make_model([[1.0, 0.0], [0.0, 10.0]]))

    assert report.status == "HEALTHY"
    assert report.dynamic_range == 10.0
    assert report.coefficient_min_abs == 1.0
    assert report.coefficient_max_abs == 10.0
    assert report.zero_row_count == 0
    assert report.zero_column_count == 0
    assert report.estimated_condition_number is None


def test_health_warns_for_poorly_scaled_coefficients():
    model = make_model([[1e-8, 0.0], [0.0, 1e4]])
    report = NumericalHealthMonitor().evaluate(model)

    assert report.status == "WARNING"
    assert report.numerical_warning is True
    assert report.dynamic_range == pytest.approx(1e12)


def test_health_warns_for_zero_rows_and_columns():
    report = NumericalHealthMonitor().evaluate(make_model([[1.0, 0.0], [0.0, 0.0]]))

    assert report.status == "WARNING"
    assert report.zero_row_count == 1
    assert report.zero_column_count == 1


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_health_marks_non_finite_matrix_coefficients_invalid(bad_value):
    report = NumericalHealthMonitor().evaluate(make_model([[1.0, bad_value]]))

    assert report.status == "INVALID"
    assert report.numerical_warning is True
    assert report.has_non_finite_coefficients is True


def test_health_reports_unavailable_for_empty_coefficient_matrix():
    report = NumericalHealthMonitor().evaluate(make_model(np.zeros((2, 3))))

    assert report.status == "UNAVAILABLE"
    assert report.dynamic_range is None
    assert report.coefficient_min_abs is None
    assert report.coefficient_max_abs is None
    assert report.zero_row_count == 2
    assert report.zero_column_count == 3


def test_health_reports_unavailable_for_zero_constraints():
    report = NumericalHealthMonitor().evaluate(make_model(np.zeros((0, 3))))

    assert report.status == "UNAVAILABLE"
    assert report.zero_row_count == 0
    assert report.zero_column_count == 3


def test_health_accepts_configured_dynamic_range_threshold():
    model = make_model([[1.0, 0.0], [0.0, 100.0]])
    report = NumericalHealthMonitor(dynamic_range_warning_threshold=50.0).evaluate(model)

    assert report.status == "WARNING"
    assert report.dynamic_range_warning_threshold == 50.0


def test_analysis_and_health_do_not_densify_sparse_matrices(monkeypatch):
    model = make_model(sparse.eye(100, format="csr"))

    def fail_toarray(*args, **kwargs):
        raise AssertionError("Sparse matrix was densified")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", fail_toarray)

    StructuralFingerprinter().analyze(model)
    NumericalHealthMonitor().evaluate(model)

    assert sparse.isspmatrix_csr(model.A)