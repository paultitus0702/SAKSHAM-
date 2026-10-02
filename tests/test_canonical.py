"""
Phase 2A: CanonicalModel and Mathematical Validation Tests.

Every test verifies actual mathematical values, not merely that code executes.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from saksham.canonical.model import (
    CanonicalModel,
    ConstraintSense,
    ObjectiveSense,
    VarType,
    _to_float64_array,
)
from saksham.canonical.validator import (
    ModelValidationError,
    ModelValidator,
    ValidationReport,
)


# ===========================================================================
# Helper: build a minimal valid 2-variable, 1-constraint model
# ===========================================================================


def _make_model(
    *,
    name="test_model",
    sense=ObjectiveSense.MINIMIZE,
    c=(3.0, 2.0),
    A_dense=((1.0, 1.0),),
    rhs=(10.0,),
    senses=(ConstraintSense.LESS_EQUAL,),
    l_var=(0.0, 0.0),
    u_var=(np.inf, np.inf),
    var_types=(VarType.CONTINUOUS, VarType.CONTINUOUS),
    var_names=("x1", "x2"),
    constr_names=("c1",),
) -> CanonicalModel:
    """Convenience factory so individual tests can override only relevant parts."""
    A = sparse.csr_matrix(np.array(A_dense, dtype=np.float64))
    return CanonicalModel(
        name=name,
        sense=sense,
        c=np.array(c, dtype=np.float64),
        A=A,
        rhs=np.array(rhs, dtype=np.float64),
        constraint_senses=list(senses),
        l_var=np.array(l_var, dtype=np.float64),
        u_var=np.array(u_var, dtype=np.float64),
        var_types=list(var_types),
        var_names=list(var_names),
        constr_names=list(constr_names),
    )


# ===========================================================================
# 1. CanonicalModel – enumerations
# ===========================================================================


class TestEnumerations:
    def test_objective_sense_values(self):
        assert ObjectiveSense.MINIMIZE.value == "minimize"
        assert ObjectiveSense.MAXIMIZE.value == "maximize"

    def test_constraint_sense_less_equal(self):
        assert ConstraintSense.LESS_EQUAL.value == "<="

    def test_constraint_sense_equal(self):
        assert ConstraintSense.EQUAL.value == "=="

    def test_constraint_sense_greater_equal(self):
        assert ConstraintSense.GREATER_EQUAL.value == ">="

    def test_constraint_sense_from_string_mps_codes(self):
        assert ConstraintSense.from_string("L") == ConstraintSense.LESS_EQUAL
        assert ConstraintSense.from_string("E") == ConstraintSense.EQUAL
        assert ConstraintSense.from_string("G") == ConstraintSense.GREATER_EQUAL

    def test_constraint_sense_from_string_symbols(self):
        assert ConstraintSense.from_string("<=") == ConstraintSense.LESS_EQUAL
        assert ConstraintSense.from_string("=") == ConstraintSense.EQUAL
        assert ConstraintSense.from_string(">=") == ConstraintSense.GREATER_EQUAL

    def test_constraint_sense_from_string_invalid(self):
        with pytest.raises(ValueError, match="Unknown constraint sense string"):
            ConstraintSense.from_string("??")

    def test_var_type_values(self):
        assert VarType.CONTINUOUS.value == "continuous"
        assert VarType.INTEGER.value == "integer"
        assert VarType.BINARY.value == "binary"


# ===========================================================================
# 2. CanonicalModel – continuous minimisation
# ===========================================================================


class TestContinuousMinimisationModel:
    """
    Model:
        min  3 x1 + 2 x2
        s.t. x1 + x2  <= 10   (c1)
             x1       >=  2   (c2)
        0 <= x1 <= inf
        0 <= x2 <= inf
    """

    def setup_method(self):
        A = sparse.csr_matrix(
            np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        )
        self.model = CanonicalModel(
            name="continuous_min",
            sense=ObjectiveSense.MINIMIZE,
            c=np.array([3.0, 2.0]),
            A=A,
            rhs=np.array([10.0, 2.0]),
            constraint_senses=[ConstraintSense.LESS_EQUAL, ConstraintSense.GREATER_EQUAL],
            l_var=np.array([0.0, 0.0]),
            u_var=np.array([np.inf, np.inf]),
            var_types=[VarType.CONTINUOUS, VarType.CONTINUOUS],
            var_names=["x1", "x2"],
            constr_names=["c1", "c2"],
        )

    def test_num_vars(self):
        assert self.model.num_vars == 2

    def test_num_constrs(self):
        assert self.model.num_constrs == 2

    def test_num_nonzeros(self):
        assert self.model.num_nonzeros == 3   # A has 3 non-zeros

    def test_density(self):
        # 3 nnz / (2 cols * 2 rows) = 0.75
        assert self.model.density == pytest.approx(0.75)

    def test_objective_sense_is_minimize(self):
        assert self.model.sense == ObjectiveSense.MINIMIZE

    def test_objective_coefficients(self):
        np.testing.assert_array_equal(self.model.c, [3.0, 2.0])

    def test_rhs_values(self):
        np.testing.assert_array_equal(self.model.rhs, [10.0, 2.0])

    def test_constraint_senses(self):
        assert self.model.constraint_senses[0] == ConstraintSense.LESS_EQUAL
        assert self.model.constraint_senses[1] == ConstraintSense.GREATER_EQUAL

    def test_lower_bounds_are_zero(self):
        np.testing.assert_array_equal(self.model.l_var, [0.0, 0.0])

    def test_upper_bounds_are_inf(self):
        assert np.isposinf(self.model.u_var[0])
        assert np.isposinf(self.model.u_var[1])

    def test_var_types_are_continuous(self):
        assert all(vt == VarType.CONTINUOUS for vt in self.model.var_types)

    def test_var_names_correct(self):
        assert self.model.var_names == ["x1", "x2"]

    def test_constr_names_correct(self):
        assert self.model.constr_names == ["c1", "c2"]

    def test_var_name_to_idx_lookup(self):
        assert self.model.var_name_to_idx["x1"] == 0
        assert self.model.var_name_to_idx["x2"] == 1

    def test_constr_name_to_idx_lookup(self):
        assert self.model.constr_name_to_idx["c1"] == 0
        assert self.model.constr_name_to_idx["c2"] == 1

    def test_is_mip_false(self):
        assert self.model.is_mip is False

    def test_continuous_count(self):
        assert self.model.continuous_count == 2

    def test_integer_count_zero(self):
        assert self.model.integer_count == 0

    def test_validation_passes(self):
        report = ModelValidator.validate(self.model)
        assert report.is_valid is True
        assert report.errors == []

    def test_summary_keys(self):
        s = self.model.summary()
        assert s["num_variables"] == 2
        assert s["num_constraints"] == 2
        assert s["sense"] == "minimize"


# ===========================================================================
# 3. Maximisation model
# ===========================================================================


class TestMaximisationModel:
    """
    Model:
        max  5 x1 + 4 x2
        s.t. 6 x1 + 4 x2 <= 24
             x1 + 2 x2    <=  6
    """

    def setup_method(self):
        self.model = _make_model(
            name="maximisation_test",
            sense=ObjectiveSense.MAXIMIZE,
            c=(5.0, 4.0),
            A_dense=((6.0, 4.0), (1.0, 2.0)),
            rhs=(24.0, 6.0),
            senses=(ConstraintSense.LESS_EQUAL, ConstraintSense.LESS_EQUAL),
            constr_names=("resource", "time"),
        )

    def test_sense_is_maximize(self):
        assert self.model.sense == ObjectiveSense.MAXIMIZE

    def test_objective_coefficients(self):
        np.testing.assert_array_equal(self.model.c, [5.0, 4.0])

    def test_matrix_coefficient_row0(self):
        row = self.model.A.getrow(0).toarray().flatten()
        np.testing.assert_array_equal(row, [6.0, 4.0])

    def test_matrix_coefficient_row1(self):
        row = self.model.A.getrow(1).toarray().flatten()
        np.testing.assert_array_equal(row, [1.0, 2.0])

    def test_validation_passes(self):
        report = ModelValidator.validate(self.model)
        assert report.is_valid is True


# ===========================================================================
# 4. Equality constraint
# ===========================================================================


class TestEqualityConstraint:
    """
    Model with an equality row:
        min  x1 + x2
        s.t. x1 + x2 == 5  (balance)
    """

    def setup_method(self):
        self.model = _make_model(
            rhs=(5.0,),
            senses=(ConstraintSense.EQUAL,),
            constr_names=("balance",),
        )

    def test_constraint_sense_is_equal(self):
        assert self.model.constraint_senses[0] == ConstraintSense.EQUAL

    def test_rhs_value(self):
        assert self.model.rhs[0] == pytest.approx(5.0)

    def test_compute_row_bounds_equal(self):
        L, U = self.model.compute_row_bounds()
        assert L[0] == pytest.approx(5.0)
        assert U[0] == pytest.approx(5.0)

    def test_validation_passes(self):
        assert ModelValidator.validate(self.model).is_valid


# ===========================================================================
# 5. LESS_EQUAL constraint
# ===========================================================================


class TestLessEqualConstraint:
    def setup_method(self):
        self.model = _make_model(
            rhs=(10.0,),
            senses=(ConstraintSense.LESS_EQUAL,),
        )

    def test_sense(self):
        assert self.model.constraint_senses[0] == ConstraintSense.LESS_EQUAL

    def test_compute_row_bounds_le(self):
        L, U = self.model.compute_row_bounds()
        assert np.isneginf(L[0])
        assert U[0] == pytest.approx(10.0)


# ===========================================================================
# 6. GREATER_EQUAL constraint
# ===========================================================================


class TestGreaterEqualConstraint:
    def setup_method(self):
        self.model = _make_model(
            rhs=(4.0,),
            senses=(ConstraintSense.GREATER_EQUAL,),
        )

    def test_sense(self):
        assert self.model.constraint_senses[0] == ConstraintSense.GREATER_EQUAL

    def test_compute_row_bounds_ge(self):
        L, U = self.model.compute_row_bounds()
        assert L[0] == pytest.approx(4.0)
        assert np.isposinf(U[0])


# ===========================================================================
# 7. Binary variables
# ===========================================================================


class TestBinaryVariables:
    """
    Binary knapsack model:
        max  10 y1 + 15 y2
        s.t. 3 y1 + 4 y2 <= 7
        y1, y2 in {0, 1}
    """

    def setup_method(self):
        A = sparse.csr_matrix(np.array([[3.0, 4.0]], dtype=np.float64))
        self.model = CanonicalModel(
            name="binary_knapsack",
            sense=ObjectiveSense.MAXIMIZE,
            c=np.array([10.0, 15.0]),
            A=A,
            rhs=np.array([7.0]),
            constraint_senses=[ConstraintSense.LESS_EQUAL],
            l_var=np.array([0.0, 0.0]),
            u_var=np.array([1.0, 1.0]),
            var_types=[VarType.BINARY, VarType.BINARY],
            var_names=["y1", "y2"],
            constr_names=["weight"],
        )

    def test_var_types_binary(self):
        assert self.model.var_types[0] == VarType.BINARY
        assert self.model.var_types[1] == VarType.BINARY

    def test_binary_count(self):
        assert self.model.binary_count == 2

    def test_is_mip(self):
        assert self.model.is_mip is True

    def test_binary_bounds(self):
        np.testing.assert_array_equal(self.model.l_var, [0.0, 0.0])
        np.testing.assert_array_equal(self.model.u_var, [1.0, 1.0])

    def test_validation_passes(self):
        assert ModelValidator.validate(self.model).is_valid

    def test_objective_coefficients(self):
        np.testing.assert_array_equal(self.model.c, [10.0, 15.0])


# ===========================================================================
# 8. Integer variables
# ===========================================================================


class TestIntegerVariables:
    """
    min  7 z1 + 3 z2
    s.t. z1 + z2 >= 4
    0 <= z1, z2 <= 10, integer
    """

    def setup_method(self):
        A = sparse.csr_matrix(np.array([[1.0, 1.0]], dtype=np.float64))
        self.model = CanonicalModel(
            name="integer_model",
            sense=ObjectiveSense.MINIMIZE,
            c=np.array([7.0, 3.0]),
            A=A,
            rhs=np.array([4.0]),
            constraint_senses=[ConstraintSense.GREATER_EQUAL],
            l_var=np.array([0.0, 0.0]),
            u_var=np.array([10.0, 10.0]),
            var_types=[VarType.INTEGER, VarType.INTEGER],
            var_names=["z1", "z2"],
            constr_names=["demand"],
        )

    def test_var_types_integer(self):
        assert all(vt == VarType.INTEGER for vt in self.model.var_types)

    def test_integer_count(self):
        assert self.model.integer_count == 2

    def test_is_mip(self):
        assert self.model.is_mip is True

    def test_validation_passes(self):
        assert ModelValidator.validate(self.model).is_valid


# ===========================================================================
# 9. Mixed variable types (continuous + integer + binary)
# ===========================================================================


class TestMixedVariableTypes:
    """
    3-variable mixed model:
        min  x1 + 5 z1 + 100 y1
        x1: continuous, z1: integer, y1: binary
    """

    def setup_method(self):
        A = sparse.csr_matrix(np.eye(3, dtype=np.float64))
        self.model = CanonicalModel(
            name="mixed_types",
            sense=ObjectiveSense.MINIMIZE,
            c=np.array([1.0, 5.0, 100.0]),
            A=A,
            rhs=np.array([10.0, 8.0, 1.0]),
            constraint_senses=[
                ConstraintSense.LESS_EQUAL,
                ConstraintSense.LESS_EQUAL,
                ConstraintSense.LESS_EQUAL,
            ],
            l_var=np.array([0.0, 0.0, 0.0]),
            u_var=np.array([np.inf, 8.0, 1.0]),
            var_types=[VarType.CONTINUOUS, VarType.INTEGER, VarType.BINARY],
            var_names=["x1", "z1", "y1"],
            constr_names=["ub_x", "ub_z", "ub_y"],
        )

    def test_continuous_count(self):
        assert self.model.continuous_count == 1

    def test_integer_count(self):
        assert self.model.integer_count == 1

    def test_binary_count(self):
        assert self.model.binary_count == 1

    def test_is_mip(self):
        assert self.model.is_mip is True

    def test_var_types_individual(self):
        assert self.model.var_types[0] == VarType.CONTINUOUS
        assert self.model.var_types[1] == VarType.INTEGER
        assert self.model.var_types[2] == VarType.BINARY

    def test_validation_passes(self):
        assert ModelValidator.validate(self.model).is_valid


# ===========================================================================
# 10. Explicit variable bounds
# ===========================================================================


class TestVariableBounds:
    """
    min  x1 + x2 + x3
    x1 in [2, 8], x2 in [-inf, 5], x3 in [0, inf] (free lower bound)
    """

    def setup_method(self):
        A = sparse.csr_matrix(np.array([[1.0, 1.0, 1.0]], dtype=np.float64))
        self.model = CanonicalModel(
            name="bounds_test",
            sense=ObjectiveSense.MINIMIZE,
            c=np.array([1.0, 1.0, 1.0]),
            A=A,
            rhs=np.array([20.0]),
            constraint_senses=[ConstraintSense.LESS_EQUAL],
            l_var=np.array([2.0, -np.inf, 0.0]),
            u_var=np.array([8.0, 5.0, np.inf]),
            var_types=[VarType.CONTINUOUS, VarType.CONTINUOUS, VarType.CONTINUOUS],
            var_names=["x1", "x2", "x3"],
            constr_names=["total"],
        )

    def test_lower_bounds(self):
        assert self.model.l_var[0] == pytest.approx(2.0)
        assert np.isneginf(self.model.l_var[1])
        assert self.model.l_var[2] == pytest.approx(0.0)

    def test_upper_bounds(self):
        assert self.model.u_var[0] == pytest.approx(8.0)
        assert self.model.u_var[1] == pytest.approx(5.0)
        assert np.isposinf(self.model.u_var[2])

    def test_validation_passes(self):
        assert ModelValidator.validate(self.model).is_valid


# ===========================================================================
# 11. compute_row_bounds — all three senses
# ===========================================================================


class TestComputeRowBounds:
    """Test that compute_row_bounds correctly converts (rhs, senses) → (L, U)."""

    def setup_method(self):
        A = sparse.csr_matrix(np.eye(3, dtype=np.float64))
        self.model = CanonicalModel(
            name="row_bounds_test",
            sense=ObjectiveSense.MINIMIZE,
            c=np.array([1.0, 1.0, 1.0]),
            A=A,
            rhs=np.array([5.0, 7.0, 3.0]),
            constraint_senses=[
                ConstraintSense.LESS_EQUAL,
                ConstraintSense.EQUAL,
                ConstraintSense.GREATER_EQUAL,
            ],
            l_var=np.array([0.0, 0.0, 0.0]),
            u_var=np.array([np.inf, np.inf, np.inf]),
            var_types=[VarType.CONTINUOUS, VarType.CONTINUOUS, VarType.CONTINUOUS],
            var_names=["a", "b", "c"],
            constr_names=["le_row", "eq_row", "ge_row"],
        )

    def test_le_row_bounds(self):
        L, U = self.model.compute_row_bounds()
        assert np.isneginf(L[0])
        assert U[0] == pytest.approx(5.0)

    def test_eq_row_bounds(self):
        L, U = self.model.compute_row_bounds()
        assert L[1] == pytest.approx(7.0)
        assert U[1] == pytest.approx(7.0)

    def test_ge_row_bounds(self):
        L, U = self.model.compute_row_bounds()
        assert L[2] == pytest.approx(3.0)
        assert np.isposinf(U[2])


# ===========================================================================
# 12. Validation – invalid matrix dimensions
# ===========================================================================


class TestValidationInvalidMatrixDimensions:

    def test_A_wrong_row_count(self):
        """A has 3 rows but constr_names implies 2 rows."""
        A = sparse.csr_matrix(np.ones((3, 2), dtype=np.float64))
        model = CanonicalModel(
            name="bad_shape",
            c=np.array([1.0, 1.0]),
            A=A,
            rhs=np.array([1.0, 1.0]),
            constraint_senses=[ConstraintSense.LESS_EQUAL, ConstraintSense.LESS_EQUAL],
            l_var=np.array([0.0, 0.0]),
            u_var=np.array([np.inf, np.inf]),
            var_types=[VarType.CONTINUOUS, VarType.CONTINUOUS],
            var_names=["x1", "x2"],
            constr_names=["c1", "c2"],
        )
        report = ModelValidator.validate(model)
        assert report.is_valid is False
        assert any("shape" in e or "A" in e for e in report.errors)

    def test_A_wrong_col_count(self):
        """A has 3 columns but var_names implies 2 variables."""
        A = sparse.csr_matrix(np.ones((1, 3), dtype=np.float64))
        model = CanonicalModel(
            name="bad_cols",
            c=np.array([1.0, 1.0]),
            A=A,
            rhs=np.array([5.0]),
            constraint_senses=[ConstraintSense.LESS_EQUAL],
            l_var=np.array([0.0, 0.0]),
            u_var=np.array([np.inf, np.inf]),
            var_types=[VarType.CONTINUOUS, VarType.CONTINUOUS],
            var_names=["x1", "x2"],
            constr_names=["c1"],
        )
        report = ModelValidator.validate(model)
        assert report.is_valid is False


# ===========================================================================
# 13. Validation – invalid objective dimension
# ===========================================================================


class TestValidationInvalidObjectiveDimension:

    def test_c_too_short(self):
        """c has only 1 entry but there are 2 variables."""
        A = sparse.csr_matrix(np.array([[1.0, 1.0]], dtype=np.float64))
        model = CanonicalModel(
            name="short_c",
            c=np.array([1.0]),          # wrong length
            A=A,
            rhs=np.array([5.0]),
            constraint_senses=[ConstraintSense.LESS_EQUAL],
            l_var=np.array([0.0, 0.0]),
            u_var=np.array([np.inf, np.inf]),
            var_types=[VarType.CONTINUOUS, VarType.CONTINUOUS],
            var_names=["x1", "x2"],
            constr_names=["c1"],
        )
        report = ModelValidator.validate(model)
        assert report.is_valid is False
        assert any("Objective" in e or "c has length" in e for e in report.errors)


# ===========================================================================
# 14. Validation – inconsistent variable bounds (l > u)
# ===========================================================================


class TestValidationInconsistentBounds:

    def test_lower_exceeds_upper(self):
        """l_var[0] = 8 > u_var[0] = 2 → invalid."""
        A = sparse.csr_matrix(np.array([[1.0, 1.0]], dtype=np.float64))
        model = CanonicalModel(
            name="bad_bounds",
            c=np.array([1.0, 1.0]),
            A=A,
            rhs=np.array([10.0]),
            constraint_senses=[ConstraintSense.LESS_EQUAL],
            l_var=np.array([8.0, 0.0]),   # l_var[0] > u_var[0]
            u_var=np.array([2.0, 5.0]),
            var_types=[VarType.CONTINUOUS, VarType.CONTINUOUS],
            var_names=["x1", "x2"],
            constr_names=["c1"],
        )
        report = ModelValidator.validate(model)
        assert report.is_valid is False
        # Error message must identify the variable and the bad bound values
        assert any("x1" in e and "inconsistent" in e.lower() for e in report.errors)

    def test_raise_on_error_raises(self):
        A = sparse.csr_matrix(np.array([[1.0]], dtype=np.float64))
        model = CanonicalModel(
            name="raise_test",
            c=np.array([1.0]),
            A=A,
            rhs=np.array([10.0]),
            constraint_senses=[ConstraintSense.LESS_EQUAL],
            l_var=np.array([5.0]),
            u_var=np.array([1.0]),
            var_types=[VarType.CONTINUOUS],
            var_names=["x1"],
            constr_names=["c1"],
        )
        with pytest.raises(ModelValidationError) as exc_info:
            ModelValidator.validate(model, raise_on_error=True)
        assert len(exc_info.value.errors) >= 1


# ===========================================================================
# 15. Validation – invalid binary variable bounds
# ===========================================================================


class TestValidationBinaryBounds:

    def test_binary_upper_bound_exceeds_one(self):
        """Binary variable with ub=2 must be rejected."""
        A = sparse.csr_matrix(np.array([[1.0]], dtype=np.float64))
        model = CanonicalModel(
            name="bad_binary",
            c=np.array([1.0]),
            A=A,
            rhs=np.array([1.0]),
            constraint_senses=[ConstraintSense.LESS_EQUAL],
            l_var=np.array([0.0]),
            u_var=np.array([2.0]),          # invalid for binary
            var_types=[VarType.BINARY],
            var_names=["y1"],
            constr_names=["c1"],
        )
        report = ModelValidator.validate(model)
        assert report.is_valid is False
        assert any("y1" in e and "bounds" in e.lower() for e in report.errors)

    def test_binary_lower_bound_negative(self):
        """Binary variable with lb=-1 must be rejected."""
        A = sparse.csr_matrix(np.array([[1.0]], dtype=np.float64))
        model = CanonicalModel(
            name="neg_binary_lb",
            c=np.array([1.0]),
            A=A,
            rhs=np.array([1.0]),
            constraint_senses=[ConstraintSense.LESS_EQUAL],
            l_var=np.array([-1.0]),         # invalid for binary
            u_var=np.array([1.0]),
            var_types=[VarType.BINARY],
            var_names=["y1"],
            constr_names=["c1"],
        )
        report = ModelValidator.validate(model)
        assert report.is_valid is False


# ===========================================================================
# 16. Validation – RHS dimension mismatch
# ===========================================================================


class TestValidationRHSDimensionMismatch:

    def test_rhs_too_long(self):
        """rhs has 3 entries but there are only 2 constraints."""
        A = sparse.csr_matrix(np.ones((2, 2), dtype=np.float64))
        model = CanonicalModel(
            name="rhs_mismatch",
            c=np.array([1.0, 1.0]),
            A=A,
            rhs=np.array([1.0, 2.0, 3.0]),   # wrong length
            constraint_senses=[ConstraintSense.LESS_EQUAL, ConstraintSense.LESS_EQUAL],
            l_var=np.array([0.0, 0.0]),
            u_var=np.array([np.inf, np.inf]),
            var_types=[VarType.CONTINUOUS, VarType.CONTINUOUS],
            var_names=["x1", "x2"],
            constr_names=["c1", "c2"],
        )
        report = ModelValidator.validate(model)
        assert report.is_valid is False
        assert any("rhs" in e.lower() for e in report.errors)


# ===========================================================================
# 17. Validation – NaN in matrix
# ===========================================================================


class TestValidationNaNValues:

    def test_nan_in_A(self):
        data = np.array([[1.0, np.nan]], dtype=np.float64)
        A = sparse.csr_matrix(data)
        model = CanonicalModel(
            name="nan_A",
            c=np.array([1.0, 1.0]),
            A=A,
            rhs=np.array([5.0]),
            constraint_senses=[ConstraintSense.LESS_EQUAL],
            l_var=np.array([0.0, 0.0]),
            u_var=np.array([np.inf, np.inf]),
            var_types=[VarType.CONTINUOUS, VarType.CONTINUOUS],
            var_names=["x1", "x2"],
            constr_names=["c1"],
        )
        report = ModelValidator.validate(model)
        assert report.is_valid is False
        assert any("NaN" in e for e in report.errors)

    def test_nan_in_c(self):
        A = sparse.csr_matrix(np.array([[1.0, 1.0]], dtype=np.float64))
        model = CanonicalModel(
            name="nan_c",
            c=np.array([1.0, np.nan]),   # NaN in objective
            A=A,
            rhs=np.array([5.0]),
            constraint_senses=[ConstraintSense.LESS_EQUAL],
            l_var=np.array([0.0, 0.0]),
            u_var=np.array([np.inf, np.inf]),
            var_types=[VarType.CONTINUOUS, VarType.CONTINUOUS],
            var_names=["x1", "x2"],
            constr_names=["c1"],
        )
        report = ModelValidator.validate(model)
        assert report.is_valid is False
        assert any("NaN" in e and ("c" in e or "Objective" in e) for e in report.errors)


# ===========================================================================
# 18. Validation – duplicate names
# ===========================================================================


class TestValidationDuplicateNames:

    def test_duplicate_variable_names(self):
        A = sparse.csr_matrix(np.ones((1, 2), dtype=np.float64))
        model = CanonicalModel(
            name="dup_vars",
            c=np.array([1.0, 1.0]),
            A=A,
            rhs=np.array([5.0]),
            constraint_senses=[ConstraintSense.LESS_EQUAL],
            l_var=np.array([0.0, 0.0]),
            u_var=np.array([np.inf, np.inf]),
            var_types=[VarType.CONTINUOUS, VarType.CONTINUOUS],
            var_names=["x1", "x1"],   # duplicate
            constr_names=["c1"],
        )
        report = ModelValidator.validate(model)
        assert report.is_valid is False
        assert any("Duplicate" in e and "x1" in e for e in report.errors)

    def test_duplicate_constraint_names(self):
        A = sparse.csr_matrix(np.ones((2, 2), dtype=np.float64))
        model = CanonicalModel(
            name="dup_constrs",
            c=np.array([1.0, 1.0]),
            A=A,
            rhs=np.array([5.0, 5.0]),
            constraint_senses=[ConstraintSense.LESS_EQUAL, ConstraintSense.LESS_EQUAL],
            l_var=np.array([0.0, 0.0]),
            u_var=np.array([np.inf, np.inf]),
            var_types=[VarType.CONTINUOUS, VarType.CONTINUOUS],
            var_names=["x1", "x2"],
            constr_names=["c1", "c1"],   # duplicate
        )
        report = ModelValidator.validate(model)
        assert report.is_valid is False
        assert any("Duplicate" in e and "c1" in e for e in report.errors)


# ===========================================================================
# 19. Validation – zero-variable model rejected
# ===========================================================================


class TestValidationZeroVariables:

    def test_empty_model_rejected(self):
        model = CanonicalModel(name="empty")
        report = ModelValidator.validate(model)
        assert report.is_valid is False
        assert any("variable" in e.lower() for e in report.errors)


# ===========================================================================
# 20. ValidationReport structure
# ===========================================================================


class TestValidationReportStructure:

    def test_valid_report_has_empty_errors(self):
        model = _make_model()
        report = ModelValidator.validate(model)
        assert report.is_valid is True
        assert isinstance(report.errors, list)
        assert len(report.errors) == 0

    def test_valid_report_has_summary(self):
        model = _make_model()
        report = ModelValidator.validate(model)
        assert "num_variables" in report.summary
        assert "num_constraints" in report.summary
        assert "sense" in report.summary

    def test_invalid_report_has_no_summary(self):
        model = CanonicalModel(name="empty")
        report = ModelValidator.validate(model)
        assert report.is_valid is False
        assert report.summary == {}

    def test_to_dict_structure(self):
        model = _make_model()
        d = ModelValidator.validate(model).to_dict()
        assert d["is_valid"] is True
        assert d["error_count"] == 0
        assert "errors" in d
        assert "warnings" in d
        assert "summary" in d


# ===========================================================================
# 21. Sparse matrix dtype normalisation
# ===========================================================================


class TestSparseMatrixNormalisation:

    def test_int_array_converted_to_float64(self):
        A_int = sparse.csr_matrix(np.eye(2, dtype=np.int32))
        model = CanonicalModel(
            name="int_matrix",
            c=np.array([1.0, 2.0]),
            A=A_int,
            rhs=np.array([5.0, 5.0]),
            constraint_senses=[ConstraintSense.LESS_EQUAL, ConstraintSense.LESS_EQUAL],
            l_var=np.array([0.0, 0.0]),
            u_var=np.array([np.inf, np.inf]),
            var_types=[VarType.CONTINUOUS, VarType.CONTINUOUS],
            var_names=["x1", "x2"],
            constr_names=["c1", "c2"],
        )
        assert model.A.dtype == np.float64

    def test_non_csr_converted_to_csr(self):
        A_coo = sparse.coo_matrix(np.eye(2, dtype=np.float64))
        model = CanonicalModel(
            name="coo_matrix",
            c=np.array([1.0, 2.0]),
            A=A_coo,
            rhs=np.array([5.0, 5.0]),
            constraint_senses=[ConstraintSense.LESS_EQUAL, ConstraintSense.LESS_EQUAL],
            l_var=np.array([0.0, 0.0]),
            u_var=np.array([np.inf, np.inf]),
            var_types=[VarType.CONTINUOUS, VarType.CONTINUOUS],
            var_names=["x1", "x2"],
            constr_names=["c1", "c2"],
        )
        assert sparse.isspmatrix_csr(model.A)


# ===========================================================================
# 22. _to_float64_array helper
# ===========================================================================


class TestToFloat64ArrayHelper:

    def test_list_becomes_array(self):
        result = _to_float64_array([1, 2, 3])
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    def test_int_array_cast(self):
        result = _to_float64_array(np.array([1, 2, 3], dtype=np.int32))
        assert result.dtype == np.float64

    def test_float64_passthrough(self):
        arr = np.array([1.0, 2.0], dtype=np.float64)
        result = _to_float64_array(arr)
        assert result.dtype == np.float64


# ===========================================================================
# 23. Warning: empty constraint rows
# ===========================================================================


class TestWarningEmptyRows:

    def test_empty_row_generates_warning(self):
        """A row of all zeros should trigger a warning (not an error)."""
        # Row 1 is all zeros
        A = sparse.csr_matrix(
            np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float64)
        )
        model = CanonicalModel(
            name="empty_row",
            c=np.array([1.0, 1.0]),
            A=A,
            rhs=np.array([5.0, 0.0]),
            constraint_senses=[ConstraintSense.LESS_EQUAL, ConstraintSense.EQUAL],
            l_var=np.array([0.0, 0.0]),
            u_var=np.array([np.inf, np.inf]),
            var_types=[VarType.CONTINUOUS, VarType.CONTINUOUS],
            var_names=["x1", "x2"],
            constr_names=["c1", "trivial"],
        )
        report = ModelValidator.validate(model)
        # Should still be valid (warning, not error)
        assert report.is_valid is True
        assert any("empty" in w.lower() for w in report.warnings)
