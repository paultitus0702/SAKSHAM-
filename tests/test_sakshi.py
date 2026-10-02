"""Phase 7 independent solution verification and certificate tests."""

import numpy as np
import pytest
from scipy import sparse

from saksham.canonical.model import (
    CanonicalModel,
    ConstraintSense,
    ObjectiveSense,
    VarType,
)
from saksham.certificate.cert_builder import SolveCertificateGenerator
from saksham.parser.lp_parser import LPParser
from saksham.sakshi.verifier import SakshiVerifier
from saksham.solvers.base import SolveResult
from saksham.solvers.registry import SolverRegistry


def make_model(
    matrix,
    objective,
    rhs=None,
    senses=None,
    lower_bounds=None,
    upper_bounds=None,
    var_types=None,
    objective_sense=ObjectiveSense.MINIMIZE,
    objective_constant=0.0,
):
    matrix = sparse.csr_matrix(matrix, dtype=np.float64)
    num_constraints, num_variables = matrix.shape
    if rhs is None:
        rhs = np.ones(num_constraints, dtype=np.float64)
    if senses is None:
        senses = [ConstraintSense.LESS_EQUAL] * num_constraints
    if lower_bounds is None:
        lower_bounds = np.zeros(num_variables, dtype=np.float64)
    if upper_bounds is None:
        upper_bounds = np.full(num_variables, np.inf, dtype=np.float64)
    if var_types is None:
        var_types = [VarType.CONTINUOUS] * num_variables
    return CanonicalModel(
        name="sakshi_test_model",
        sense=objective_sense,
        c=np.asarray(objective, dtype=np.float64),
        c0=objective_constant,
        A=matrix,
        rhs=np.asarray(rhs, dtype=np.float64),
        constraint_senses=senses,
        l_var=np.asarray(lower_bounds, dtype=np.float64),
        u_var=np.asarray(upper_bounds, dtype=np.float64),
        var_types=var_types,
        var_names=[f"x{index}" for index in range(num_variables)],
        constr_names=[f"c{index}" for index in range(num_constraints)],
    )


def make_result(values, model, objective_value=None, status="OPTIMAL", success=True):
    values = np.asarray(values, dtype=np.float64)
    if objective_value is None and values.shape == model.c.shape:
        objective_value = float(model.c0 + model.c @ values)
    return SolveResult(
        status=status,
        objective_value=objective_value,
        primal_solution=values,
        solver_name="test_solver",
        success=success,
        message="test result",
    )


def test_valid_continuous_solution_is_verified():
    model = make_model(
        [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        [2.0, 3.0],
        rhs=[4.0, 3.0, 3.0],
        senses=[
            ConstraintSense.GREATER_EQUAL,
            ConstraintSense.LESS_EQUAL,
            ConstraintSense.LESS_EQUAL,
        ],
        objective_constant=5.0,
    )
    result = make_result([3.0, 1.0], model)

    verification = SakshiVerifier().verify(model, result)

    assert verification.status == "VERIFIED"
    assert verification.verdict == "VERIFIED"
    assert verification.is_valid is True
    assert verification.solver_status == "OPTIMAL"
    assert verification.bounds_satisfied is True
    assert verification.constraints_satisfied is True
    assert verification.integrality_satisfied is True
    assert verification.objective_consistent is True
    assert verification.independently_calculated_objective == pytest.approx(14.0)
    assert verification.max_constraint_violation == pytest.approx(0.0)
    assert verification.max_bound_violation == pytest.approx(0.0)
    assert verification.max_integrality_violation == pytest.approx(0.0)


def test_constraint_violation_fails_verification():
    model = make_model([[1.0]], [1.0], rhs=[1.0])
    result = make_result([2.0], model)

    verification = SakshiVerifier().verify(model, result)

    assert verification.status == "FAILED"
    assert verification.constraints_satisfied is False
    assert verification.max_constraint_violation == pytest.approx(1.0)


def test_bound_violation_fails_verification():
    model = make_model(
        np.zeros((0, 1)), [1.0], lower_bounds=[0.0], upper_bounds=[2.0]
    )
    result = make_result([2.5], model)

    verification = SakshiVerifier().verify(model, result)

    assert verification.status == "FAILED"
    assert verification.bounds_satisfied is False
    assert verification.max_bound_violation == pytest.approx(0.5)


def test_integer_variable_fractional_value_fails_integrality():
    model = make_model(np.zeros((0, 1)), [1.0], var_types=[VarType.INTEGER])

    verification = SakshiVerifier().verify(model, make_result([1.25], model))

    assert verification.status == "FAILED"
    assert verification.integrality_satisfied is False
    assert verification.max_integrality_violation == pytest.approx(0.25)


def test_binary_variable_outside_zero_one_fails_integrality():
    model = make_model(
        np.zeros((0, 1)),
        [1.0],
        lower_bounds=[0.0],
        upper_bounds=[1.0],
        var_types=[VarType.BINARY],
    )

    verification = SakshiVerifier().verify(model, make_result([0.4], model))

    assert verification.status == "FAILED"
    assert verification.integrality_satisfied is False
    assert verification.max_integrality_violation == pytest.approx(0.4)


def test_objective_discrepancy_is_reported_as_failure():
    model = make_model([[1.0]], [2.0], rhs=[1.0])
    result = make_result([1.0], model, objective_value=3.0)

    verification = SakshiVerifier().verify(model, result)

    assert verification.status == "FAILED"
    assert verification.objective_consistent is False
    assert verification.independently_calculated_objective == pytest.approx(2.0)
    assert verification.objective_discrepancy == pytest.approx(1.0)


def test_tolerance_accepts_small_constraint_error_but_rejects_larger_error():
    model = make_model([[1.0]], [1.0], rhs=[1.0])
    verifier = SakshiVerifier(feasibility_tolerance=1e-6)

    within = verifier.verify(model, make_result([1.0 + 5e-7], model))
    outside = verifier.verify(model, make_result([1.0 + 2e-6], model))

    assert within.status == "VERIFIED"
    assert within.tolerance == pytest.approx(1e-6)
    assert outside.status == "FAILED"


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_primal_solution_is_invalid(bad_value):
    model = make_model([[1.0]], [1.0], rhs=[1.0])

    verification = SakshiVerifier().verify(model, make_result([bad_value], model))

    assert verification.status == "INVALID"
    assert verification.is_valid is False
    assert verification.unavailable_checks


def test_malformed_result_dimensions_are_invalid():
    model = make_model([[1.0, 1.0]], [1.0, 2.0], rhs=[1.0])
    result = SolveResult(status="OPTIMAL", success=True, primal_solution=np.array([1.0]))

    verification = SakshiVerifier().verify(model, result)

    assert verification.status == "INVALID"
    assert any("dimension" in warning.lower() for warning in verification.warnings)


def test_non_success_solver_status_has_unavailable_verification():
    model = make_model([[1.0]], [1.0], rhs=[1.0])
    result = SolveResult(status="INFEASIBLE", success=False, solver_name="highs")

    verification = SakshiVerifier().verify(model, result)

    assert verification.status == "UNAVAILABLE"
    assert verification.solver_status == "INFEASIBLE"
    assert verification.bounds_satisfied is None
    assert verification.unavailable_checks


def test_invalid_model_and_missing_result_are_explicit():
    verifier = SakshiVerifier()
    invalid_model = CanonicalModel()
    valid_model = make_model([[1.0]], [1.0], rhs=[1.0])

    invalid = verifier.verify(invalid_model, SolveResult())
    missing = verifier.verify(valid_model, None)

    assert invalid.status == "INVALID"
    assert missing.status == "UNAVAILABLE"
    assert missing.solver_status is None


def test_maximization_objective_is_calculated_without_claiming_optimality():
    model = make_model(
        np.zeros((0, 1)),
        [4.0],
        lower_bounds=[0.0],
        upper_bounds=[3.0],
        objective_sense=ObjectiveSense.MAXIMIZE,
        objective_constant=-2.0,
    )
    result = make_result([2.0], model)

    verification = SakshiVerifier().verify(model, result)

    assert verification.status == "VERIFIED"
    assert verification.independently_calculated_objective == pytest.approx(6.0)
    assert verification.details["optimality_verified"] is False


def test_certificate_separates_reported_and_verified_information():
    model = make_model([[1.0]], [2.0], rhs=[1.0], objective_constant=3.0)
    result = make_result([1.0], model)

    certificate = SolveCertificateGenerator().build_certificate(model, result)

    assert certificate.model_name == model.name
    assert certificate.solver_name == "test_solver"
    assert certificate.solver_status == "OPTIMAL"
    assert certificate.objective_sense == "minimize"
    assert certificate.solver_reported_objective_value == pytest.approx(5.0)
    assert certificate.independently_calculated_objective_value == pytest.approx(5.0)
    assert certificate.verification_status == "VERIFIED"
    assert certificate.feasibility_status == "SATISFIED"
    assert certificate.integrality_status == "SATISFIED"
    assert certificate.optimality_claim == "NOT_VERIFIED"
    assert certificate.sha256_hash
    assert any("optimality" in item for item in certificate.details["not_claimed"])


def test_certificate_digest_is_deterministic_and_failure_is_preserved():
    model = make_model([[1.0]], [1.0], rhs=[1.0])
    result = make_result([2.0], model)
    generator = SolveCertificateGenerator()

    first = generator.build_certificate(model, result)
    second = generator.build_certificate(model, result)

    assert first.sha256_hash == second.sha256_hash
    assert first.verification_status == "FAILED"
    assert first.feasibility_status == "VIOLATED"


def test_sparse_verification_does_not_densify_matrix(monkeypatch):
    model = make_model(sparse.eye(20, format="csr"), np.ones(20), rhs=np.ones(20))
    result = make_result(np.ones(20), model)

    def fail_toarray(*args, **kwargs):
        raise AssertionError("Sparse matrix was densified")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", fail_toarray)

    verification = SakshiVerifier().verify(model, result)

    assert verification.status == "VERIFIED"


def test_lp_solver_verifier_certificate_pipeline():
    lp_text = """
    Minimize
      obj: x + 2 y
    Subject To
      demand: x + y >= 4
    Bounds
      0 <= x <= 4
      0 <= y
    End
    """
    model = LPParser().parse_string(lp_text, model_name="phase7_pipeline")
    result = SolverRegistry().solve(model)
    verification = SakshiVerifier().verify(model, result)
    certificate = SolveCertificateGenerator().build_certificate(
        model, result, verification=verification
    )

    assert result.status == "OPTIMAL"
    assert verification.status == "VERIFIED"
    assert certificate.verification_status == "VERIFIED"
    assert certificate.independently_calculated_objective_value == pytest.approx(4.0)
    assert certificate.optimality_claim == "NOT_VERIFIED"