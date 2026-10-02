"""Phase 6 solver abstraction, HiGHS backend, and pipeline tests."""

import numpy as np
import pytest
from scipy import sparse

from saksham.canonical.model import (
    CanonicalModel,
    ConstraintSense,
    ObjectiveSense,
    VarType,
)
from saksham.canonical.validator import ModelValidationError
from saksham.fingerprint.analyzer import StructuralFingerprinter
from saksham.health.conditioner import NumericalHealthMonitor
from saksham.isg.graph_builder import ISGBuilder
from saksham.parser.lp_parser import LPParser
from saksham.strategy.recommender import StrategyRecommender
from saksham.solvers.base import SolveResult
from saksham.solvers.highs_backend import HighsSolverBackend
from saksham.solvers.registry import SolverRegistry, SolverUnavailableError


def make_model(
    matrix,
    objective,
    rhs=None,
    constraint_senses=None,
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
    if constraint_senses is None:
        constraint_senses = [ConstraintSense.LESS_EQUAL] * num_constraints
    if lower_bounds is None:
        lower_bounds = np.zeros(num_variables, dtype=np.float64)
    if upper_bounds is None:
        upper_bounds = np.full(num_variables, np.inf, dtype=np.float64)
    if var_types is None:
        var_types = [VarType.CONTINUOUS] * num_variables
    return CanonicalModel(
        name="solver_test_model",
        sense=objective_sense,
        c=np.asarray(objective, dtype=np.float64),
        c0=objective_constant,
        A=matrix,
        rhs=np.asarray(rhs, dtype=np.float64),
        constraint_senses=constraint_senses,
        l_var=np.asarray(lower_bounds, dtype=np.float64),
        u_var=np.asarray(upper_bounds, dtype=np.float64),
        var_types=var_types,
        var_names=[f"x{index}" for index in range(num_variables)],
        constr_names=[f"c{index}" for index in range(num_constraints)],
    )


def solve_model(model, strategy=None):
    backend = HighsSolverBackend()
    backend.load_model(model)
    if strategy is not None:
        backend.apply_strategy(strategy)
    return backend.solve()


def test_solve_continuous_minimization():
    model = make_model(
        [[1.0, 1.0]],
        [1.0, 2.0],
        rhs=[4.0],
        constraint_senses=[ConstraintSense.GREATER_EQUAL],
        upper_bounds=[4.0, np.inf],
    )

    result = solve_model(model)

    assert result.status == "OPTIMAL"
    assert result.success is True
    assert result.objective_value == pytest.approx(4.0)
    assert result.variable_values == pytest.approx([4.0, 0.0])


def test_solve_continuous_maximization():
    model = make_model(
        [[1.0, 1.0]],
        [3.0, 2.0],
        rhs=[4.0],
        constraint_senses=[ConstraintSense.LESS_EQUAL],
        upper_bounds=[2.0, 4.0],
        objective_sense=ObjectiveSense.MAXIMIZE,
    )

    result = solve_model(model)

    assert result.status == "OPTIMAL"
    assert result.objective_value == pytest.approx(10.0)
    assert result.variable_values == pytest.approx([2.0, 2.0])


@pytest.mark.parametrize(
    ("sense", "rhs", "objective_sense", "expected"),
    [
        (ConstraintSense.LESS_EQUAL, 3.0, ObjectiveSense.MAXIMIZE, 3.0),
        (ConstraintSense.GREATER_EQUAL, 2.0, ObjectiveSense.MINIMIZE, 2.0),
        (ConstraintSense.EQUAL, 5.0, ObjectiveSense.MINIMIZE, 5.0),
    ],
)
def test_constraint_senses_are_converted_correctly(sense, rhs, objective_sense, expected):
    objective = [1.0] if sense != ConstraintSense.EQUAL else [1.0, 2.0]
    matrix = [[1.0]] if len(objective) == 1 else [[1.0, 1.0]]
    model = make_model(
        matrix,
        objective,
        rhs=[rhs],
        constraint_senses=[sense],
        objective_sense=objective_sense,
    )

    result = solve_model(model)

    assert result.status == "OPTIMAL"
    assert result.objective_value == pytest.approx(expected)


def test_variable_lower_and_upper_bounds_are_respected():
    lower_bounded = make_model(
        np.zeros((0, 1)),
        [1.0],
        lower_bounds=[3.0],
        upper_bounds=[8.0],
    )
    upper_bounded = make_model(
        np.zeros((0, 1)),
        [1.0],
        lower_bounds=[0.0],
        upper_bounds=[4.0],
        objective_sense=ObjectiveSense.MAXIMIZE,
    )

    minimum = solve_model(lower_bounded)
    maximum = solve_model(upper_bounded)

    assert minimum.variable_values == pytest.approx([3.0])
    assert maximum.variable_values == pytest.approx([4.0])


def test_objective_constant_is_included_in_reported_value():
    model = make_model(
        [[1.0]],
        [3.0],
        rhs=[2.0],
        constraint_senses=[ConstraintSense.GREATER_EQUAL],
        objective_constant=5.0,
    )

    result = solve_model(model)

    assert result.objective_value == pytest.approx(11.0)


def test_sparse_matrix_conversion_does_not_densify_or_mutate_model(monkeypatch):
    model = make_model(
        sparse.csr_matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        [1.0, 1.0, 0.0],
        rhs=[2.0, 3.0],
    )
    original_matrix = model.A.copy()
    original_objective = model.c.copy()

    def fail_toarray(*args, **kwargs):
        raise AssertionError("Sparse matrix was densified")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", fail_toarray)
    result = solve_model(model)

    assert result.status == "OPTIMAL"
    assert sparse.isspmatrix_csr(model.A)
    assert (model.A != original_matrix).nnz == 0
    assert np.array_equal(model.c, original_objective)


def test_infeasible_model_reports_infeasible():
    model = make_model(
        [[1.0], [1.0]],
        [1.0],
        rhs=[2.0, 1.0],
        constraint_senses=[
            ConstraintSense.GREATER_EQUAL,
            ConstraintSense.LESS_EQUAL,
        ],
    )

    result = solve_model(model)

    assert result.status == "INFEASIBLE"
    assert result.success is False
    assert result.objective_value is None


def test_unbounded_model_reports_unbounded():
    model = make_model(
        np.zeros((0, 1)),
        [-1.0],
        objective_sense=ObjectiveSense.MINIMIZE,
    )

    result = solve_model(model)

    assert result.status == "UNBOUNDED"
    assert result.success is False


def test_invalid_model_is_rejected_before_solver_load():
    model = make_model([[1.0]], [1.0], rhs=[np.nan])
    backend = HighsSolverBackend()

    with pytest.raises(ModelValidationError):
        backend.load_model(model)


def test_solver_availability_and_registry_selection():
    registry = SolverRegistry()

    assert HighsSolverBackend().is_available() is True
    assert registry.available() == {"highs": True}
    assert registry.get("highs").solver_name == "highs"


def test_solver_registry_reports_unknown_or_unavailable_backend():
    registry = SolverRegistry(register_defaults=False)

    with pytest.raises(SolverUnavailableError):
        registry.get("highs")
    assert registry.available() == {}


def test_solver_result_exposes_statistics_and_messages():
    result = solve_model(make_model([[1.0]], [2.0], rhs=[3.0]))

    assert isinstance(result, SolveResult)
    assert result.solver_name == "highs"
    assert result.message
    assert result.termination_reason
    assert result.wall_time_seconds is not None
    assert result.iteration_count is None or result.iteration_count >= 0
    assert isinstance(result.details, dict)


def test_integer_variable_is_solved_as_mip():
    model = make_model(
        [[2.0]],
        [3.0],
        rhs=[3.0],
        constraint_senses=[ConstraintSense.LESS_EQUAL],
        upper_bounds=[1.0],
        var_types=[VarType.BINARY],
        objective_sense=ObjectiveSense.MAXIMIZE,
    )

    result = solve_model(model)

    assert result.status == "OPTIMAL"
    assert result.variable_values == pytest.approx([1.0])
    assert result.objective_value == pytest.approx(3.0)


def test_backend_rejects_unsupported_input_type():
    with pytest.raises(TypeError):
        HighsSolverBackend().load_model(object())


def test_invalid_variable_type_is_not_silently_coerced():
    model = make_model([[1.0]], [1.0])
    model.var_types[0] = "semi_continuous"

    with pytest.raises(ModelValidationError):
        HighsSolverBackend().load_model(model)


def test_backend_result_status_is_deterministic():
    model = make_model([[1.0]], [1.0], rhs=[2.0])

    first = solve_model(model)
    second = solve_model(model)

    assert first.status == second.status == "OPTIMAL"
    assert first.objective_value == pytest.approx(second.objective_value)


def test_registry_connects_strategy_recommendation_to_backend():
    model = make_model([[1.0]], [1.0], rhs=[2.0])
    recommendation = StrategyRecommender().recommend(model)
    registry = SolverRegistry()

    result = registry.solve(model, recommendation)

    assert result.status == "OPTIMAL"
    assert result.solver_name == "highs"


def test_registry_returns_explicit_error_for_invalid_model():
    model = make_model([[1.0]], [1.0], rhs=[np.nan])

    result = SolverRegistry().solve(model)

    assert result.status == "ERROR"
    assert result.termination_reason == "INVALID_MODEL"
    assert result.objective_value is None
    assert result.error_message


def test_solver_emits_final_telemetry_without_fabricated_statistics():
    events = []
    model = make_model([[1.0]], [1.0], rhs=[2.0])
    backend = HighsSolverBackend()
    backend.load_model(model)

    result = backend.solve(telemetry_callback=events.append)

    assert len(events) == 1
    assert events[0]["event"] == "solve_finished"
    assert events[0]["status"] == result.status
    assert result.iteration_count is None or result.iteration_count >= 0


def test_lp_to_solve_pipeline_end_to_end():
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
    model = LPParser().parse_string(lp_text, model_name="pipeline_test")
    fingerprint = StructuralFingerprinter().analyze(model)
    health = NumericalHealthMonitor().evaluate(model)
    isg = ISGBuilder().build_bipartite_graph(model)
    recommendation = StrategyRecommender().recommend(
        fingerprint,
        model=model,
        health=health,
        isg=isg,
    )
    result = SolverRegistry().solve(model, recommendation)

    assert result.status == "OPTIMAL"
    assert result.objective_value == pytest.approx(4.0)
    assert result.variable_values.sum() >= 4.0 - 1e-8
    assert result.solver_name == "highs"