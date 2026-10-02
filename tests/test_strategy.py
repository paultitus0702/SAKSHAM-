"""Phase 5 deterministic, explainable strategy recommendation tests."""

import numpy as np
import pytest
from scipy import sparse

from saksham.canonical.model import (
    CanonicalModel,
    ConstraintSense,
    ObjectiveSense,
    VarType,
)
from saksham.fingerprint.analyzer import StructuralFingerprinter
from saksham.health.conditioner import NumericalHealthMonitor
from saksham.isg.graph_builder import ISGBuilder
from saksham.strategy.recommender import StrategyRecommender


def make_model(
    matrix,
    objective=None,
    constraint_senses=None,
    var_types=None,
    name="strategy_test",
):
    matrix = sparse.csr_matrix(matrix, dtype=np.float64)
    num_constraints, num_variables = matrix.shape
    if objective is None:
        objective = np.ones(num_variables, dtype=np.float64)
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
        sense=ObjectiveSense.MINIMIZE,
        c=np.asarray(objective, dtype=np.float64),
        A=matrix,
        rhs=np.ones(num_constraints, dtype=np.float64),
        constraint_senses=constraint_senses,
        l_var=lower_bounds,
        u_var=upper_bounds,
        var_types=var_types,
        var_names=[f"x{index}" for index in range(num_variables)],
        constr_names=[f"c{index}" for index in range(num_constraints)],
    )


def analyze_model(model):
    return (
        StructuralFingerprinter().analyze(model),
        NumericalHealthMonitor().evaluate(model),
        ISGBuilder().build_bipartite_graph(model),
    )


def test_small_continuous_lp_gets_direct_lp_recommendation():
    model = make_model([[1.0, 1.0], [1.0, -1.0]], objective=[2.0, 3.0])

    recommendation = StrategyRecommender().recommend(model)

    assert recommendation.primary_strategy == "DIRECT_LP"
    assert recommendation.primary_algorithm == "dual_simplex"
    assert recommendation.status == "READY"
    assert recommendation.confidence == "MEDIUM"
    assert recommendation.model_summary["num_variables"] == 2
    assert recommendation.model_summary["num_constraints"] == 2


def test_sparse_continuous_lp_gets_general_sparse_recommendation():
    model = make_model(
        [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0, 0.0]]
    )

    recommendation = StrategyRecommender().recommend(model)

    assert recommendation.primary_strategy == "GENERAL_SPARSE"
    assert recommendation.model_summary["density"] == pytest.approx(0.2)
    assert any("density" in reason.lower() for reason in recommendation.rationale)


@pytest.mark.parametrize(
    ("types", "expected_counts"),
    [
        ([VarType.BINARY, VarType.BINARY], {"continuous": 0, "integer": 0, "binary": 2}),
        ([VarType.INTEGER, VarType.INTEGER], {"continuous": 0, "integer": 2, "binary": 0}),
        ([VarType.CONTINUOUS, VarType.INTEGER, VarType.BINARY], {"continuous": 1, "integer": 1, "binary": 1}),
    ],
)
def test_integer_and_mixed_models_get_mip_recommendation(types, expected_counts):
    model = make_model([[1.0] * len(types)], var_types=types)

    recommendation = StrategyRecommender().recommend(model)

    assert recommendation.primary_strategy == "GENERAL_MIP"
    assert recommendation.confidence == "HIGH"
    assert recommendation.model_summary["variable_type_counts"] == expected_counts
    assert any("integer" in reason.lower() or "binary" in reason.lower()
               for reason in recommendation.rationale)


def test_network_conservation_evidence_is_a_candidate_only():
    model = make_model(
        [[1.0, 2.0], [-1.0, -2.0]],
        constraint_senses=[ConstraintSense.EQUAL, ConstraintSense.EQUAL],
    )

    recommendation = StrategyRecommender().recommend(model)

    assert recommendation.primary_strategy == "NETWORK_OR_FLOW_CANDIDATE"
    assert recommendation.secondary_strategies == ["DIRECT_LP"]
    assert any("candidate" in warning.lower() for warning in recommendation.warnings)
    assert recommendation.model_summary["block_structure_status"] == "unknown"
    assert "DECOMPOSITION_CANDIDATE" not in recommendation.secondary_strategies


def test_binary_knapsack_signal_is_exposed_as_a_candidate():
    model = make_model(
        [[2.0, 3.0]],
        var_types=[VarType.BINARY, VarType.BINARY],
    )

    recommendation = StrategyRecommender().recommend(model)

    assert recommendation.primary_strategy == "GENERAL_MIP"
    assert "BINARY_KNAPSACK_CANDIDATE" in recommendation.secondary_strategies


def test_unhealthy_model_gets_numerical_preconditioning_recommendation():
    model = make_model([[1e-8, 0.0, 0.0, 0.0, 0.0], [0.0, 1e4, 0.0, 0.0, 0.0]])

    recommendation = StrategyRecommender().recommend(model)

    assert recommendation.status == "WARNING"
    assert recommendation.primary_strategy == "GENERAL_SPARSE"
    assert "NUMERICAL_PRECONDITIONING" in recommendation.secondary_strategies
    assert any("dynamic range" in warning.lower() for warning in recommendation.warnings)
    assert recommendation.model_summary["health_status"] == "WARNING"


def test_numerical_warning_status_is_preserved_without_optional_isg():
    model = make_model([[1e-8, 0.0], [0.0, 1e4]])
    fingerprint = StructuralFingerprinter().analyze(model)
    health = NumericalHealthMonitor().evaluate(model)

    recommendation = StrategyRecommender().recommend(
        fingerprint,
        health=health,
    )

    assert recommendation.status == "WARNING"
    assert "NUMERICAL_PRECONDITIONING" in recommendation.secondary_strategies
    assert any("ISG was not supplied" in warning for warning in recommendation.warnings)


def test_invalid_non_finite_model_is_not_recommended_for_solving():
    model = make_model([[1.0, np.inf]])

    recommendation = StrategyRecommender().recommend(model)

    assert recommendation.status == "INVALID"
    assert recommendation.primary_strategy == "INVALID_MODEL"
    assert recommendation.primary_algorithm == "not_recommended"
    assert recommendation.warnings
    assert recommendation.secondary_strategies == []


def test_recommender_rejects_missing_or_unsupported_input():
    recommender = StrategyRecommender()

    with pytest.raises(ValueError):
        recommender.recommend()
    with pytest.raises(TypeError):
        recommender.recommend(object())


def test_recommender_supports_legacy_fingerprint_only_call():
    model = make_model(
        [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]]
    )
    fingerprint = StructuralFingerprinter().analyze(model)

    recommendation = StrategyRecommender().recommend(fingerprint)

    assert recommendation.primary_strategy == "GENERAL_SPARSE"
    assert recommendation.model_summary["num_variables"] == 5
    assert recommendation.model_summary["health_status"] == "UNAVAILABLE"
    assert recommendation.status == "LIMITED"


def test_precomputed_phase_outputs_are_consumed_without_reanalysis():
    model = make_model([[1.0, 0.0], [0.0, 1.0]])
    fingerprint, health, isg = analyze_model(model)

    recommendation = StrategyRecommender().recommend(
        fingerprint,
        model=model,
        health=health,
        isg=isg,
    )

    assert recommendation.model_summary["isg_summary"] == isg.summary
    assert recommendation.model_summary["health_status"] == health.status


def test_recommendation_is_deterministic_and_does_not_mutate_model():
    model = make_model(
        [[1.0, 0.0, 0.0], [0.0, -2.0, 0.0]],
        objective=[1.0, 0.0, 3.0],
        var_types=[VarType.CONTINUOUS, VarType.INTEGER, VarType.BINARY],
    )
    original_matrix = model.A.copy()
    original_objective = model.c.copy()
    original_types = list(model.var_types)
    recommender = StrategyRecommender()

    first = recommender.recommend(model)
    second = recommender.recommend(model)

    assert first == second
    assert (model.A != original_matrix).nnz == 0
    assert np.array_equal(model.c, original_objective)
    assert model.var_types == original_types


def test_explanation_contains_concrete_model_and_graph_evidence():
    model = make_model([[1.0, 0.0], [0.0, 1.0]], objective=[1.0, 0.0])

    recommendation = StrategyRecommender().recommend(model)

    explanation = " ".join(recommendation.rationale).lower()
    assert "2 variable" in explanation
    assert "2 constraint" in explanation
    assert "2 nonzero constraint coefficients" in explanation
    assert "density" in explanation
    assert recommendation.model_summary["isg_summary"]["variable_constraint_edge_count"] == 2