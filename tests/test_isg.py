"""Phase 4 Industrial Structure Graph tests."""

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
from saksham.isg.graph_builder import ISGBuilder, ISGGraphData


def make_model(
    matrix,
    objective=None,
    constraint_senses=None,
    var_types=None,
    sense=ObjectiveSense.MINIMIZE,
):
    matrix = sparse.csr_matrix(matrix, dtype=np.float64)
    num_constraints, num_variables = matrix.shape
    if objective is None:
        objective = np.arange(1, num_variables + 1, dtype=np.float64)
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
        name="isg_test_model",
        sense=sense,
        c=np.asarray(objective, dtype=np.float64),
        A=matrix,
        rhs=np.arange(10, 10 + num_constraints, dtype=np.float64),
        constraint_senses=constraint_senses,
        l_var=lower_bounds,
        u_var=upper_bounds,
        var_types=var_types,
        var_names=[f"x{index + 1}" for index in range(num_variables)],
        constr_names=[f"c{index + 1}" for index in range(num_constraints)],
    )


def test_builder_rejects_non_model_and_invalid_model():
    builder = ISGBuilder()

    with pytest.raises(TypeError):
        builder.build_bipartite_graph(None)
    with pytest.raises(ModelValidationError):
        builder.build_bipartite_graph(CanonicalModel())


def test_graph_data_preserves_existing_positional_status_argument():
    graph = ISGGraphData("bipartite", [], [], "legacy-status")

    assert graph.status == "legacy-status"
    assert graph.summary == {}


def test_graph_contains_typed_nodes_and_weighted_edges():
    model = make_model(
        [[2.5, 0.0, -1.0], [0.0, 4.0, 0.0]],
        objective=[3.0, 0.0, -2.0],
        constraint_senses=[ConstraintSense.EQUAL, ConstraintSense.GREATER_EQUAL],
        var_types=[VarType.CONTINUOUS, VarType.INTEGER, VarType.BINARY],
        sense=ObjectiveSense.MAXIMIZE,
    )

    graph = ISGBuilder().build_bipartite_graph(model)

    variable_nodes = [node for node in graph.nodes if node["type"] == "VARIABLE"]
    constraint_nodes = [node for node in graph.nodes if node["type"] == "CONSTRAINT"]
    objective_nodes = [node for node in graph.nodes if node["type"] == "OBJECTIVE"]
    assert len(variable_nodes) == 3
    assert len(constraint_nodes) == 2
    assert len(objective_nodes) == 1
    assert objective_nodes[0] == {
        "id": "objective",
        "type": "OBJECTIVE",
        "sense": "maximize",
    }

    variable_constraint_edges = [
        edge for edge in graph.edges if edge["type"] == "VARIABLE_IN_CONSTRAINT"
    ]
    objective_edges = [
        edge for edge in graph.edges if edge["type"] == "VARIABLE_IN_OBJECTIVE"
    ]
    assert variable_constraint_edges == [
        {
            "source": "var:x1",
            "target": "con:c1",
            "type": "VARIABLE_IN_CONSTRAINT",
            "coefficient": 2.5,
        },
        {
            "source": "var:x3",
            "target": "con:c1",
            "type": "VARIABLE_IN_CONSTRAINT",
            "coefficient": -1.0,
        },
        {
            "source": "var:x2",
            "target": "con:c2",
            "type": "VARIABLE_IN_CONSTRAINT",
            "coefficient": 4.0,
        },
    ]
    assert objective_edges == [
        {
            "source": "objective",
            "target": "var:x1",
            "type": "VARIABLE_IN_OBJECTIVE",
            "coefficient": 3.0,
        },
        {
            "source": "objective",
            "target": "var:x3",
            "type": "VARIABLE_IN_OBJECTIVE",
            "coefficient": -2.0,
        },
    ]
    assert graph.status == "OK"


def test_graph_preserves_variable_and_constraint_metadata():
    model = make_model(
        [[1.0, 0.0], [0.0, 1.0]],
        var_types=[VarType.INTEGER, VarType.BINARY],
        constraint_senses=[ConstraintSense.LESS_EQUAL, ConstraintSense.EQUAL],
    )

    graph = ISGBuilder().build_bipartite_graph(model)
    nodes_by_id = {node["id"]: node for node in graph.nodes}

    assert nodes_by_id["var:x1"] == {
        "id": "var:x1",
        "type": "VARIABLE",
        "name": "x1",
        "index": 0,
        "var_type": "integer",
        "lower_bound": 0.0,
        "upper_bound": float("inf"),
    }
    assert nodes_by_id["var:x2"]["var_type"] == "binary"
    assert nodes_by_id["var:x2"]["upper_bound"] == 1.0
    assert nodes_by_id["con:c1"] == {
        "id": "con:c1",
        "type": "CONSTRAINT",
        "name": "c1",
        "index": 0,
        "sense": "<=",
        "rhs": 10.0,
    }
    assert nodes_by_id["con:c2"]["sense"] == "=="
    assert nodes_by_id["con:c2"]["rhs"] == 11.0


def test_zero_coefficients_do_not_create_edges():
    graph = ISGBuilder().build_bipartite_graph(
        make_model([[0.0, 2.0]], objective=[0.0, 5.0])
    )

    assert len(graph.edges) == 2
    assert all(edge["coefficient"] != 0.0 for edge in graph.edges)


def test_graph_summary_reports_counts_density_degrees_and_isolates():
    model = make_model(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        objective=[1.0, 0.0, 0.0, 0.0],
    )

    graph = ISGBuilder().build_bipartite_graph(model)

    assert graph.summary == {
        "variable_node_count": 4,
        "constraint_node_count": 2,
        "objective_node_count": 1,
        "variable_constraint_edge_count": 1,
        "objective_edge_count": 1,
        "bipartite_density": 0.125,
        "objective_edge_density": 0.25,
        "isolated_variable_ids": ["var:x2", "var:x3", "var:x4"],
        "isolated_constraint_ids": ["con:c2"],
        "maximum_variable_degree": 2,
        "maximum_constraint_degree": 1,
    }


def test_mixed_integer_graph_retains_variable_types():
    model = make_model(
        [[1.0, 1.0]],
        var_types=[VarType.CONTINUOUS, VarType.BINARY],
    )

    graph = ISGBuilder().build_bipartite_graph(model)

    variable_types = {
        node["name"]: node["var_type"]
        for node in graph.nodes
        if node["type"] == "VARIABLE"
    }
    assert variable_types == {"x1": "continuous", "x2": "binary"}


def test_graph_construction_is_deterministic():
    model = make_model([[1.0, 0.0], [-2.0, 3.0]], objective=[0.0, 5.0])
    builder = ISGBuilder()

    first = builder.build_bipartite_graph(model)
    second = builder.build_bipartite_graph(model)

    assert first.nodes == second.nodes
    assert first.edges == second.edges
    assert first.summary == second.summary


def test_graph_builder_does_not_densify_or_mutate_sparse_matrix(monkeypatch):
    matrix = sparse.eye(100, format="csr")
    model = make_model(matrix)
    original_nnz = model.A.nnz

    def fail_toarray(*args, **kwargs):
        raise AssertionError("Sparse matrix was densified")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", fail_toarray)

    graph = ISGBuilder().build_bipartite_graph(model)

    assert len(graph.edges) == 200
    assert sparse.isspmatrix_csr(model.A)
    assert model.A.nnz == original_nnz