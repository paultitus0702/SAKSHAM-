"""Build SAKSHAM's solver-independent Industrial Structure Graph.

The ISG is an intermediate representation of variable, constraint, and
objective relationships. Unlike Phase 3 fingerprinting, which summarizes
model characteristics, the ISG preserves the individual weighted connections
needed by future strategy and decomposition tools.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List

from scipy import sparse

from ..canonical.model import CanonicalModel
from ..canonical.validator import ModelValidationError, ModelValidator


@dataclass
class ISGGraphData:
    """Ordered typed nodes and weighted edges derived from a CanonicalModel.

    Nodes use ``var:<name>``, ``con:<name>``, and ``objective`` identifiers;
    edge types are ``VARIABLE_IN_CONSTRAINT`` and ``VARIABLE_IN_OBJECTIVE``.
    """

    layer_name: str = "bipartite"
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "UNAVAILABLE"
    summary: Dict[str, Any] = field(default_factory=dict)


class ISGBuilder:
    """Build graph nodes and relationships directly from canonical model data."""

    def build_bipartite_graph(self, model: CanonicalModel) -> ISGGraphData:
        """Create variable, constraint, and objective nodes plus coefficient edges.

        Variable-constraint links come from nonzero entries of ``model.A``;
        objective links come from nonzero entries of ``model.c``. No dense
        matrix is formed, and the input model is not modified.
        """
        if not isinstance(model, CanonicalModel):
            raise TypeError("model must be a CanonicalModel instance.")

        validation = ModelValidator.validate(model)
        if not validation.is_valid:
            raise ModelValidationError(
                f"Model '{model.name}' is invalid and cannot be converted to an ISG.",
                errors=validation.errors,
            )

        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        variable_ids = [f"var:{name}" for name in model.var_names]
        constraint_ids = [f"con:{name}" for name in model.constr_names]

        for index, name in enumerate(model.var_names):
            nodes.append(
                {
                    "id": variable_ids[index],
                    "type": "VARIABLE",
                    "name": name,
                    "index": index,
                    "var_type": model.var_types[index].value,
                    "lower_bound": float(model.l_var[index]),
                    "upper_bound": float(model.u_var[index]),
                }
            )

        for index, name in enumerate(model.constr_names):
            nodes.append(
                {
                    "id": constraint_ids[index],
                    "type": "CONSTRAINT",
                    "name": name,
                    "index": index,
                    "sense": model.constraint_senses[index].value,
                    "rhs": float(model.rhs[index]),
                }
            )

        nodes.append(
            {
                "id": "objective",
                "type": "OBJECTIVE",
                "sense": model.sense.value,
            }
        )

        matrix = model.A.tocsr(copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()

        variable_constraint_degrees = [0] * model.num_vars
        constraint_degrees = [0] * model.num_constrs
        for row_index, constraint_id in enumerate(constraint_ids):
            start, end = matrix.indptr[row_index : row_index + 2]
            for position in range(start, end):
                column_index = int(matrix.indices[position])
                variable_id = variable_ids[column_index]
                edges.append(
                    {
                        "source": variable_id,
                        "target": constraint_id,
                        "type": "VARIABLE_IN_CONSTRAINT",
                        "coefficient": float(matrix.data[position]),
                    }
                )
                variable_constraint_degrees[column_index] += 1
                constraint_degrees[row_index] += 1

        objective_degrees = [0] * model.num_vars
        for variable_index, coefficient in enumerate(model.c):
            if coefficient != 0.0:
                edges.append(
                    {
                        "source": "objective",
                        "target": variable_ids[variable_index],
                        "type": "VARIABLE_IN_OBJECTIVE",
                        "coefficient": float(coefficient),
                    }
                )
                objective_degrees[variable_index] = 1

        variable_degrees = [
            constraint_degree + objective_degree
            for constraint_degree, objective_degree in zip(
                variable_constraint_degrees, objective_degrees
            )
        ]
        possible_bipartite_edges = model.num_vars * model.num_constrs
        variable_constraint_edge_count = sum(variable_constraint_degrees)
        bipartite_density = (
            variable_constraint_edge_count / possible_bipartite_edges
            if possible_bipartite_edges
            else 0.0
        )
        objective_edge_count = sum(objective_degrees)
        objective_edge_density = (
            objective_edge_count / model.num_vars if model.num_vars else 0.0
        )
        isolated_variable_ids = [
            variable_ids[index]
            for index, degree in enumerate(variable_degrees)
            if degree == 0
        ]
        isolated_constraint_ids = [
            constraint_ids[index]
            for index, degree in enumerate(constraint_degrees)
            if degree == 0
        ]

        return ISGGraphData(
            layer_name="bipartite",
            nodes=nodes,
            edges=edges,
            summary={
                "variable_node_count": model.num_vars,
                "constraint_node_count": model.num_constrs,
                "objective_node_count": 1,
                "variable_constraint_edge_count": variable_constraint_edge_count,
                "objective_edge_count": objective_edge_count,
                "bipartite_density": bipartite_density,
                "objective_edge_density": objective_edge_density,
                "isolated_variable_ids": isolated_variable_ids,
                "isolated_constraint_ids": isolated_constraint_ids,
                "maximum_variable_degree": max(variable_degrees, default=0),
                "maximum_constraint_degree": max(constraint_degrees, default=0),
            },
            status="OK",
        )
