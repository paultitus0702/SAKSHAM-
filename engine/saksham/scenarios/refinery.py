"""Deterministic synthetic refinery-style planning model generation."""
from typing import Dict, List

import numpy as np
from scipy import sparse

from ..canonical.model import (
    CanonicalModel,
    ConstraintSense,
    ObjectiveSense,
    VarType,
)
from ..canonical.validator import ModelValidator


_FEEDS = ("light_crude", "heavy_crude")
_PRODUCTS = ("gasoline", "distillate")
_BASE_YIELDS = np.array([[0.94, 0.82], [0.86, 0.91]], dtype=np.float64)


def generate_refinery_scenario(
    num_periods: int = 3,
    seed: int = 0,
    *,
    binary_operations: bool = False,
    benchmark_name: str = "refinery_custom",
) -> CanonicalModel:
    """Generate a small synthetic refinery planning LP or MILP.

    This illustrative model is generated locally and is not real refinery
    data. The seed controls all varying coefficients and right-hand sides.
    """
    if isinstance(num_periods, bool) or not isinstance(num_periods, int):
        raise ValueError("num_periods must be an integer.")
    if not 1 <= num_periods <= 12:
        raise ValueError("num_periods must be between 1 and 12.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if not isinstance(binary_operations, bool):
        raise ValueError("binary_operations must be a boolean.")
    if not isinstance(benchmark_name, str) or not benchmark_name.strip():
        raise ValueError("benchmark_name must be a non-empty string.")

    random = np.random.default_rng(int(seed))
    yields = _BASE_YIELDS + random.uniform(-0.015, 0.015, size=_BASE_YIELDS.shape)
    feed_availability = random.uniform(5.5, 7.0, size=(num_periods, len(_FEEDS)))
    processing_capacity = random.uniform(8.5, 10.0, size=num_periods)
    demand_minimum = random.uniform(1.2, 1.8, size=(num_periods, len(_PRODUCTS)))
    demand_maximum = demand_minimum + random.uniform(
        1.0, 2.0, size=(num_periods, len(_PRODUCTS))
    )
    feed_cost = random.uniform(2.0, 4.0, size=len(_FEEDS))
    product_revenue = random.uniform(8.0, 10.0, size=len(_PRODUCTS))
    inventory_cost = random.uniform(
        0.1, 0.3, size=(num_periods, len(_PRODUCTS))
    )

    var_names: List[str] = []
    var_types: List[VarType] = []
    lower_bounds: List[float] = []
    upper_bounds: List[float] = []
    objective: List[float] = []

    def add_variable(
        name: str,
        var_type: VarType,
        lower: float,
        upper: float,
        objective_coefficient: float,
    ) -> None:
        var_names.append(name)
        var_types.append(var_type)
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        objective.append(objective_coefficient)

    allocation_names: Dict[tuple[int, int, int], str] = {}
    inventory_names: Dict[tuple[int, int], str] = {}
    sales_names: Dict[tuple[int, int], str] = {}
    operating_names: Dict[int, str] = {}

    for period in range(num_periods):
        for feed_index, feed in enumerate(_FEEDS):
            for product_index, product in enumerate(_PRODUCTS):
                name = f"feed_t{period + 1}_{feed}_to_{product}"
                allocation_names[period, feed_index, product_index] = name
                add_variable(
                    name,
                    VarType.CONTINUOUS,
                    0.0,
                    float(feed_availability[period, feed_index]),
                    -float(feed_cost[feed_index]),
                )
        for product_index, product in enumerate(_PRODUCTS):
            inventory_name = f"inventory_t{period + 1}_{product}"
            inventory_names[period, product_index] = inventory_name
            add_variable(
                inventory_name,
                VarType.CONTINUOUS,
                0.0,
                5.0,
                -float(inventory_cost[period, product_index]),
            )
            sales_name = f"sales_t{period + 1}_{product}"
            sales_names[period, product_index] = sales_name
            add_variable(
                sales_name,
                VarType.CONTINUOUS,
                float(demand_minimum[period, product_index]),
                float(demand_maximum[period, product_index]),
                float(product_revenue[product_index]),
            )
        if binary_operations:
            operating_name = f"operating_t{period + 1}"
            operating_names[period] = operating_name
            add_variable(operating_name, VarType.BINARY, 0.0, 1.0, -2.0)

    variable_indices = {name: index for index, name in enumerate(var_names)}
    row_indices: List[int] = []
    column_indices: List[int] = []
    coefficients: List[float] = []
    constraint_names: List[str] = []
    constraint_senses: List[ConstraintSense] = []
    right_hand_sides: List[float] = []

    def add_constraint(
        name: str,
        terms: Dict[str, float],
        sense: ConstraintSense,
        rhs: float,
    ) -> None:
        row = len(constraint_names)
        constraint_names.append(name)
        constraint_senses.append(sense)
        right_hand_sides.append(rhs)
        for variable_name, coefficient in terms.items():
            if coefficient:
                row_indices.append(row)
                column_indices.append(variable_indices[variable_name])
                coefficients.append(float(coefficient))

    for period in range(num_periods):
        for feed_index, feed in enumerate(_FEEDS):
            add_constraint(
                f"feed_availability_t{period + 1}_{feed}",
                {
                    allocation_names[period, feed_index, product_index]: 1.0
                    for product_index in range(len(_PRODUCTS))
                },
                ConstraintSense.LESS_EQUAL,
                float(feed_availability[period, feed_index]),
            )

        capacity_terms = {
            allocation_names[period, feed_index, product_index]: 1.0
            for feed_index in range(len(_FEEDS))
            for product_index in range(len(_PRODUCTS))
        }
        capacity_rhs = float(processing_capacity[period])
        if binary_operations:
            capacity_terms[operating_names[period]] = -capacity_rhs
            capacity_rhs = 0.0
        add_constraint(
            f"processing_capacity_t{period + 1}",
            capacity_terms,
            ConstraintSense.LESS_EQUAL,
            capacity_rhs,
        )

        for product_index, product in enumerate(_PRODUCTS):
            balance_terms: Dict[str, float] = {
                allocation_names[period, feed_index, product_index]: float(
                    yields[feed_index, product_index]
                )
                for feed_index in range(len(_FEEDS))
            }
            if period:
                balance_terms[inventory_names[period - 1, product_index]] = 1.0
            balance_terms[inventory_names[period, product_index]] = -1.0
            balance_terms[sales_names[period, product_index]] = -1.0
            add_constraint(
                f"inventory_balance_t{period + 1}_{product}",
                balance_terms,
                ConstraintSense.EQUAL,
                0.0,
            )

    matrix = sparse.csr_matrix(
        (coefficients, (row_indices, column_indices)),
        shape=(len(constraint_names), len(var_names)),
        dtype=np.float64,
    )
    model = CanonicalModel(
        name=(
            f"Synthetic refinery planning benchmark: {benchmark_name.strip()} "
            f"(seed={int(seed)})"
        ),
        sense=ObjectiveSense.MAXIMIZE,
        c=np.asarray(objective, dtype=np.float64),
        A=matrix,
        rhs=np.asarray(right_hand_sides, dtype=np.float64),
        constraint_senses=constraint_senses,
        l_var=np.asarray(lower_bounds, dtype=np.float64),
        u_var=np.asarray(upper_bounds, dtype=np.float64),
        var_types=var_types,
        var_names=var_names,
        constr_names=constraint_names,
    )
    ModelValidator.validate(model, raise_on_error=True)
    return model
