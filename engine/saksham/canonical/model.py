"""
SAKSHAM Canonical Internal Representation (CIR).

Represents LP, MILP optimization problems in standard matrix form:

    minimize (or maximize)   c^T x  +  c0
    subject to:
        A x  {<=, =, >=}  rhs        (row-wise constraint senses)
        l_var  <=  x  <=  u_var
        x_j in { Real, Integer, {0,1} }

The representation is independent of HiGHS, SciPy, or any other solver backend.
Future modules (fingerprinting, ISG, strategy, health, solver, Sakshi, certificate)
consume this object as their common input contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ObjectiveSense(str, Enum):
    """Direction of the optimization objective."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ConstraintSense(str, Enum):
    """
    Relational operator for a single linear constraint  A_i x {sense} rhs_i.

    Members are named after their mathematical meaning so code that reads
    ``ConstraintSense.LESS_EQUAL`` is self-documenting.
    """

    LESS_EQUAL = "<="
    EQUAL = "=="
    GREATER_EQUAL = ">="

    # Convenient aliases kept as class-level references (not Enum members)
    @classmethod
    def from_string(cls, s: str) -> "ConstraintSense":
        """Parse common string representations into ConstraintSense."""
        mapping = {
            "<=": cls.LESS_EQUAL,
            "<": cls.LESS_EQUAL,
            "=": cls.EQUAL,
            "==": cls.EQUAL,
            ">=": cls.GREATER_EQUAL,
            ">": cls.GREATER_EQUAL,
            "L": cls.LESS_EQUAL,   # MPS row type
            "E": cls.EQUAL,        # MPS row type
            "G": cls.GREATER_EQUAL,# MPS row type
        }
        result = mapping.get(s.strip())
        if result is None:
            raise ValueError(
                f"Unknown constraint sense string '{s}'. "
                f"Accepted: {list(mapping.keys())}"
            )
        return result


class VarType(str, Enum):
    """Integrality domain of a decision variable."""

    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BINARY = "binary"


# ---------------------------------------------------------------------------
# CanonicalModel
# ---------------------------------------------------------------------------


@dataclass
class CanonicalModel:
    """
    Core mathematical representation of an optimization problem in SAKSHAM.

    All arrays and lists use 0-based indexing throughout the engine.

    Attributes
    ----------
    name : str
        Human-readable identifier for the model instance.
    sense : ObjectiveSense
        Whether the objective is minimized or maximized.
    c : np.ndarray, shape (n,), dtype float64
        Linear objective coefficient vector.
    c0 : float
        Constant objective offset (does not affect the optimal solution).
    A : scipy.sparse.csr_matrix, shape (m, n), dtype float64
        Sparse constraint coefficient matrix in Compressed Sparse Row format.
        CSR is chosen because row-slicing (constraint access) is O(nnz_row).
    rhs : np.ndarray, shape (m,), dtype float64
        Right-hand-side values.  Together with ``constraint_senses`` they
        define:  A[i, :] x  {sense_i}  rhs[i].
    constraint_senses : list of ConstraintSense, length m
        Per-row relational operators.
    l_var : np.ndarray, shape (n,), dtype float64
        Variable lower bounds.  -np.inf is allowed (free variable).
    u_var : np.ndarray, shape (n,), dtype float64
        Variable upper bounds.  +np.inf is allowed (no explicit upper bound).
    var_types : list of VarType, length n
        Domain of each variable (CONTINUOUS, INTEGER, or BINARY).
    var_names : list of str, length n
        Unique name for each variable (column).
    constr_names : list of str, length m
        Unique name for each constraint (row).
    var_name_to_idx : dict {str: int}
        Reverse lookup: variable name → 0-based column index.
        Populated automatically in ``__post_init__`` if left empty.
    constr_name_to_idx : dict {str: int}
        Reverse lookup: constraint name → 0-based row index.
        Populated automatically in ``__post_init__`` if left empty.
    """

    # ---- identity ----
    name: str = "unnamed_model"
    sense: ObjectiveSense = ObjectiveSense.MINIMIZE

    # ---- objective ----
    c: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    c0: float = 0.0

    # ---- constraint matrix ----
    A: sparse.csr_matrix = field(
        default_factory=lambda: sparse.csr_matrix((0, 0), dtype=np.float64)
    )

    # ---- constraint right-hand side and senses ----
    rhs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    constraint_senses: List[ConstraintSense] = field(default_factory=list)

    # ---- variable bounds ----
    l_var: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    u_var: np.ndarray = field(
        default_factory=lambda: np.full(0, np.inf, dtype=np.float64)
    )

    # ---- variable metadata ----
    var_types: List[VarType] = field(default_factory=list)
    var_names: List[str] = field(default_factory=list)
    constr_names: List[str] = field(default_factory=list)

    # ---- reverse-lookup indices (auto-built) ----
    var_name_to_idx: Dict[str, int] = field(default_factory=dict)
    constr_name_to_idx: Dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Post-initialisation normalisation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Normalise array dtypes and build reverse-lookup dicts."""
        self.c = _to_float64_array(self.c)
        self.rhs = _to_float64_array(self.rhs)
        self.l_var = _to_float64_array(self.l_var)
        self.u_var = _to_float64_array(self.u_var)

        # Ensure the constraint matrix is CSR float64
        if self.A is not None and not sparse.issparse(self.A):
            self.A = sparse.csr_matrix(np.asarray(self.A, dtype=np.float64))
        elif self.A is not None and not sparse.isspmatrix_csr(self.A):
            self.A = self.A.tocsr()
        if self.A is not None and self.A.dtype != np.float64:
            self.A = self.A.astype(np.float64)

        # Auto-build reverse-lookup dicts when callers omit them
        if not self.var_name_to_idx and self.var_names:
            self.var_name_to_idx = {n: i for i, n in enumerate(self.var_names)}
        if not self.constr_name_to_idx and self.constr_names:
            self.constr_name_to_idx = {n: i for i, n in enumerate(self.constr_names)}

    # ------------------------------------------------------------------
    # Derived dimensional properties
    # ------------------------------------------------------------------

    @property
    def num_vars(self) -> int:
        """Number of decision variables (columns)."""
        return len(self.var_names)

    @property
    def num_constrs(self) -> int:
        """Number of linear constraints (rows)."""
        return len(self.constr_names)

    @property
    def num_nonzeros(self) -> int:
        """Number of non-zero coefficients in the constraint matrix A."""
        return int(self.A.nnz) if self.A is not None else 0

    @property
    def density(self) -> float:
        """Fill density of A:  nnz / (m * n).  Returns 0 for empty models."""
        total = self.num_vars * self.num_constrs
        return float(self.num_nonzeros / total) if total > 0 else 0.0

    # ---- variable-type counts ----

    @property
    def continuous_count(self) -> int:
        return sum(1 for vt in self.var_types if vt == VarType.CONTINUOUS)

    @property
    def integer_count(self) -> int:
        return sum(1 for vt in self.var_types if vt == VarType.INTEGER)

    @property
    def binary_count(self) -> int:
        return sum(1 for vt in self.var_types if vt == VarType.BINARY)

    @property
    def is_mip(self) -> bool:
        """True when the model contains at least one integer or binary variable."""
        return (self.integer_count + self.binary_count) > 0

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def compute_row_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert (rhs, constraint_senses) to interval form  L_con <= A x <= U_con.

        This is the format expected by most LP/MIP solver APIs (e.g., HiGHS).

        Returns
        -------
        L_con : np.ndarray, shape (m,)
            Lower bound for each row (-np.inf for <= constraints).
        U_con : np.ndarray, shape (m,)
            Upper bound for each row (+np.inf for >= constraints).

        Raises
        ------
        ValueError
            If an unknown ConstraintSense is encountered.
        """
        m = self.num_constrs
        L_con = np.full(m, -np.inf, dtype=np.float64)
        U_con = np.full(m, np.inf, dtype=np.float64)

        for i, (sense, rhs_val) in enumerate(zip(self.constraint_senses, self.rhs)):
            if sense == ConstraintSense.LESS_EQUAL:
                U_con[i] = rhs_val
            elif sense == ConstraintSense.EQUAL:
                L_con[i] = rhs_val
                U_con[i] = rhs_val
            elif sense == ConstraintSense.GREATER_EQUAL:
                L_con[i] = rhs_val
            else:
                raise ValueError(
                    f"Unknown ConstraintSense '{sense}' at row index {i}."
                )

        return L_con, U_con

    def summary(self) -> Dict[str, Any]:
        """
        Return a serialisable dictionary summarising model dimensions.

        This is used by the API layer and logging; it contains no mathematical
        detail beyond what is safe to expose as metadata.
        """
        sense_counts = {
            "less_equal": sum(
                1 for s in self.constraint_senses if s == ConstraintSense.LESS_EQUAL
            ),
            "equal": sum(
                1 for s in self.constraint_senses if s == ConstraintSense.EQUAL
            ),
            "greater_equal": sum(
                1 for s in self.constraint_senses if s == ConstraintSense.GREATER_EQUAL
            ),
        }
        return {
            "name": self.name,
            "sense": self.sense.value,
            "num_variables": self.num_vars,
            "num_constraints": self.num_constrs,
            "num_nonzeros": self.num_nonzeros,
            "density": round(self.density, 6),
            "variable_types": {
                "continuous": self.continuous_count,
                "integer": self.integer_count,
                "binary": self.binary_count,
            },
            "constraint_senses": sense_counts,
            "is_mip": self.is_mip,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_float64_array(value: Any) -> np.ndarray:
    """Coerce *value* to a 1-D float64 numpy array without silent data loss."""
    if isinstance(value, np.ndarray):
        if value.dtype == np.float64:
            return value
        return value.astype(np.float64)
    return np.asarray(value, dtype=np.float64)
