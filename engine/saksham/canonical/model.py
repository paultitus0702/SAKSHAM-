"""
Canonical Internal Representation (CIR) data structure.
Defines sparse LP/MILP/QP models in standard matrix form:
    min  c^T x + 0.5 x^T Q x
    s.t. L <= A x <= U
         l <= x <= u
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np
from scipy import sparse


@dataclass
class CanonicalModel:
    """
    Standard Canonical Model container using SciPy CSR/CSC sparse matrices.
    Phase 1 Scaffolding: Minimal interface definition. Full validation in Phase 2.
    """
    name: str = "unnamed_model"
    num_vars: int = 0
    num_constrs: int = 0
    
    # Cost vector c (shape: [num_vars])
    c: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    
    # Sparse constraint matrix A (CSR format)
    A: sparse.csr_matrix = field(default_factory=lambda: sparse.csr_matrix((0, 0), dtype=np.float64))
    
    # Constraint bounds: L <= A x <= U
    L_con: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    U_con: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    
    # Variable bounds: l <= x <= u
    l_var: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    u_var: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    
    # Variable metadata
    var_names: List[str] = field(default_factory=list)
    constr_names: List[str] = field(default_factory=list)
    
    # Status label
    _status: str = "SCAFFOLDING_STUB"

    @property
    def num_nonzeros(self) -> int:
        return self.A.nnz if self.A is not None else 0

    @property
    def density(self) -> float:
        total = self.num_vars * self.num_constrs
        return float(self.num_nonzeros / total) if total > 0 else 0.0
