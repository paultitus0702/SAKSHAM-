"""
Phase 1 Scaffolding & Integration Tests.

Verifies engine imports, HiGHS availability, CanonicalModel data structures,
and the FastAPI health endpoint.  These tests must remain green throughout
all subsequent development phases.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy import sparse

import highspy
import saksham
from saksham.canonical.model import (
    CanonicalModel,
    ConstraintSense,
    ObjectiveSense,
    VarType,
)
from saksham.solvers.base import AbstractSolverBackend

from backend.app.main import app


def test_engine_package_imports():
    """Verify saksham engine imports and has expected version."""
    assert hasattr(saksham, "__version__")
    assert saksham.__version__ == "0.1.0-alpha"


def test_canonical_model_sparse_matrix_properties():
    """
    Verify CanonicalModel stores a sparse constraint matrix and exposes
    correct num_nonzeros / density properties.

    Model:
        min  x1 + 2*x2
        s.t. x1       <= 5     (row 0)
             x2       <= 5     (row 1)
        0 <= x1, x2 <= 5
    """
    A = sparse.csr_matrix(np.eye(2, dtype=np.float64))  # 2 nonzeros
    model = CanonicalModel(
        name="phase1_scaffold_test",
        sense=ObjectiveSense.MINIMIZE,
        c=np.array([1.0, 2.0]),
        A=A,
        rhs=np.array([5.0, 5.0]),
        constraint_senses=[ConstraintSense.LESS_EQUAL, ConstraintSense.LESS_EQUAL],
        l_var=np.array([0.0, 0.0]),
        u_var=np.array([5.0, 5.0]),
        var_types=[VarType.CONTINUOUS, VarType.CONTINUOUS],
        var_names=["x1", "x2"],
        constr_names=["c1", "c2"],
    )
    assert model.num_vars == 2
    assert model.num_constrs == 2
    assert model.num_nonzeros == 2
    assert model.density == 0.5   # 2 nnz / (2 vars * 2 constrs)


def test_highs_backend_initialization():
    """Verify that the official HiGHS C++ wrapper initialises without error."""
    h = highspy.Highs()
    version = h.version()
    assert version is not None
    assert len(version) > 0


def test_backend_health_endpoint():
    """Verify FastAPI /api/v1/health returns 200 with OPERATIONAL engine status."""
    client = TestClient(app)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["engine"]["status"] == "OPERATIONAL"
    assert data["solvers"]["highs"]["available"] is True
