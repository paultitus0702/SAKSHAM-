"""
Phase 1 Scaffolding & Integration Tests.
Verifies engine imports, HiGHS availability, CanonicalModel data structures, and FastAPI health check.
"""
import pytest
from fastapi.testclient import TestClient
import numpy as np
from scipy import sparse
import highspy

import saksham
from saksham.canonical.model import CanonicalModel
from saksham.solvers.base import AbstractSolverBackend
from backend.app.main import app


def test_engine_package_imports():
    """Verify saksham engine imports and has expected version."""
    assert hasattr(saksham, "__version__")
    assert saksham.__version__ == "0.1.0-alpha"


def test_canonical_model_data_structures():
    """Verify CanonicalModel sparse matrix operations and properties."""
    # Create small 2x2 identity constraint matrix
    A = sparse.csr_matrix(np.eye(2, dtype=np.float64))
    model = CanonicalModel(
        name="test_scaffolding_model",
        num_vars=2,
        num_constrs=2,
        A=A,
        c=np.array([1.0, 2.0]),
        L_con=np.array([0.0, 0.0]),
        U_con=np.array([10.0, 10.0]),
        l_var=np.array([0.0, 0.0]),
        u_var=np.array([5.0, 5.0]),
    )
    assert model.num_nonzeros == 2
    assert model.density == 0.5  # 2 nonzeros out of 4 entries = 0.5


def test_highs_backend_initialization():
    """Verify that the official HiGHS C++ wrapper is functional."""
    h = highspy.Highs()
    version = h.version()
    assert version is not None
    assert len(version) > 0


def test_backend_health_endpoint():
    """Verify FastAPI /api/v1/health endpoint returns 200 and healthy status."""
    client = TestClient(app)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["engine"]["status"] == "OPERATIONAL"
    assert data["solvers"]["highs"]["available"] is True
