"""
SAKSHAM: Sovereign Structure-Aware Sparse Optimization Engine
Core Research Kernel Package
"""

__version__ = "0.1.0-alpha"
__author__ = "SAKSHAM Development Team"

from .canonical.model import CanonicalModel
from .solvers.base import AbstractSolverBackend, SolveResult

__all__ = [
    "CanonicalModel",
    "AbstractSolverBackend",
    "SolveResult",
    "__version__",
]
