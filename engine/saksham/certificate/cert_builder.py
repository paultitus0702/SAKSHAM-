"""Build deterministic certificates from solver output and SAKSHI checks.

Certificates preserve solver-reported fields separately from independently
verified metrics. They record feasibility checks but never claim that
optimality has been independently proven.
"""
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Dict, List, Optional

from ..canonical.model import CanonicalModel
from ..sakshi.verifier import SakshiVerificationResult, SakshiVerifier
from ..solvers.base import SolveResult


@dataclass
class SolveCertificate:
    model_name: str = ""
    status: str = "UNAVAILABLE"
    objective_value: Optional[float] = None
    primal_residual_norm: Optional[float] = None
    dual_residual_norm: Optional[float] = None
    sha256_hash: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    solver_name: Optional[str] = None
    solver_status: Optional[str] = None
    objective_sense: Optional[str] = None
    solver_reported_objective_value: Optional[float] = None
    independently_calculated_objective_value: Optional[float] = None
    verification_status: str = "UNAVAILABLE"
    feasibility_status: str = "UNAVAILABLE"
    integrality_status: str = "UNAVAILABLE"
    objective_consistency_status: str = "UNAVAILABLE"
    max_constraint_violation: Optional[float] = None
    max_bound_violation: Optional[float] = None
    max_integrality_violation: Optional[float] = None
    objective_discrepancy: Optional[float] = None
    verification_tolerance: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    optimality_claim: str = "NOT_VERIFIED"


class SolveCertificateGenerator:
    """Generate deterministic solve records with explicit evidence provenance."""

    def __init__(self, verifier: Optional[SakshiVerifier] = None) -> None:
        self._verifier = verifier or SakshiVerifier()

    def build_certificate(
        self,
        model: CanonicalModel,
        result: Optional[SolveResult],
        verification: Optional[SakshiVerificationResult] = None,
    ) -> SolveCertificate:
        """Create a SHA-256-sealed certificate without timestamps or claims.

        ``objective_value`` remains the legacy solver-reported value;
        ``independently_calculated_objective_value`` is SAKSHI's recomputation.
        """
        if not isinstance(model, CanonicalModel):
            raise TypeError("model must be a CanonicalModel instance.")
        if result is not None and not isinstance(result, SolveResult):
            raise TypeError("result must be a SolveResult or None.")
        if verification is None:
            verification = self._verifier.verify(model, result)
        elif not isinstance(verification, SakshiVerificationResult):
            raise TypeError(
                "verification must be a SakshiVerificationResult or None."
            )

        bounds_status, feasibility_status = self._feasibility_status(verification)
        integrality_status = self._check_status(verification.integrality_satisfied)
        objective_status = self._objective_status(verification.objective_consistent)
        constraint_norm = verification.details.get("constraint_residual_norm")
        if constraint_norm is not None:
            constraint_norm = self._finite_float_or_none(constraint_norm)

        certificate = SolveCertificate(
            model_name=model.name,
            status=getattr(result, "status", "UNAVAILABLE") if result else "UNAVAILABLE",
            objective_value=self._finite_float_or_none(
                getattr(result, "objective_value", None) if result else None
            ),
            primal_residual_norm=constraint_norm,
            solver_name=getattr(result, "solver_name", None) if result else None,
            solver_status=getattr(result, "status", None) if result else None,
            objective_sense=getattr(getattr(model, "sense", None), "value", None),
            solver_reported_objective_value=self._finite_float_or_none(
                getattr(result, "objective_value", None) if result else None
            ),
            independently_calculated_objective_value=(
                verification.independently_calculated_objective
            ),
            verification_status=verification.status,
            feasibility_status=feasibility_status,
            integrality_status=integrality_status,
            objective_consistency_status=objective_status,
            max_constraint_violation=verification.max_constraint_violation,
            max_bound_violation=verification.max_bound_violation,
            max_integrality_violation=verification.max_integrality_violation,
            objective_discrepancy=verification.objective_discrepancy,
            verification_tolerance=verification.feasibility_tolerance,
            warnings=list(verification.warnings),
            optimality_claim="NOT_VERIFIED",
            details={
                "solver_reported": {
                    "solver_name": getattr(result, "solver_name", None) if result else None,
                    "status": getattr(result, "status", None) if result else None,
                    "objective_value": self._finite_float_or_none(
                        getattr(result, "objective_value", None) if result else None
                    ),
                    "termination_reason": (
                        getattr(result, "termination_reason", None) if result else None
                    ),
                    "message": getattr(result, "message", None) if result else None,
                },
                "sakshi_verified": {
                    "status": verification.status,
                    "bounds_satisfied": verification.bounds_satisfied,
                    "constraints_satisfied": verification.constraints_satisfied,
                    "integrality_satisfied": verification.integrality_satisfied,
                    "objective_consistent": verification.objective_consistent,
                    "independently_calculated_objective": (
                        verification.independently_calculated_objective
                    ),
                },
                "unavailable_checks": list(verification.unavailable_checks),
                "not_claimed": [
                    "global optimality",
                    "dual feasibility or KKT conditions",
                ],
                "bounds_status": bounds_status,
            },
        )
        certificate.sha256_hash = self._digest(certificate)
        return certificate

    @staticmethod
    def _check_status(value: Optional[bool]) -> str:
        if value is None:
            return "UNAVAILABLE"
        return "SATISFIED" if value else "VIOLATED"

    @classmethod
    def _feasibility_status(
        cls,
        verification: SakshiVerificationResult,
    ) -> tuple[str, str]:
        bounds_status = cls._check_status(verification.bounds_satisfied)
        constraints_status = cls._check_status(verification.constraints_satisfied)
        if "VIOLATED" in {bounds_status, constraints_status}:
            return bounds_status, "VIOLATED"
        if "UNAVAILABLE" in {bounds_status, constraints_status}:
            return bounds_status, "UNAVAILABLE"
        return bounds_status, "SATISFIED"

    @staticmethod
    def _objective_status(value: Optional[bool]) -> str:
        if value is None:
            return "UNAVAILABLE"
        return "CONSISTENT" if value else "MISMATCH"

    @staticmethod
    def _finite_float_or_none(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return converted if converted == converted and abs(converted) != float("inf") else None

    @classmethod
    def _digest(cls, certificate: SolveCertificate) -> str:
        payload = {
            key: value
            for key, value in certificate.__dict__.items()
            if key != "sha256_hash"
        }
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()
