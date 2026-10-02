"""Deterministic heuristic strategy recommendations for optimization models.

Recommendations summarize evidence from the canonical model, Phase 3
fingerprinting and numerical health, and the Phase 4 ISG. They are guidance
for a future solver layer, not solver commands or claims of optimality.
The small-model limits below only qualify explanation text; they are not
solver capability limits or optimality claims.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..canonical.model import CanonicalModel, VarType
from ..fingerprint.analyzer import FingerprintResult, StructuralFingerprinter
from ..health.conditioner import HealthReport, NumericalHealthMonitor
from ..isg.graph_builder import ISGBuilder, ISGGraphData


_DIRECT_MODEL_VARIABLE_LIMIT = 1000
_DIRECT_MODEL_CONSTRAINT_LIMIT = 1000
_DIRECT_MODEL_NONZERO_LIMIT = 10000


@dataclass
class StrategyRecommendation:
    """Explainable heuristic recommendation consumed by the solver interface."""

    primary_algorithm: str = "dual_simplex"
    presolve: str = "on"
    scaling: str = "auto"
    rationale: List[str] = field(default_factory=list)
    status: str = "LIMITED"
    primary_strategy: str = "DIRECT_LP"
    secondary_strategies: List[str] = field(default_factory=list)
    confidence: str = "LOW"
    warnings: List[str] = field(default_factory=list)
    model_summary: Dict[str, Any] = field(default_factory=dict)


class StrategyRecommender:
    """Recommend a solver approach using existing model analyses.

    ``recommend(fingerprint)`` remains supported. Passing a CanonicalModel
    computes the existing fingerprint, health report, and ISG once; callers
    that already have those results can supply them to avoid repeated work.
    Structural candidates remain explicitly heuristic.
    """

    def recommend(
        self,
        fingerprint: Optional[Union[FingerprintResult, CanonicalModel]] = None,
        *,
        model: Optional[CanonicalModel] = None,
        health: Optional[HealthReport] = None,
        isg: Optional[ISGGraphData] = None,
    ) -> StrategyRecommendation:
        """Return a deterministic, evidence-backed strategy recommendation."""
        if isinstance(fingerprint, CanonicalModel):
            if model is not None:
                raise ValueError("Pass the CanonicalModel either positionally or by keyword, not both.")
            model = fingerprint
            fingerprint = None
        elif fingerprint is not None and not isinstance(fingerprint, FingerprintResult):
            raise TypeError(
                "fingerprint must be a FingerprintResult or CanonicalModel."
            )

        if model is not None and not isinstance(model, CanonicalModel):
            raise TypeError("model must be a CanonicalModel instance.")
        if fingerprint is None and model is None:
            raise ValueError("A CanonicalModel or FingerprintResult is required.")

        if model is not None:
            if fingerprint is None:
                fingerprint = StructuralFingerprinter().analyze(model)
            if fingerprint.status == "INVALID":
                if health is None:
                    health = NumericalHealthMonitor().evaluate(model)
                return self._invalid_recommendation(fingerprint, health)
            if health is None:
                health = NumericalHealthMonitor().evaluate(model)
            if health.status == "INVALID":
                return self._invalid_recommendation(fingerprint, health)
            if isg is None:
                isg = ISGBuilder().build_bipartite_graph(model)

        assert fingerprint is not None
        if fingerprint.status == "INVALID":
            return self._invalid_recommendation(fingerprint, health)
        if health is not None and health.status == "INVALID":
            return self._invalid_recommendation(fingerprint, health)

        summary = self._build_model_summary(fingerprint, health, isg)
        variable_type_counts = summary["variable_type_counts"]
        has_integer_variables = (
            variable_type_counts["integer"] + variable_type_counts["binary"] > 0
        )
        is_sparse = "sparse" in fingerprint.tags
        network_candidate = (
            "potentially_network_like" in fingerprint.tags
            or fingerprint.conservation_rows_detected > 0
        )
        knapsack_candidate = fingerprint.knapsack_rows_detected > 0
        block_status = summary["block_structure_status"]
        block_candidate = (
            fingerprint.block_angular_candidate
            or block_status in {"candidate", "detected"}
        )

        secondary_strategies: List[str] = []
        warnings: List[str] = []
        if has_integer_variables:
            primary_strategy = "GENERAL_MIP"
            primary_algorithm = "mip"
            rationale = [
                "Integer/binary variable counts are "
                f"{variable_type_counts['integer']} integer and "
                f"{variable_type_counts['binary']} binary; use a MIP-oriented approach."
            ]
            confidence = "HIGH"
        elif network_candidate:
            primary_strategy = "NETWORK_OR_FLOW_CANDIDATE"
            primary_algorithm = "dual_simplex"
            rationale = [
                "Fingerprint reports "
                f"{fingerprint.conservation_rows_detected} conservation-row candidate(s) "
                "and/or a potentially network-like coefficient pattern."
            ]
            confidence = "LOW"
            warnings.append(
                "Network/flow structure is a candidate signal, not a confirmed model class."
            )
            secondary_strategies.append(
                "GENERAL_SPARSE" if is_sparse else "DIRECT_LP"
            )
        elif is_sparse:
            primary_strategy = "GENERAL_SPARSE"
            primary_algorithm = "dual_simplex"
            rationale = [
                "All variables are continuous and fingerprint classifies the "
                "constraint matrix as sparse."
            ]
            confidence = "MEDIUM"
        else:
            primary_strategy = "DIRECT_LP"
            primary_algorithm = "dual_simplex"
            rationale = [
                "All variables are continuous and the fingerprint does not "
                "classify the matrix as sparse; a direct LP approach is the baseline."
            ]
            confidence = "MEDIUM"

        rationale.extend(self._size_evidence(fingerprint))
        rationale.extend(self._structural_evidence(fingerprint, isg))

        if network_candidate and has_integer_variables:
            secondary_strategies.append("NETWORK_OR_FLOW_CANDIDATE")
            warnings.append(
                "Network/flow structure is a candidate signal, not a confirmed model class."
            )
        if knapsack_candidate:
            secondary_strategies.append("BINARY_KNAPSACK_CANDIDATE")
            warnings.append(
                "Binary-knapsack structure is a conservative fingerprint candidate."
            )
        if block_candidate:
            secondary_strategies.append("DECOMPOSITION_CANDIDATE")
            warnings.append(
                "Block structure is a candidate signal; no decomposition is performed."
            )
        elif block_status == "unknown":
            warnings.append("Block structure is unknown; no decomposition strategy is inferred.")

        status = "READY"
        if health is None:
            status = "LIMITED"
            warnings.append("Numerical health was not supplied and is unavailable.")
            summary["health_status"] = "UNAVAILABLE"
        elif health.status == "WARNING":
            status = "WARNING"
            secondary_strategies.append("NUMERICAL_PRECONDITIONING")
            warnings.append(health.details)
        elif health.status == "UNAVAILABLE":
            status = "LIMITED"
            warnings.append(health.details)

        if isg is None:
            if status == "READY":
                status = "LIMITED"
            warnings.append("ISG was not supplied; graph evidence is unavailable.")

        if health is not None and health.status == "WARNING":
            confidence = "LOW" if confidence == "MEDIUM" else confidence

        return StrategyRecommendation(
            primary_algorithm=primary_algorithm,
            presolve="on",
            scaling="auto",
            rationale=rationale,
            status=status,
            primary_strategy=primary_strategy,
            secondary_strategies=self._unique(secondary_strategies),
            confidence=confidence,
            warnings=self._unique(warnings),
            model_summary=summary,
        )

    @staticmethod
    def _invalid_recommendation(
        fingerprint: FingerprintResult,
        health: Optional[HealthReport],
    ) -> StrategyRecommendation:
        warnings = list(fingerprint.raw_metrics.get("validation_errors", []))
        if health is not None:
            warnings.extend(health.validation_errors)
            if health.details:
                warnings.append(health.details)
        return StrategyRecommendation(
            primary_algorithm="not_recommended",
            rationale=["Model validation or numerical health reports invalid input."],
            status="INVALID",
            primary_strategy="INVALID_MODEL",
            confidence="LOW",
            warnings=StrategyRecommender._unique(warnings),
            model_summary=StrategyRecommender._build_model_summary(
                fingerprint, health, None
            ),
        )

    @staticmethod
    def _build_model_summary(
        fingerprint: FingerprintResult,
        health: Optional[HealthReport],
        isg: Optional[ISGGraphData],
    ) -> Dict[str, Any]:
        raw_metrics = fingerprint.raw_metrics
        summary: Dict[str, Any] = {
            "num_variables": fingerprint.num_vars,
            "num_constraints": fingerprint.num_constrs,
            "num_nonzeros": fingerprint.num_nonzeros,
            "density": fingerprint.density,
            "tags": list(fingerprint.tags),
            "variable_type_counts": dict(
                raw_metrics.get(
                    "variable_type_counts",
                    {"continuous": 0, "integer": 0, "binary": 0},
                )
            ),
            "constraint_sense_counts": dict(
                raw_metrics.get(
                    "constraint_sense_counts",
                    {"equal": 0, "less_equal": 0, "greater_equal": 0},
                )
            ),
            "conservation_rows_detected": fingerprint.conservation_rows_detected,
            "knapsack_rows_detected": fingerprint.knapsack_rows_detected,
            "network_structure_status": raw_metrics.get(
                "network_structure_status",
                "candidate"
                if "potentially_network_like" in fingerprint.tags
                else "unknown",
            ),
            "block_structure_status": raw_metrics.get("block_structure_status", "unknown"),
            "isg_summary": dict(isg.summary) if isg is not None else None,
            "health_status": health.status if health is not None else "UNAVAILABLE",
            "coefficient_dynamic_range": (
                health.dynamic_range if health is not None else None
            ),
            "zero_row_count": health.zero_row_count if health is not None else None,
            "zero_column_count": health.zero_column_count if health is not None else None,
        }
        return summary

    @staticmethod
    def _size_evidence(fingerprint: FingerprintResult) -> List[str]:
        evidence = [
            f"Model has {fingerprint.num_vars} variables, "
            f"{fingerprint.num_constrs} constraints, and "
            f"{fingerprint.num_nonzeros} nonzero constraint coefficients "
            "after duplicate and explicit-zero normalization.",
            f"Constraint-matrix density is {fingerprint.density:.6g}.",
        ]
        is_small = (
            fingerprint.num_vars <= _DIRECT_MODEL_VARIABLE_LIMIT
            and fingerprint.num_constrs <= _DIRECT_MODEL_CONSTRAINT_LIMIT
            and fingerprint.num_nonzeros <= _DIRECT_MODEL_NONZERO_LIMIT
        )
        evidence.append(
            "Model dimensions and nonzeros are within the small-model context limits."
            if is_small
            else "Model exceeds at least one small-model context limit."
        )
        return evidence

    @staticmethod
    def _structural_evidence(
        fingerprint: FingerprintResult,
        isg: Optional[ISGGraphData],
    ) -> List[str]:
        evidence: List[str] = []
        type_counts = fingerprint.raw_metrics.get("variable_type_counts", {})
        if type_counts:
            evidence.append(
                "Variable types: "
                f"{type_counts.get('continuous', 0)} continuous, "
                f"{type_counts.get('integer', 0)} integer, "
                f"{type_counts.get('binary', 0)} binary."
            )
        sense_counts = fingerprint.raw_metrics.get("constraint_sense_counts", {})
        if sense_counts:
            evidence.append(
                "Constraint senses: "
                f"{sense_counts.get('equal', 0)} equality, "
                f"{sense_counts.get('less_equal', 0)} <=, "
                f"{sense_counts.get('greater_equal', 0)} >=."
            )
        if isg is not None:
            evidence.append(
                "ISG reports "
                f"{isg.summary.get('variable_constraint_edge_count', 0)} "
                "variable-constraint edges, "
                f"{isg.summary.get('objective_edge_count', 0)} objective edges, "
                f"{len(isg.summary.get('isolated_variable_ids', []))} isolated variables, "
                f"and {len(isg.summary.get('isolated_constraint_ids', []))} isolated constraints."
            )
        return evidence

    @staticmethod
    def _unique(values: List[str]) -> List[str]:
        return list(dict.fromkeys(value for value in values if value))
