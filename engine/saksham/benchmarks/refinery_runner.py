"""Offline synthetic industrial benchmark suite built on the SAKSHAM pipeline."""
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from ..canonical.model import CanonicalModel
from ..pipeline import PipelineConfiguration, PipelineResult, run_pipeline
from ..scenarios.refinery import generate_refinery_scenario
from ..solvers.registry import SolverRegistry


_BENCHMARKS = {
    "refinery_small": {"num_periods": 1, "binary_operations": False},
    "refinery_blending": {"num_periods": 2, "binary_operations": False},
    "refinery_inventory": {"num_periods": 3, "binary_operations": False},
    "refinery_binary_operations": {"num_periods": 2, "binary_operations": True},
}


@dataclass(frozen=True)
class RefineryBenchmarkReport:
    """Measured model, analysis, solve, and verification details for a run."""

    benchmark_name: str
    seed: int
    num_variables: int
    num_constraints: int
    nonzeros: int
    objective_sense: str
    integer_variable_count: int
    binary_variable_count: int
    fingerprint_summary: Dict[str, Any]
    numerical_health_status: str
    recommended_strategy: Optional[Dict[str, Any]]
    solver_status: str
    objective_value: Optional[float]
    solve_time_seconds: Optional[float]
    sakshi_verification_status: str
    certificate_digest: Optional[str]
    pipeline_status: str
    success: bool
    errors: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible report."""
        report = asdict(self)
        report["errors"] = list(self.errors)
        return report


def generate_refinery_benchmarks(seed: int = 0) -> Dict[str, CanonicalModel]:
    """Build all small, deterministic Phase 10 models without external data."""
    return {
        name: generate_refinery_scenario(
            num_periods=configuration["num_periods"],
            seed=seed,
            binary_operations=configuration["binary_operations"],
            benchmark_name=name,
        )
        for name, configuration in _BENCHMARKS.items()
    }


class RefineryBenchmarkRunner:
    """Run named synthetic refinery profiles through the existing pipeline."""

    def __init__(
        self,
        configuration: Optional[PipelineConfiguration] = None,
        solver_registry: Optional[SolverRegistry] = None,
    ) -> None:
        self.configuration = configuration
        self.solver_registry = solver_registry

    def run_benchmark(self, benchmark_name: str, seed: int = 0) -> RefineryBenchmarkReport:
        """Generate and execute one benchmark; unknown names fail explicitly."""
        if not isinstance(benchmark_name, str) or benchmark_name not in _BENCHMARKS:
            raise ValueError(
                f"Unknown refinery benchmark '{benchmark_name}'. "
                f"Choose one of {', '.join(_BENCHMARKS)}."
            )
        benchmark_configuration = _BENCHMARKS[benchmark_name]
        model = generate_refinery_scenario(
            num_periods=benchmark_configuration["num_periods"],
            seed=seed,
            binary_operations=benchmark_configuration["binary_operations"],
            benchmark_name=benchmark_name,
        )
        pipeline_result = run_pipeline(
            model,
            configuration=self.configuration,
            solver_registry=self.solver_registry,
            input_name=benchmark_name,
        )
        return _build_report(benchmark_name, seed, model, pipeline_result)

    def run_suite(self, seed: int = 0) -> Dict[str, RefineryBenchmarkReport]:
        """Run the four profiles in stable order with the provided seed."""
        return {
            name: self.run_benchmark(name, seed=seed)
            for name in _BENCHMARKS
        }


def _build_report(
    benchmark_name: str,
    seed: int,
    model: CanonicalModel,
    result: PipelineResult,
) -> RefineryBenchmarkReport:
    fingerprint = result.fingerprint
    health = result.health
    strategy = result.strategy
    solver_result = result.solver_result
    certificate = result.certificate
    return RefineryBenchmarkReport(
        benchmark_name=benchmark_name,
        seed=int(seed),
        num_variables=model.num_vars,
        num_constraints=model.num_constrs,
        nonzeros=model.num_nonzeros,
        objective_sense=model.sense.value,
        integer_variable_count=model.integer_count,
        binary_variable_count=model.binary_count,
        fingerprint_summary=(
            {
                "status": fingerprint.status,
                "num_variables": fingerprint.num_vars,
                "num_constraints": fingerprint.num_constrs,
                "nonzeros": fingerprint.num_nonzeros,
                "density": fingerprint.density,
                "tags": list(fingerprint.tags),
                "conservation_rows_detected": fingerprint.conservation_rows_detected,
                "knapsack_rows_detected": fingerprint.knapsack_rows_detected,
            }
            if fingerprint is not None
            else {"status": "UNAVAILABLE"}
        ),
        numerical_health_status=health.status if health is not None else "UNAVAILABLE",
        recommended_strategy=(
            {
                "primary_strategy": strategy.primary_strategy,
                "primary_algorithm": strategy.primary_algorithm,
                "status": strategy.status,
                "confidence": strategy.confidence,
            }
            if strategy is not None
            else None
        ),
        solver_status=solver_result.status if solver_result is not None else "UNAVAILABLE",
        objective_value=(
            solver_result.objective_value if solver_result is not None else None
        ),
        solve_time_seconds=(
            solver_result.wall_time_seconds if solver_result is not None else None
        ),
        sakshi_verification_status=(
            result.verification.status if result.verification is not None else "UNAVAILABLE"
        ),
        certificate_digest=certificate.sha256_hash if certificate is not None else None,
        pipeline_status=result.status,
        success=result.success,
        errors=tuple(error.message for error in result.errors),
    )