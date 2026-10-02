"""Connect the existing SAKSHAM analysis and solve components.

The pipeline coordinates existing modules; it does not duplicate their
calculations. It accepts LP/MPS paths, records stage-specific errors, and
returns JSON-safe summaries without exposing the CanonicalModel object.
"""
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np

from ..canonical.model import CanonicalModel
from ..canonical.validator import ModelValidator
from ..certificate.cert_builder import SolveCertificate, SolveCertificateGenerator
from ..fingerprint.analyzer import FingerprintResult, StructuralFingerprinter
from ..health.conditioner import HealthReport, NumericalHealthMonitor
from ..isg.graph_builder import ISGBuilder, ISGGraphData
from ..parser.base import ModelParseError
from ..parser.lp_parser import LPParser
from ..parser.mps_parser import MPSParser
from ..sakshi.verifier import SakshiVerificationResult, SakshiVerifier
from ..solvers.base import SolveResult
from ..solvers.registry import SolverRegistry, SolverUnavailableError, UnsupportedStrategyError
from ..strategy.recommender import StrategyRecommendation, StrategyRecommender


@dataclass(frozen=True)
class PipelineConfiguration:
    feasibility_tolerance: float = 1e-7
    integrality_tolerance: float = 1e-6


@dataclass
class PipelineError:
    stage: str
    message: str
    code: Optional[str] = None


@dataclass
class PipelineResult:
    success: bool = False
    status: str = "FAILED"
    input_name: Optional[str] = None
    input_format: Optional[str] = None
    parser_status: str = "NOT_STARTED"
    model_summary: Optional[Dict[str, Any]] = None
    fingerprint: Optional[FingerprintResult] = None
    health: Optional[HealthReport] = None
    isg: Optional[ISGGraphData] = None
    strategy: Optional[StrategyRecommendation] = None
    selected_solver: Optional[str] = None
    solver_executed: bool = False
    solver_result: Optional[SolveResult] = None
    verification: Optional[SakshiVerificationResult] = None
    certificate: Optional[SolveCertificate] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[PipelineError] = field(default_factory=list)
    stage: str = "INPUT"

    def to_dict(self) -> Dict[str, Any]:
        """Return an API-safe representation without paths or sparse objects."""
        return {
            "success": self.success,
            "status": self.status,
            "stage": self.stage,
            "input": {"name": self.input_name, "format": self.input_format},
            "parser_status": self.parser_status,
            "model_summary": _json_safe(self.model_summary),
            "fingerprint": _json_safe(self.fingerprint),
            "health": _json_safe(self.health),
            "isg": (
                {"status": self.isg.status, "summary": _json_safe(self.isg.summary)}
                if self.isg is not None
                else None
            ),
            "strategy": _json_safe(self.strategy),
            "selected_solver": self.selected_solver,
            "solver_executed": self.solver_executed,
            "solver_result": _json_safe(self.solver_result),
            "verification": _json_safe(self.verification),
            "certificate": _json_safe(self.certificate),
            "warnings": list(self.warnings),
            "errors": [_json_safe(error) for error in self.errors],
        }


def run_pipeline(
    file_path: Union[str, Path],
    configuration: Optional[Union[PipelineConfiguration, Mapping[str, Any]]] = None,
    *,
    solver_registry: Optional[SolverRegistry] = None,
    input_name: Optional[str] = None,
) -> PipelineResult:
    """Run parser → analysis → recommendation → solve → verification → certificate."""
    path = Path(file_path) if file_path is not None else None
    result = PipelineResult(input_name=input_name or (path.name if path else None))
    if path is None:
        return _record_error(result, "INPUT", "A model file path is required.", "MISSING_INPUT")
    if not path.is_file():
        return _record_error(result, "INPUT", "The requested model file does not exist.", "FILE_NOT_FOUND")

    extension = path.suffix.lower()
    result.input_format = extension.removeprefix(".")
    parser_by_extension = {".lp": LPParser, ".mps": MPSParser}
    parser_type = parser_by_extension.get(extension)
    if parser_type is None:
        return _record_error(
            result,
            "PARSER",
            f"Unsupported model extension '{extension or '(none)'}'; expected .lp or .mps.",
            "UNSUPPORTED_FORMAT",
        )

    result.stage = "PARSER"
    try:
        model = parser_type().parse_file(path)
        result.parser_status = "OK"
    except (ModelParseError, OSError, UnicodeError, ValueError) as error:
        return _record_error(
            result,
            "PARSER",
            _safe_error_message(error, path),
            "PARSE_FAILED",
            parser_status="FAILED",
        )
    except Exception as error:
        return _record_error(
            result,
            "PARSER",
            _safe_error_message(error, path),
            "PARSER_ERROR",
            parser_status="FAILED",
        )

    result.stage = "CANONICAL_MODEL"
    try:
        validation = ModelValidator.validate(model)
        if not validation.is_valid:
            return _record_error(
                result,
                "CANONICAL_MODEL",
                "; ".join(validation.errors),
                "MODEL_INVALID",
            )
        result.model_summary = model.summary()
        result.warnings.extend(validation.warnings)
    except Exception as error:
        return _record_error(result, "CANONICAL_MODEL", str(error), "VALIDATION_ERROR")

    result.stage = "FINGERPRINT"
    try:
        result.fingerprint = StructuralFingerprinter().analyze(model)
        if result.fingerprint.status == "INVALID":
            return _record_error(
                result,
                "FINGERPRINT",
                "; ".join(result.fingerprint.raw_metrics.get("validation_errors", [])),
                "FINGERPRINT_INVALID",
            )
    except Exception as error:
        return _record_error(result, "FINGERPRINT", str(error), "FINGERPRINT_ERROR")

    result.stage = "HEALTH"
    try:
        result.health = NumericalHealthMonitor().evaluate(model)
        if result.health.status == "INVALID":
            return _record_error(result, "HEALTH", result.health.details, "HEALTH_INVALID")
        if result.health.status in {"WARNING", "UNAVAILABLE"}:
            result.warnings.append(result.health.details)
    except Exception as error:
        return _record_error(result, "HEALTH", str(error), "HEALTH_ERROR")

    result.stage = "ISG"
    try:
        result.isg = ISGBuilder().build_bipartite_graph(model)
    except Exception as error:
        return _record_error(result, "ISG", str(error), "ISG_ERROR")

    result.stage = "STRATEGY"
    try:
        result.strategy = StrategyRecommender().recommend(
            result.fingerprint,
            model=model,
            health=result.health,
            isg=result.isg,
        )
        result.warnings.extend(result.strategy.warnings)
        if result.strategy.status == "INVALID":
            return _record_error(
                result,
                "STRATEGY",
                "; ".join(result.strategy.warnings) or "Strategy analysis rejected the model.",
                "STRATEGY_INVALID",
            )
    except Exception as error:
        return _record_error(result, "STRATEGY", str(error), "STRATEGY_ERROR")

    result.stage = "SOLVER"
    registry = solver_registry or SolverRegistry()
    try:
        backend = registry.select_for_strategy(result.strategy)
        result.selected_solver = backend.solver_name
        backend.load_model(model)
        backend.apply_strategy(result.strategy)
        result.solver_executed = True
        result.solver_result = backend.solve()
        if not isinstance(result.solver_result, SolveResult):
            return _record_error(
                result,
                "SOLVER",
                "Selected backend returned an unsupported result type.",
                "INVALID_SOLVER_RESULT",
            )
        if result.solver_result.status in {"ERROR", "UNSUPPORTED"}:
            result.errors.append(
                PipelineError(
                    stage="SOLVER",
                    message=(
                        result.solver_result.error_message
                        or result.solver_result.message
                        or "Solver did not produce a solution."
                    ),
                    code=result.solver_result.termination_reason,
                )
            )
    except (SolverUnavailableError, UnsupportedStrategyError) as error:
        return _record_error(result, "SOLVER", str(error), "SOLVER_UNAVAILABLE")
    except Exception as error:
        return _record_error(result, "SOLVER", str(error), "SOLVER_ERROR")

    result.stage = "SAKSHI"
    try:
        active_configuration = _configuration(configuration)
        verifier = SakshiVerifier(
            feasibility_tolerance=active_configuration.feasibility_tolerance,
            integrality_tolerance=active_configuration.integrality_tolerance,
        )
        result.verification = verifier.verify(model, result.solver_result)
        if result.verification.status == "FAILED":
            result.errors.append(
                PipelineError(
                    stage="SAKSHI",
                    message="SAKSHI found one or more failed solution checks.",
                    code="VERIFICATION_FAILED",
                )
            )
        elif result.verification.status in {"INVALID", "UNAVAILABLE"}:
            result.warnings.extend(result.verification.warnings)
    except Exception as error:
        return _record_error(result, "SAKSHI", str(error), "VERIFICATION_ERROR")

    result.stage = "CERTIFICATE"
    try:
        result.certificate = SolveCertificateGenerator().build_certificate(
            model,
            result.solver_result,
            verification=result.verification,
        )
    except Exception as error:
        return _record_error(result, "CERTIFICATE", str(error), "CERTIFICATE_ERROR")

    result.warnings.extend(result.certificate.warnings)
    result.warnings = list(dict.fromkeys(warning for warning in result.warnings if warning))
    solver_success = result.solver_result.status in {"OPTIMAL", "FEASIBLE"}
    result.success = bool(
        solver_success
        and result.verification.status == "VERIFIED"
        and result.certificate is not None
    )
    result.status = "COMPLETE" if result.success else "PARTIAL"
    result.stage = "COMPLETE" if result.success else result.stage
    return result


def _configuration(
    configuration: Optional[Union[PipelineConfiguration, Mapping[str, Any]]],
) -> PipelineConfiguration:
    if configuration is None:
        return PipelineConfiguration()
    if isinstance(configuration, PipelineConfiguration):
        return configuration
    return PipelineConfiguration(
        feasibility_tolerance=float(configuration.get("feasibility_tolerance", 1e-7)),
        integrality_tolerance=float(configuration.get("integrality_tolerance", 1e-6)),
    )


def _record_error(
    result: PipelineResult,
    stage: str,
    message: str,
    code: Optional[str] = None,
    *,
    parser_status: Optional[str] = None,
) -> PipelineResult:
    result.success = False
    result.status = "FAILED"
    result.stage = stage
    if parser_status is not None:
        result.parser_status = parser_status
    result.errors.append(PipelineError(stage=stage, message=message, code=code))
    return result


def _safe_error_message(error: Exception, path: Path) -> str:
    message = str(error).replace(str(path), path.name)
    return message or error.__class__.__name__


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)