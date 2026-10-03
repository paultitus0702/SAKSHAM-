"""Metadata-driven Netlib benchmark execution through the shared pipeline."""
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..pipeline import PipelineResult, run_pipeline


AUTHENTIC_NETLIB = "AUTHENTIC_NETLIB"
SYNTHETIC_TEST_FIXTURE = "SYNTHETIC_TEST_FIXTURE"
UNCLASSIFIED_INPUT = "UNCLASSIFIED_INPUT"
_SOURCE_TYPES = {AUTHENTIC_NETLIB, SYNTHETIC_TEST_FIXTURE, UNCLASSIFIED_INPUT}
_SUPPORTED_EXTENSIONS = {".lp", ".mps"}
_DEFAULT_METADATA_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "netlib" / "benchmarks.json"
)
_DEFAULT_BENCHMARK_DIRECTORY = _DEFAULT_METADATA_PATH.parent


@dataclass(frozen=True)
class NetlibBenchmarkMetadata:
    """Stable identity and provenance declaration for one benchmark input."""

    name: str
    filename: str
    source_type: str
    source_identifier: Optional[str] = None
    expected_category: Optional[str] = None
    notes: Optional[str] = None


@dataclass(frozen=True)
class NetlibBenchmarkResult:
    """Measured output from one input file passed through SAKSHAM."""

    benchmark_name: str
    source_type: str
    source_identifier: Optional[str]
    input_path: Optional[str]
    parser_status: str
    model_dimensions: Optional[Dict[str, int]]
    nonzeros: Optional[int]
    density: Optional[float]
    numerical_health_status: str
    recommended_strategy: Optional[Dict[str, Any]]
    solver_name: Optional[str]
    solver_status: str
    objective_value: Optional[float]
    solve_time_seconds: Optional[float]
    sakshi_verification_status: str
    certificate_status: str
    certificate_digest: Optional[str]
    failure_stage: Optional[str]
    failure_code: Optional[str]
    failure_message: Optional[str]
    warnings: tuple[str, ...]
    success: bool

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible result while preserving stable field order."""
        result = asdict(self)
        result["warnings"] = list(self.warnings)
        return result


def load_benchmark_metadata(
    metadata_path: Optional[Path] = None,
) -> tuple[NetlibBenchmarkMetadata, ...]:
    """Load and validate the deterministic benchmark registry JSON."""
    path = Path(metadata_path) if metadata_path is not None else _DEFAULT_METADATA_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load Netlib benchmark metadata: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Netlib metadata must be an object with schema_version 1.")
    entries = payload.get("benchmarks")
    if not isinstance(entries, list):
        raise ValueError("Netlib metadata 'benchmarks' must be a list.")

    metadata: List[NetlibBenchmarkMetadata] = []
    seen_names: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Netlib metadata entry {index} must be an object.")
        name = entry.get("name")
        filename = entry.get("filename")
        source_type = entry.get("source_type")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Netlib metadata entry {index} has an invalid name.")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(f"Netlib metadata entry '{name}' has an invalid filename.")
        if Path(filename).name != filename or filename in {".", ".."}:
            raise ValueError(f"Netlib metadata entry '{name}' filename must be a basename.")
        if not isinstance(source_type, str) or source_type not in _SOURCE_TYPES:
            raise ValueError(f"Netlib metadata entry '{name}' has an invalid source_type.")
        normalized_name = name.strip().casefold()
        if normalized_name in seen_names:
            raise ValueError(f"Duplicate Netlib benchmark name '{name}'.")
        seen_names.add(normalized_name)
        metadata.append(
            NetlibBenchmarkMetadata(
                name=name.strip(),
                filename=filename,
                source_type=source_type,
                source_identifier=_optional_string(entry.get("source_identifier")),
                expected_category=_optional_string(entry.get("expected_category")),
                notes=_optional_string(entry.get("notes")),
            )
        )
    return tuple(metadata)


class NetlibRunner:
    """Discover files and run configured entries through ``run_pipeline``."""

    def __init__(
        self,
        benchmark_directory: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        *,
        metadata: Optional[Iterable[NetlibBenchmarkMetadata]] = None,
    ) -> None:
        self.benchmark_directory = (
            Path(benchmark_directory)
            if benchmark_directory is not None
            else _DEFAULT_BENCHMARK_DIRECTORY
        )
        entries = tuple(metadata) if metadata is not None else load_benchmark_metadata(metadata_path)
        self._metadata: Dict[str, NetlibBenchmarkMetadata] = {}
        for entry in entries:
            _validate_metadata(entry)
            key = entry.name.casefold()
            if key in self._metadata:
                raise ValueError(f"Duplicate Netlib benchmark name '{entry.name}'.")
            self._metadata[key] = entry

    @property
    def metadata(self) -> tuple[NetlibBenchmarkMetadata, ...]:
        """Return configured metadata in deterministic order."""
        return tuple(self._metadata.values())

    def discover_benchmarks(self) -> tuple[NetlibBenchmarkMetadata, ...]:
        """Discover top-level LP/MPS files and associate known metadata."""
        if not self.benchmark_directory.is_dir():
            return ()
        discovered: List[NetlibBenchmarkMetadata] = []
        for path in sorted(
            self.benchmark_directory.iterdir(), key=lambda candidate: candidate.name.casefold()
        ):
            if not path.is_file() or path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
                continue
            configured = self._metadata.get(path.stem.casefold())
            if configured is None:
                discovered.append(
                    NetlibBenchmarkMetadata(
                        name=path.stem,
                        filename=path.name,
                        source_type=UNCLASSIFIED_INPUT,
                        notes="Local model file discovered; provenance was not verified.",
                    )
                )
            else:
                discovered.append(replace(configured, filename=path.name))
        return tuple(discovered)

    def run_benchmark(self, model_name: str) -> Dict[str, Any]:
        """Run one configured or discovered benchmark and return its report."""
        if not isinstance(model_name, str) or not model_name.strip():
            return _input_failure(
                str(model_name),
                UNCLASSIFIED_INPUT,
                "BENCHMARK_NOT_CONFIGURED",
                "A non-empty benchmark name is required.",
            ).to_dict()
        key = model_name.strip().casefold()
        metadata = self._metadata.get(key)
        if metadata is None:
            metadata = next(
                (entry for entry in self.discover_benchmarks() if entry.name.casefold() == key),
                None,
            )
        if metadata is None:
            return _input_failure(
                model_name,
                UNCLASSIFIED_INPUT,
                "BENCHMARK_NOT_CONFIGURED",
                f"Benchmark '{model_name}' is not configured or discoverable.",
            ).to_dict()

        path, resolution_error = self._resolve_path(metadata)
        if resolution_error is not None:
            code, message = resolution_error
            return _input_failure(
                metadata.name,
                metadata.source_type,
                code,
                message,
                source_identifier=metadata.source_identifier,
                input_path=str(self.benchmark_directory / metadata.filename),
            ).to_dict()
        assert path is not None
        relative_path = _display_path(path, self.benchmark_directory)
        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            return _input_failure(
                metadata.name,
                metadata.source_type,
                "UNSUPPORTED_FORMAT",
                f"Unsupported benchmark extension '{path.suffix or '(none)'}'; expected .lp or .mps.",
                source_identifier=metadata.source_identifier,
                input_path=relative_path,
                failure_stage="PARSER",
                parser_status="FAILED",
            ).to_dict()

        result = run_pipeline(path, input_name=metadata.name)
        return _pipeline_report(metadata, relative_path, result).to_dict()

    def run_benchmarks(
        self,
        names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Run selected entries, or every configured entry, without fail-fast."""
        selected_names = (
            [metadata.name for metadata in self.metadata]
            if names is None
            else list(names)
        )
        return {name: self.run_benchmark(name) for name in selected_names}

    def run_all(self) -> Dict[str, Dict[str, Any]]:
        """Run the entire configured validation set."""
        return self.run_benchmarks()

    def _resolve_path(
        self,
        metadata: NetlibBenchmarkMetadata,
    ) -> tuple[Optional[Path], Optional[tuple[str, str]]]:
        exact_path = self.benchmark_directory / metadata.filename
        if exact_path.is_file():
            return exact_path, None

        same_stem = [
            path
            for path in self.benchmark_directory.glob(f"{Path(metadata.filename).stem}.*")
            if path.is_file()
        ] if self.benchmark_directory.is_dir() else []
        supported_alternatives = [
            path for path in same_stem if path.suffix.lower() in _SUPPORTED_EXTENSIONS
        ]
        if len(supported_alternatives) == 1:
            return supported_alternatives[0], None
        if len(supported_alternatives) > 1:
            return None, (
                "AMBIGUOUS_INPUT",
                f"Multiple LP/MPS files match benchmark '{metadata.name}'.",
            )
        return None, (
            "INPUT_NOT_FOUND",
            f"Benchmark file '{metadata.filename}' was not found in '{self.benchmark_directory}'.",
        )


def _pipeline_report(
    metadata: NetlibBenchmarkMetadata,
    input_path: str,
    result: PipelineResult,
) -> NetlibBenchmarkResult:
    summary = result.model_summary
    solver_result = result.solver_result
    error = result.errors[0] if result.errors else None
    failure_stage = error.stage if error is not None else None
    failure_code = error.code if error is not None else None
    failure_message = error.message if error is not None else None

    if not result.success:
        if solver_result is not None and solver_result.status not in {"OPTIMAL", "FEASIBLE"}:
            failure_stage = failure_stage or "SOLVER"
            failure_code = failure_code or solver_result.termination_reason or solver_result.status
            failure_message = failure_message or solver_result.message or solver_result.error_message
        elif result.verification is not None and result.verification.status in {"FAILED", "INVALID"}:
            failure_stage = failure_stage or "SAKSHI"
            failure_message = failure_message or "; ".join(result.verification.warnings)
        elif result.certificate is None:
            failure_stage = failure_stage or "CERTIFICATE"
        elif failure_stage is None:
            failure_stage = result.stage

    certificate_status = (
        "GENERATED"
        if result.certificate is not None
        else "FAILED"
        if failure_stage == "CERTIFICATE"
        else "UNAVAILABLE"
    )
    return NetlibBenchmarkResult(
        benchmark_name=metadata.name,
        source_type=metadata.source_type,
        source_identifier=metadata.source_identifier,
        input_path=input_path,
        parser_status=result.parser_status,
        model_dimensions=(
            {
                "variables": int(summary["num_variables"]),
                "constraints": int(summary["num_constraints"]),
            }
            if summary is not None
            else None
        ),
        nonzeros=int(summary["num_nonzeros"]) if summary is not None else None,
        density=float(summary["density"]) if summary is not None else None,
        numerical_health_status=result.health.status if result.health is not None else "UNAVAILABLE",
        recommended_strategy=(
            {
                "primary_strategy": result.strategy.primary_strategy,
                "primary_algorithm": result.strategy.primary_algorithm,
                "status": result.strategy.status,
                "confidence": result.strategy.confidence,
            }
            if result.strategy is not None
            else None
        ),
        solver_name=(
            solver_result.solver_name
            if solver_result is not None
            else result.selected_solver
        ),
        solver_status=solver_result.status if solver_result is not None else "UNAVAILABLE",
        objective_value=solver_result.objective_value if solver_result is not None else None,
        solve_time_seconds=(
            solver_result.wall_time_seconds if solver_result is not None else None
        ),
        sakshi_verification_status=(
            result.verification.status if result.verification is not None else "UNAVAILABLE"
        ),
        certificate_status=certificate_status,
        certificate_digest=(
            result.certificate.sha256_hash if result.certificate is not None else None
        ),
        failure_stage=failure_stage,
        failure_code=failure_code,
        failure_message=failure_message,
        warnings=tuple(result.warnings),
        success=result.success,
    )


def _input_failure(
    name: str,
    source_type: str,
    code: str,
    message: str,
    *,
    source_identifier: Optional[str] = None,
    input_path: Optional[str] = None,
    failure_stage: str = "INPUT",
    parser_status: str = "NOT_STARTED",
) -> NetlibBenchmarkResult:
    return NetlibBenchmarkResult(
        benchmark_name=name,
        source_type=source_type,
        source_identifier=source_identifier,
        input_path=input_path,
        parser_status=parser_status,
        model_dimensions=None,
        nonzeros=None,
        density=None,
        numerical_health_status="UNAVAILABLE",
        recommended_strategy=None,
        solver_name=None,
        solver_status="UNAVAILABLE",
        objective_value=None,
        solve_time_seconds=None,
        sakshi_verification_status="UNAVAILABLE",
        certificate_status="UNAVAILABLE",
        certificate_digest=None,
        failure_stage=failure_stage,
        failure_code=code,
        failure_message=message,
        warnings=(),
        success=False,
    )


def _validate_metadata(metadata: NetlibBenchmarkMetadata) -> None:
    if not isinstance(metadata, NetlibBenchmarkMetadata):
        raise TypeError("metadata entries must be NetlibBenchmarkMetadata instances.")
    if not metadata.name.strip() or not metadata.filename.strip():
        raise ValueError("Benchmark metadata name and filename must be non-empty.")
    if Path(metadata.filename).name != metadata.filename:
        raise ValueError("Benchmark metadata filename must be a basename.")
    if metadata.source_type not in _SOURCE_TYPES:
        raise ValueError(f"Unsupported benchmark source type '{metadata.source_type}'.")


def _optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Optional Netlib metadata values must be strings or null.")
    return value


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()