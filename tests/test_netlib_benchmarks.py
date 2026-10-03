"""Phase 11 Netlib orchestration tests; no network data is required."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from saksham.benchmarks import (
    AUTHENTIC_NETLIB,
    SYNTHETIC_TEST_FIXTURE,
    UNCLASSIFIED_INPUT,
    NetlibBenchmarkMetadata,
    NetlibRunner,
    load_benchmark_metadata,
)
from saksham.pipeline import run_pipeline as existing_run_pipeline


VALID_LP = """\
Minimize
 obj: x + 2 y
Subject To
 demand: x + y >= 4
Bounds
 0 <= x <= 4
 0 <= y
End
"""

INFEASIBLE_LP = """\
Minimize
 obj: x
Subject To
 lower: x >= 2
 upper: x <= 1
End
"""


def _fixture_metadata(name: str, filename: str) -> NetlibBenchmarkMetadata:
    return NetlibBenchmarkMetadata(
        name=name,
        filename=filename,
        source_type=SYNTHETIC_TEST_FIXTURE,
        source_identifier=f"tests/test_data/{filename}",
        notes="Local deterministic test fixture; not authentic Netlib data.",
    )


def _write_model(directory: Path, name: str, content: str) -> Path:
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


def test_default_metadata_loads_three_authentic_netlib_candidates():
    metadata = load_benchmark_metadata()

    assert [entry.name for entry in metadata] == ["afiro", "adlittle", "blend"]
    assert all(entry.source_type == AUTHENTIC_NETLIB for entry in metadata)
    assert all(entry.source_identifier.startswith("https://www.netlib.org/") for entry in metadata)
    assert all(entry.expected_category is None for entry in metadata)


def test_metadata_loader_rejects_duplicate_names(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmarks": [
                    {"name": "same", "filename": "one.lp", "source_type": AUTHENTIC_NETLIB},
                    {"name": "SAME", "filename": "two.lp", "source_type": AUTHENTIC_NETLIB},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate Netlib benchmark"):
        load_benchmark_metadata(metadata_path)


def test_discovery_finds_supported_files_and_marks_unknown_provenance(tmp_path):
    _write_model(tmp_path, "afiro.lp", VALID_LP)
    _write_model(tmp_path, "local_case.mps", "NAME TEST\n")
    _write_model(tmp_path, "notes.txt", "not an optimization model")
    runner = NetlibRunner(
        tmp_path,
        metadata=(
            NetlibBenchmarkMetadata("afiro", "afiro.mps", AUTHENTIC_NETLIB),
        ),
    )

    discovered = runner.discover_benchmarks()

    assert [(entry.name, entry.filename) for entry in discovered] == [
        ("afiro", "afiro.lp"),
        ("local_case", "local_case.mps"),
    ]
    assert discovered[0].source_type == AUTHENTIC_NETLIB
    assert discovered[1].source_type == UNCLASSIFIED_INPUT


def test_one_local_benchmark_runs_through_existing_pipeline(tmp_path):
    _write_model(tmp_path, "small.lp", VALID_LP)
    runner = NetlibRunner(tmp_path, metadata=(_fixture_metadata("small", "small.lp"),))

    result = runner.run_benchmark("small")

    assert result["source_type"] == SYNTHETIC_TEST_FIXTURE
    assert result["parser_status"] == "OK"
    assert result["model_dimensions"] == {"variables": 2, "constraints": 1}
    assert result["nonzeros"] == 2
    assert result["density"] == 1.0
    assert result["numerical_health_status"] == "HEALTHY"
    assert result["solver_status"] == "OPTIMAL"
    assert result["sakshi_verification_status"] == "VERIFIED"
    assert result["certificate_status"] == "GENERATED"
    assert result["certificate_digest"]
    assert result["failure_stage"] is None
    assert result["success"] is True


def test_selected_benchmarks_continue_after_a_parser_failure(tmp_path):
    _write_model(tmp_path, "good.lp", VALID_LP)
    _write_model(tmp_path, "bad.lp", "Minimize\n obj: x\nSubject To\n c: x ??? 2\nEnd\n")
    runner = NetlibRunner(
        tmp_path,
        metadata=(
            _fixture_metadata("good", "good.lp"),
            _fixture_metadata("bad", "bad.lp"),
        ),
    )

    results = runner.run_benchmarks(["bad", "good"])

    assert list(results) == ["bad", "good"]
    assert results["bad"]["failure_stage"] == "PARSER"
    assert results["bad"]["parser_status"] == "FAILED"
    assert results["good"]["success"] is True


def test_missing_configured_authentic_benchmark_is_unavailable_not_fabricated(tmp_path):
    runner = NetlibRunner(tmp_path)

    result = runner.run_benchmark("afiro")

    assert result["source_type"] == AUTHENTIC_NETLIB
    assert result["failure_stage"] == "INPUT"
    assert result["failure_code"] == "INPUT_NOT_FOUND"
    assert result["parser_status"] == "NOT_STARTED"
    assert result["objective_value"] is None
    assert result["success"] is False


def test_run_all_reports_each_configured_missing_benchmark(tmp_path):
    results = NetlibRunner(tmp_path).run_all()

    assert list(results) == ["afiro", "adlittle", "blend"]
    assert all(result["failure_code"] == "INPUT_NOT_FOUND" for result in results.values())
    assert all(result["source_type"] == AUTHENTIC_NETLIB for result in results.values())


def test_unsupported_extension_is_reported_without_calling_parser(tmp_path):
    _write_model(tmp_path, "not_lp.txt", "not a model")
    metadata = NetlibBenchmarkMetadata(
        "not_lp", "not_lp.txt", SYNTHETIC_TEST_FIXTURE
    )

    result = NetlibRunner(tmp_path, metadata=(metadata,)).run_benchmark("not_lp")

    assert result["failure_stage"] == "PARSER"
    assert result["failure_code"] == "UNSUPPORTED_FORMAT"
    assert result["parser_status"] == "FAILED"


def test_infeasible_solver_result_and_unavailable_verification_are_propagated(tmp_path):
    _write_model(tmp_path, "infeasible.lp", INFEASIBLE_LP)
    runner = NetlibRunner(
        tmp_path,
        metadata=(_fixture_metadata("infeasible", "infeasible.lp"),),
    )

    result = runner.run_benchmark("infeasible")

    assert result["solver_status"] == "INFEASIBLE"
    assert result["failure_stage"] == "SOLVER"
    assert result["sakshi_verification_status"] == "UNAVAILABLE"
    assert result["certificate_status"] == "GENERATED"
    assert result["certificate_digest"]
    assert result["success"] is False


def test_certificate_failure_is_reported_from_pipeline(tmp_path, monkeypatch):
    _write_model(tmp_path, "small.lp", VALID_LP)

    def fail_certificate(*args, **kwargs):
        raise RuntimeError("certificate unavailable")

    monkeypatch.setattr(
        "saksham.pipeline.runner.SolveCertificateGenerator.build_certificate",
        fail_certificate,
    )
    result = NetlibRunner(
        tmp_path,
        metadata=(_fixture_metadata("small", "small.lp"),),
    ).run_benchmark("small")

    assert result["failure_stage"] == "CERTIFICATE"
    assert result["failure_code"] == "CERTIFICATE_ERROR"
    assert result["certificate_status"] == "FAILED"
    assert result["certificate_digest"] is None


def test_result_structure_is_deterministic_and_distinguishes_sources(tmp_path):
    _write_model(tmp_path, "fixture.lp", VALID_LP)
    runner = NetlibRunner(
        tmp_path,
        metadata=(_fixture_metadata("fixture", "fixture.lp"),),
    )

    first = runner.run_benchmark("fixture")
    second = runner.run_benchmark("fixture")

    assert first.keys() == second.keys()
    assert first["source_type"] == SYNTHETIC_TEST_FIXTURE
    assert first["certificate_digest"] == second["certificate_digest"]
    assert first["objective_value"] == second["objective_value"]
    assert first["solve_time_seconds"] is not None
    assert second["solve_time_seconds"] is not None


def test_runner_delegates_pipeline_work_instead_of_duplicating_it(tmp_path, monkeypatch):
    _write_model(tmp_path, "fixture.lp", VALID_LP)
    calls = []

    def tracked_pipeline(path, **kwargs):
        calls.append((Path(path).name, kwargs.get("input_name")))
        return existing_run_pipeline(path, **kwargs)

    monkeypatch.setattr("saksham.benchmarks.netlib_runner.run_pipeline", tracked_pipeline)
    runner = NetlibRunner(
        tmp_path,
        metadata=(_fixture_metadata("fixture", "fixture.lp"),),
    )

    result = runner.run_benchmark("fixture")

    assert result["success"] is True
    assert calls == [("fixture.lp", "fixture")]


def test_unknown_name_returns_structured_failure_and_suite_keeps_running(tmp_path):
    _write_model(tmp_path, "good.lp", VALID_LP)
    runner = NetlibRunner(tmp_path, metadata=(_fixture_metadata("good", "good.lp"),))

    results = runner.run_benchmarks(["missing", "good"])

    assert results["missing"]["failure_code"] == "BENCHMARK_NOT_CONFIGURED"
    assert results["good"]["success"] is True


def test_backend_metadata_reports_runner_ready_but_files_required():
    response = TestClient(app).get("/api/v1/system/info")

    assert response.status_code == 200
    assert (
        response.json()["modules"]["netlib_benchmarks"]
        == "IMPLEMENTED_PHASE_11_FILES_REQUIRED"
    )