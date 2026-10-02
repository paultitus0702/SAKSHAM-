"""Phase 8 orchestration and API integration tests."""

import json

from fastapi.testclient import TestClient

from backend.app.main import app
from saksham.pipeline import run_pipeline


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

VALID_MPS = """\
NAME          PHASE8
ROWS
 N  COST
 G  DEMAND
COLUMNS
    X         COST 1   DEMAND 1
RHS
    RHS1      DEMAND 2
BOUNDS
 UP BND1      X 3
ENDATA
"""


def test_valid_supported_lp_runs_every_pipeline_stage(tmp_path):
    model_path = tmp_path / "small.lp"
    model_path.write_text(VALID_LP, encoding="utf-8")

    result = run_pipeline(model_path)

    assert result.success is True
    assert result.status == "COMPLETE"
    assert result.parser_status == "OK"
    assert result.model_summary["num_variables"] == 2
    assert result.fingerprint.status == "OK"
    assert result.health.status == "HEALTHY"
    assert result.isg.status == "OK"
    assert result.strategy.primary_strategy == "DIRECT_LP"
    assert result.selected_solver == "highs"
    assert result.solver_executed is True
    assert result.solver_result.objective_value == 4.0
    assert result.solver_result.primal_solution is not None
    assert result.verification.status == "VERIFIED"
    assert result.certificate.verification_status == "VERIFIED"

    response = result.to_dict()
    assert response["input"]["name"] == "small.lp"
    assert "solver_result" in response
    assert json.loads(json.dumps(response))["certificate"]["optimality_claim"] == "NOT_VERIFIED"
    assert "tmp_path" not in json.dumps(response)


def test_valid_mps_runs_complete_pipeline(tmp_path):
    model_path = tmp_path / "small.mps"
    model_path.write_text(VALID_MPS, encoding="utf-8")

    result = run_pipeline(model_path)

    assert result.success is True
    assert result.input_format == "mps"
    assert result.solver_result.objective_value == 2.0
    assert result.verification.status == "VERIFIED"
    assert result.certificate.verification_status == "VERIFIED"


def test_invalid_lp_returns_structured_parser_failure_without_solver(tmp_path):
    class SpyRegistry:
        called = False

        def select_for_strategy(self, strategy):
            self.called = True
            raise AssertionError("Solver selection must not happen after parser failure")

    model_path = tmp_path / "invalid.lp"
    model_path.write_text("Minimize\n obj: x\nSubject To\n c1: x ??? 2\nEnd\n", encoding="utf-8")
    registry = SpyRegistry()

    result = run_pipeline(model_path, solver_registry=registry)

    assert result.success is False
    assert result.errors[0].stage == "PARSER"
    assert result.parser_status == "FAILED"
    assert registry.called is False
    assert result.solver_executed is False


def test_unsupported_extension_returns_structured_error(tmp_path):
    model_path = tmp_path / "model.txt"
    model_path.write_text("not a model", encoding="utf-8")

    result = run_pipeline(model_path)

    assert result.success is False
    assert result.errors[0].stage == "PARSER"
    assert result.errors[0].code == "UNSUPPORTED_FORMAT"


def test_missing_file_returns_structured_input_error(tmp_path):
    result = run_pipeline(tmp_path / "missing.lp")

    assert result.success is False
    assert result.errors[0].stage == "INPUT"
    assert result.errors[0].code == "FILE_NOT_FOUND"


def test_repeat_runs_have_consistent_analysis_and_certificate(tmp_path):
    model_path = tmp_path / "repeat.lp"
    model_path.write_text(VALID_LP, encoding="utf-8")

    first = run_pipeline(model_path)
    second = run_pipeline(model_path)

    assert first.fingerprint == second.fingerprint
    assert first.health == second.health
    assert first.isg.summary == second.isg.summary
    assert first.strategy == second.strategy
    assert first.solver_result.status == second.solver_result.status
    assert first.certificate.sha256_hash == second.certificate.sha256_hash


def test_solve_upload_endpoint_processes_valid_lp():
    client = TestClient(app)

    response = client.post(
        "/api/v1/solve",
        files={"file": ("upload.lp", VALID_LP, "text/plain")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["input"]["name"] == "upload.lp"
    assert body["selected_solver"] == "highs"
    assert body["solver_executed"] is True
    assert body["verification"]["status"] == "VERIFIED"
    assert body["certificate"]["sha256_hash"]
    assert "saksham-upload-" not in response.text


def test_solve_upload_endpoint_returns_structured_parse_error():
    client = TestClient(app)

    response = client.post(
        "/api/v1/solve",
        files={
            "file": (
                "bad.lp",
                "Minimize\n obj: x\nSubject To\n c1: x ??? 2\nEnd\n",
                "text/plain",
            )
        },
    )

    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["errors"][0]["stage"] == "PARSER"