"""Phase 8 orchestration and API integration tests."""

import json
import asyncio

from fastapi.testclient import TestClient

from backend.app.main import ModelUploadBodyLimitMiddleware, app
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


def test_existing_lp_fixtures_cover_supported_pipeline_classes():
    from pathlib import Path

    fixture_directory = Path("tests/test_data")
    expected_fixtures = [
        "continuous_simple.lp",
        "equality_model.lp",
        "maximization_bounded.lp",
        "binary_knapsack.lp",
        "mixed_integer.lp",
    ]

    for fixture_name in expected_fixtures:
        result = run_pipeline(fixture_directory / fixture_name)
        assert result.success is True, (fixture_name, result.errors)
        assert result.solver_result.status == "OPTIMAL"
        assert result.verification.status == "VERIFIED"
        assert result.certificate.sha256_hash


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


def test_invalid_bounds_fixture_reports_canonical_model_stage():
    result = run_pipeline("tests/test_data/invalid_bounds.lp")

    assert result.success is False
    assert result.stage == "CANONICAL_MODEL"
    assert result.parser_status == "OK"
    assert result.errors[0].code == "MODEL_INVALID"
    assert "inconsistent bounds" in result.errors[0].message
    assert "x1" in result.errors[0].message


def test_infeasible_model_returns_solver_status_and_unavailable_verification(tmp_path):
    model_path = tmp_path / "infeasible.lp"
    model_path.write_text(
        "Minimize\n obj: x\nSubject To\n lower: x >= 2\n upper: x <= 1\nEnd\n",
        encoding="utf-8",
    )

    result = run_pipeline(model_path)

    assert result.success is False
    assert result.status == "PARTIAL"
    assert result.solver_executed is True
    assert result.solver_result.status == "INFEASIBLE"
    assert result.verification.status == "UNAVAILABLE"
    assert result.certificate.verification_status == "UNAVAILABLE"


def test_certificate_generation_failure_is_structured(tmp_path, monkeypatch):
    model_path = tmp_path / "certificate-failure.lp"
    model_path.write_text(VALID_LP, encoding="utf-8")

    def fail_certificate(*args, **kwargs):
        raise RuntimeError("certificate storage failed")

    monkeypatch.setattr(
        "saksham.pipeline.runner.SolveCertificateGenerator.build_certificate",
        fail_certificate,
    )
    result = run_pipeline(model_path)

    assert result.success is False
    assert result.stage == "CERTIFICATE"
    assert result.solver_result.status == "OPTIMAL"
    assert result.verification.status == "VERIFIED"
    assert result.errors[0].code == "CERTIFICATE_ERROR"
    assert "certificate storage failed" in result.errors[0].message


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
    assert "saksham-upload-" not in response.text


def test_solve_upload_endpoint_handles_missing_file_without_traceback():
    client = TestClient(app)

    response = client.post("/api/v1/solve")

    assert response.status_code == 422
    assert "Traceback" not in response.text


def test_solve_upload_endpoint_reports_invalid_bounds_at_model_stage():
    client = TestClient(app)

    response = client.post(
        "/api/v1/solve",
        files={
            "file": (
                "invalid_bounds.lp",
                "Minimize\n obj: 2 x1 + 3 x2\nSubject To\n c1: x1 + x2 <= 10\nBounds\n 5 <= x1 <= 2\n 0 <= x2 <= 10\nEnd\n",
                "text/plain",
            )
        },
    )

    assert response.status_code == 422
    body = response.json()
    assert body["errors"][0]["stage"] == "CANONICAL_MODEL"
    assert body["errors"][0]["code"] == "MODEL_INVALID"
    assert "saksham-upload-" not in response.text


def test_health_response_identifies_registered_and_non_solver_components():
    health = TestClient(app).get("/api/v1/health").json()

    assert set(health["solvers"]) == {"highs", "scipy"}
    assert health["solvers"]["highs"]["available"] is True
    assert health["solvers"]["scipy"]["solver_backend"] is False
    assert health["solvers"]["scipy"]["role"] == "numerical_library"


def test_upload_body_limit_rejects_content_length_before_route_parsing():
    route_called = False
    sent = []

    async def inner_app(scope, receive, send):
        nonlocal route_called
        route_called = True

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/solve",
        "headers": [(b"content-length", b"100")],
    }
    middleware = ModelUploadBodyLimitMiddleware(inner_app, max_bytes=10)

    asyncio.run(middleware(scope, receive, send))

    assert route_called is False
    assert sent[0]["status"] == 413
    assert b"UPLOAD_TOO_LARGE" in sent[1]["body"]


def test_upload_body_limit_rejects_chunked_body_over_limit():
    received = False
    sent = []

    async def inner_app(scope, receive, send):
        message = await receive()
        while message.get("more_body"):
            message = await receive()

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": b"123456", "more_body": True}
        return {"type": "http.request", "body": b"789012", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/solve",
        "headers": [(b"transfer-encoding", b"chunked")],
    }
    middleware = ModelUploadBodyLimitMiddleware(inner_app, max_bytes=10)

    asyncio.run(middleware(scope, receive, send))

    assert sent[0]["status"] == 413
    assert b"UPLOAD_TOO_LARGE" in sent[1]["body"]


def test_solve_endpoint_rejects_oversized_declared_request_before_parsing():
    response = TestClient(app).post(
        "/api/v1/solve",
        headers={"content-length": str(32 * 1024 * 1024 + 65_537)},
    )

    assert response.status_code == 413
    assert response.json()["errors"][0]["code"] == "UPLOAD_TOO_LARGE"


def test_health_marks_scipy_as_numerical_library_not_solver():
    response = TestClient(app).get("/api/v1/health")

    assert response.status_code == 200
    health = response.json()
    assert health["solvers"]["highs"]["available"] is True
    assert health["solvers"]["scipy"]["role"] == "numerical_library"
    assert health["solvers"]["scipy"]["solver_backend"] is False


def test_cors_allows_configured_frontend_and_rejects_unconfigured_origin():
    client = TestClient(app)
    allowed = client.options(
        "/api/v1/solve",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    denied = client.options(
        "/api/v1/solve",
        headers={
            "Origin": "https://external.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert allowed.headers["access-control-allow-credentials"] == "true"
    assert denied.status_code == 400
    assert "access-control-allow-origin" not in denied.headers