"""
Root API Router for SAKSHAM Backend.
Provides health/system information and the synchronous model-solve pipeline.
"""
from saksham.benchmarks.netlib_runner import NetlibRunner
from pathlib import Path
import tempfile
from typing import Any, Dict

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import highspy
import scipy
import numpy as np
import networkx as nx
from saksham import __version__ as engine_version
from saksham.pipeline import PipelineConfiguration, PipelineError, PipelineResult, run_pipeline
from ..config import settings

api_router = APIRouter(prefix="/api/v1")
_MAX_MODEL_UPLOAD_BYTES = 32 * 1024 * 1024


@api_router.get("/health")
def get_health() -> Dict[str, Any]:
    """
    SAKSHAM Health Check Endpoint.
    Verifies that the API, engine, and numerical backend (HiGHS) are operational.
    """
    # Verify HiGHS can initialize
    highs = highspy.Highs()
    highs_version = highs.version()
    
    return {
        "status": "healthy",
        "app_name": settings.system.app_name,
        "environment": settings.system.environment,
        "engine": {
            "name": "saksham-engine",
            "version": engine_version,
            "status": "OPERATIONAL"
        },
        "solvers": {
            "highs": {
                "available": True,
                "version": highs_version,
                "status": "READY"
            },
            "scipy": {
                "available": True,
                "version": scipy.__version__,
                "status": "READY",
                "role": "numerical_library",
                "solver_backend": False,
            }
        },
        "scaffolding_phase": "Phase 8: End-to-End Solve Pipeline Available"
    }


@api_router.get("/system/info")
def get_system_info() -> Dict[str, Any]:
    """System information and module availability."""
    return {
        "app_name": settings.system.app_name,
        "version": settings.system.version,
        "modules": {
            "canonical_model": "IMPLEMENTED",
            "parser": "IMPLEMENTED",
            "fingerprint": "IMPLEMENTED",
            "isg": "IMPLEMENTED",
            "strategy": "IMPLEMENTED",
            "solver_backend": "IMPLEMENTED",
            "sakshi_verifier": "IMPLEMENTED",
            "solve_certificate": "IMPLEMENTED",
            "refinery_scenario": "IMPLEMENTED_PHASE_10",
            "refinery_benchmarks": "IMPLEMENTED_PHASE_10",
            "netlib_benchmarks": "IMPLEMENTED_PHASE_11_FILES_REQUIRED"
        }
    }


@api_router.post("/solve")
def solve_uploaded_model(file: UploadFile = File(...)) -> JSONResponse:
    """Run an uploaded LP/MPS model without persisting its client filename."""
    raw_name = (file.filename or "model").replace("\\", "/").split("/")[-1]
    display_name = raw_name or "model"
    extension = Path(display_name).suffix.lower()

    with tempfile.TemporaryDirectory(prefix="saksham-upload-") as temporary_directory:
        temporary_path = Path(temporary_directory) / f"model{extension}"
        total_bytes = 0
        try:
            with temporary_path.open("wb") as destination:
                while chunk := file.file.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > _MAX_MODEL_UPLOAD_BYTES:
                        failure = PipelineResult(
                            status="FAILED",
                            input_name=display_name,
                            input_format=extension.removeprefix("."),
                            stage="INPUT",
                            errors=[
                                PipelineError(
                                    stage="INPUT",
                                    code="UPLOAD_TOO_LARGE",
                                    message="Uploaded model exceeds the 32 MiB limit.",
                                )
                            ],
                        )
                        return JSONResponse(
                            status_code=413,
                            content=failure.to_dict(),
                        )
                    destination.write(chunk)

            configuration = PipelineConfiguration(
                feasibility_tolerance=settings.engine.feasibility_tolerance,
                integrality_tolerance=settings.engine.integrality_tolerance,
            )
            result = run_pipeline(
                temporary_path,
                configuration=configuration,
                input_name=display_name,
            )
            status_code = 200 if result.model_summary is not None else 422
            return JSONResponse(status_code=status_code, content=result.to_dict())
        except OSError:
            failure = PipelineResult(
                status="FAILED",
                input_name=display_name,
                input_format=extension.removeprefix("."),
                stage="INPUT",
                errors=[
                    PipelineError(
                        stage="INPUT",
                        code="UPLOAD_FAILED",
                        message="Uploaded model could not be staged for processing.",
                    )
                ],
            )
            return JSONResponse(status_code=500, content=failure.to_dict())
        finally:
            file.file.close()
            @api_router.get("/benchmarks/netlib")
def get_netlib_benchmarks() -> Dict[str, Any]:
    """Return configured authentic Netlib benchmark metadata."""
    runner = NetlibRunner()

    return {
        "source_type": "AUTHENTIC_NETLIB",
        "count": len(runner.metadata),
        "benchmarks": [
            {
                "name": benchmark.name,
                "filename": benchmark.filename,
                "source_type": benchmark.source_type,
                "source_identifier": benchmark.source_identifier,
                "expected_category": benchmark.expected_category,
                "notes": benchmark.notes,
            }
            for benchmark in runner.metadata
        ],
    }


@api_router.get("/benchmarks/netlib/{benchmark_name}")
def run_netlib_benchmark(benchmark_name: str) -> Dict[str, Any]:
    """Execute one authentic Netlib benchmark through the shared pipeline."""
    runner = NetlibRunner()
    return runner.run_benchmark(benchmark_name)
