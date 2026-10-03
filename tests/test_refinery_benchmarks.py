"""Phase 10 synthetic refinery generation and benchmark reporting tests."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from saksham.benchmarks import (
    RefineryBenchmarkRunner,
    generate_refinery_benchmarks,
)
from saksham.canonical.validator import ModelValidator
from saksham.fingerprint.analyzer import StructuralFingerprinter
from saksham.health.conditioner import NumericalHealthMonitor
from saksham.isg.graph_builder import ISGBuilder
from saksham.scenarios.refinery import generate_refinery_scenario
from saksham.solvers.registry import SolverRegistry
from saksham.strategy.recommender import StrategyRecommender


def test_refinery_generation_is_reproducible_for_same_seed():
    first = generate_refinery_scenario(2, seed=31, binary_operations=True)
    second = generate_refinery_scenario(2, seed=31, binary_operations=True)

    assert first.name == second.name
    assert first.var_names == second.var_names
    assert first.constr_names == second.constr_names
    assert first.var_types == second.var_types
    assert (first.A != second.A).nnz == 0
    np.testing.assert_array_equal(first.c, second.c)
    np.testing.assert_array_equal(first.rhs, second.rhs)
    np.testing.assert_array_equal(first.l_var, second.l_var)
    np.testing.assert_array_equal(first.u_var, second.u_var)


def test_different_seeds_create_different_valid_instances():
    first = generate_refinery_scenario(2, seed=31)
    second = generate_refinery_scenario(2, seed=32)

    assert ModelValidator.validate(first).is_valid
    assert ModelValidator.validate(second).is_valid
    assert first.var_names == second.var_names
    assert first.constr_names == second.constr_names
    assert (first.A != second.A).nnz > 0
    assert not np.array_equal(first.c, second.c)


def test_benchmark_suite_has_expected_profiles_and_binary_model():
    suite = generate_refinery_benchmarks(seed=9)

    assert list(suite) == [
        "refinery_small",
        "refinery_blending",
        "refinery_inventory",
        "refinery_binary_operations",
    ]
    assert [suite[name].num_vars for name in suite] == [8, 16, 24, 18]
    assert [suite[name].num_constrs for name in suite] == [5, 10, 15, 10]
    assert suite["refinery_binary_operations"].binary_count == 2
    assert suite["refinery_binary_operations"].integer_count == 0
    assert all(
        "Synthetic refinery planning benchmark" in model.name
        for model in suite.values()
    )


def test_all_profiles_pass_existing_model_analysis_components():
    for model in generate_refinery_benchmarks(seed=5).values():
        validation = ModelValidator.validate(model)
        fingerprint = StructuralFingerprinter().analyze(model)
        health = NumericalHealthMonitor().evaluate(model)
        isg = ISGBuilder().build_bipartite_graph(model)
        strategy = StrategyRecommender().recommend(
            fingerprint,
            model=model,
            health=health,
            isg=isg,
        )

        assert validation.is_valid
        assert fingerprint.status == "OK"
        assert health.status == "HEALTHY"
        assert isg.status == "OK"
        assert strategy.status == "READY"


def test_binary_benchmark_runs_pipeline_and_reports_measured_results():
    report = RefineryBenchmarkRunner().run_benchmark(
        "refinery_binary_operations", seed=23
    )

    assert report.success is True
    assert report.pipeline_status == "COMPLETE"
    assert report.num_variables == 18
    assert report.num_constraints == 10
    assert report.nonzeros > 0
    assert report.objective_sense == "maximize"
    assert report.binary_variable_count == 2
    assert report.fingerprint_summary["status"] == "OK"
    assert report.numerical_health_status == "HEALTHY"
    assert report.recommended_strategy["primary_strategy"] == "GENERAL_MIP"
    assert report.solver_status == "OPTIMAL"
    assert report.objective_value is not None
    assert report.solve_time_seconds is not None
    assert report.sakshi_verification_status == "VERIFIED"
    assert report.certificate_digest
    assert json.loads(json.dumps(report.to_dict()))["seed"] == 23


def test_benchmark_runs_are_reproducible_without_claiming_timing_reproducibility():
    runner = RefineryBenchmarkRunner()
    first = runner.run_benchmark("refinery_small", seed=13)
    second = runner.run_benchmark("refinery_small", seed=13)

    assert first.solver_status == second.solver_status == "OPTIMAL"
    assert first.objective_value == pytest.approx(second.objective_value)
    assert first.certificate_digest == second.certificate_digest
    assert first.solve_time_seconds is not None
    assert second.solve_time_seconds is not None


def test_suite_runner_returns_all_reports_in_stable_order():
    reports = RefineryBenchmarkRunner().run_suite(seed=3)

    assert list(reports) == [
        "refinery_small",
        "refinery_blending",
        "refinery_inventory",
        "refinery_binary_operations",
    ]
    assert all(report.success for report in reports.values())


def test_unavailable_solver_information_is_not_fabricated():
    runner = RefineryBenchmarkRunner(
        solver_registry=SolverRegistry(register_defaults=False)
    )
    report = runner.run_benchmark("refinery_small", seed=3)

    assert report.success is False
    assert report.solver_status == "UNAVAILABLE"
    assert report.objective_value is None
    assert report.solve_time_seconds is None
    assert report.sakshi_verification_status == "UNAVAILABLE"
    assert report.certificate_digest is None
    assert report.errors


@pytest.mark.parametrize(
    "configuration",
    [
        {"num_periods": 0},
        {"num_periods": 13},
        {"num_periods": True},
        {"seed": -1},
        {"seed": 1.5},
        {"binary_operations": 1},
        {"benchmark_name": " "},
    ],
)
def test_invalid_generator_configuration_is_rejected(configuration):
    with pytest.raises(ValueError):
        generate_refinery_scenario(**configuration)


def test_unknown_benchmark_is_rejected_explicitly():
    with pytest.raises(ValueError, match="Unknown refinery benchmark"):
        RefineryBenchmarkRunner().run_benchmark("netlib_afiro", seed=1)


def test_backend_phase_metadata_keeps_netlib_separate():
    response = TestClient(app).get("/api/v1/system/info")

    assert response.status_code == 200
    modules = response.json()["modules"]
    assert modules["refinery_scenario"] == "IMPLEMENTED_PHASE_10"
    assert modules["refinery_benchmarks"] == "IMPLEMENTED_PHASE_10"
    assert modules["netlib_benchmarks"] == "IMPLEMENTED_PHASE_11_FILES_REQUIRED"