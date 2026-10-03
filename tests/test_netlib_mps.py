"""Netlib compact-MPS expansion and authentic local pipeline regressions."""
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from saksham.parser.base import ModelParseError
from saksham.parser.mps_parser import MPSParser
from saksham.pipeline import run_pipeline


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_NETLIB_DIRECTORY = _REPOSITORY_ROOT / "data" / "netlib"
_COMPRESSED_FIXTURE = Path(__file__).parent / "test_data" / "netlib_compressed_minimal.mps"
_AUTHENTIC_DIMENSIONS = {
    "afiro": (32, 27, 83),
    "adlittle": (97, 56, 383),
    "blend": (83, 74, 491),
}


def test_minimal_netlib_compressed_input_reproduces_missing_rows_failure():
    model = MPSParser().parse_file(_COMPRESSED_FIXTURE)

    assert model.name == "netlib_compressed_minimal"
    assert model.var_names == ["x"]
    assert model.constr_names == ["CON"]
    assert model.A[0, 0] == 1.0
    assert model.rhs.tolist() == [2.0]
    assert sparse.isspmatrix_csr(model.A)
    assert np.array_equal(model.c, np.array([1.0]))


def test_compressed_netlib_checksum_is_validated():
    content = _COMPRESSED_FIXTURE.read_text(encoding="ascii")
    tampered_content = content.replace("GCON", "GXXX")

    with pytest.raises(ModelParseError, match="checksum mismatch"):
        MPSParser().parse_string(tampered_content)


def test_compressed_netlib_ranges_are_rejected_explicitly():
    lines = _COMPRESSED_FIXTURE.read_text(encoding="ascii").splitlines()
    statistics = [int(lines[1][index : index + 9]) for index in range(0, 72, 9)]
    statistics[6] = 1
    statistics[7] = 1
    lines[1] = "".join(f"{value:9d}" for value in statistics)

    with pytest.raises(ModelParseError, match="RANGES sections are not supported"):
        MPSParser().parse_string("\n".join(lines) + "\n")


@pytest.mark.parametrize("benchmark_name", ["afiro", "adlittle", "blend"])
def test_authentic_netlib_file_runs_through_parser_and_existing_pipeline(benchmark_name):
    benchmark_path = _NETLIB_DIRECTORY / f"{benchmark_name}.mps"
    if not benchmark_path.is_file():
        pytest.skip(f"Local authentic Netlib input is unavailable: {benchmark_path.name}")

    model = MPSParser().parse_file(benchmark_path)
    expected_variables, expected_constraints, expected_nonzeros = _AUTHENTIC_DIMENSIONS[
        benchmark_name
    ]
    assert model.num_vars == expected_variables
    assert model.num_constrs == expected_constraints
    assert model.num_nonzeros == expected_nonzeros

    result = run_pipeline(benchmark_path)

    assert result.parser_status == "OK"
    assert result.stage == "COMPLETE"
    assert result.fingerprint.status == "OK"
    assert result.health.status in {"HEALTHY", "WARNING"}
    assert result.isg.status == "OK"
    assert result.strategy.status in {"READY", "WARNING"}
    assert result.solver_result.status in {"OPTIMAL", "FEASIBLE"}
    assert result.verification.status == "VERIFIED"
    assert result.certificate.sha256_hash