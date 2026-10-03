# SAKSHAM (सक्षम)
### Sovereign Structure-Aware Sparse Optimization Engine for India's Industrial Planning and Decision Systems

SAKSHAM ingests LP/MPS models, computes sparse structural and numerical diagnostics, builds weighted variable-constraint relationships, and provides heuristic recommendations before solving through its registered backend.

---

## Current and Planned Capabilities

Implemented through Phase 10:

1. **Model Ingestion**: CPLEX LP and MPS parsing into Canonical Internal Representation (CIR).
2. **Structural Fingerprinting**: Sparse matrix, coefficient-range, and conservative row-structure signals; no block decomposition is inferred.
3. **Industrial Structure Graph (ISG)**: Sparse weighted variable-constraint relationships with objective participation.
4. **Strategy Selection**: Deterministic heuristic recommendations with evidence and candidate labels.
5. **Optimization Engine**: One registered HiGHS (`highspy`) LP/MILP backend with solve-completion telemetry.
6. **Numerical Health Monitoring**: Coefficient dynamic range and zero-row/column diagnostics; exact condition estimates are not calculated.
7. **Solve Certificate Generator**: Deterministic SHA-256 digest separating solver-reported values from SAKSHI checks.
8. **SAKSHI Independent Verifier**: Primal feasibility, bounds, integrality, and objective-consistency checks. It does not prove dual feasibility, KKT conditions, or global optimality.
9. **Audit, Security, and Reliability**: Phase 9 architecture/security/reliability audit completed; documented limitations remain explicit.
10. **Synthetic Industrial Benchmarks**: Offline, seeded refinery-style planning LP/MILP suite exercised through the existing analysis, solve, verification, and certificate pipeline. Models are illustrative and are not real refinery data.

Implemented benchmark runner; authentic execution requires user-supplied Netlib files:

11. **Netlib Benchmark Validation**: Metadata-driven validation for `afiro`, `adlittle`, and `blend`, separate from the synthetic Phase 10 suite. No authentic benchmark files are currently included.

---

## Architecture Overview

- **Core Engine (`engine/`)**: Standalone Python scientific package (`saksham-engine`) using NumPy, SciPy sparse matrices, and HiGHS. The ISG currently uses its own deterministic node/edge representation.
- **Backend API (`backend/`)**: FastAPI health/system endpoints and a synchronous LP/MPS upload-and-solve endpoint. Job queues, persistence, and Server-Sent Events (SSE) are not implemented.
- **Frontend Dashboard (`frontend/`)**: React + Vite + Tailwind CSS dark-industrial cockpit.
- **Configuration (`config/`)**: Shared local defaults; separate Docker and cluster configurations are future work.

## Phase 11: Authentic Netlib Validation

Netlib provides public linear-programming instances for checking that the existing SAKSHAM parser and solve pipeline can process established model files. The Phase 11 runner does not download data and does not substitute synthetic refinery models.

The configured candidates are `afiro`, `adlittle`, and `blend`. Netlib publishes these in its [LP data index](https://www.netlib.org/lp/data/) using its compact `emps` MPS encoding. Place the original selected files in `data/netlib/` using the configured `.mps` filenames; SAKSHAM validates their stream checksums and expands the supported encoding before passing standard MPS text to the existing parser. Ordinary supported MPS files can also be supplied through `NetlibRunner` using its configurable input directory.

Run one candidate, a selected list, or all configured candidates through the existing Phase 8 pipeline:

```python
from pathlib import Path
from saksham.benchmarks import NetlibRunner

runner = NetlibRunner(benchmark_directory=Path("data/netlib"))
one = runner.run_benchmark("afiro")
selected = runner.run_benchmarks(["afiro", "blend"])
all_configured = runner.run_all()
```

Each result reports the declared source type, file/source identifier, parser status, model dimensions, sparse structure, numerical health, strategy, solver status/objective/time, SAKSHI status, certificate status/digest, warnings, and any failure stage/code/message. `AUTHENTIC_NETLIB` comes from the metadata entry and is a provenance declaration, not a content signature; discovered files without matching metadata are `UNCLASSIFIED_INPUT`. Offline test models are marked `SYNTHETIC_TEST_FIXTURE`.

A successful result requires the shared pipeline to complete with a `FEASIBLE` or `OPTIMAL` solver status, `VERIFIED` SAKSHI checks, and a generated certificate. SAKSHI checks feasibility and objective consistency; it does not independently prove global optimality. Missing authentic files are reported as `INPUT_NOT_FOUND`, with unavailable measurements left null. The runner currently reports all three configured candidates as unavailable until their files are supplied.

The current MPS parser rejects `RANGES` and `OBJSENSE` sections rather than silently changing model semantics. The LP parser supports a CPLEX-style linear subset, not quadratic, SOS, or other advanced sections. Parser compatibility and solver/verification status are reported per file; one failed benchmark does not stop the rest of the set.

---

## Local Development Quickstart

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Node.js 18+ & npm

### Backend & Engine Setup
```powershell
# 1. Activate Python virtual environment
.\venv\Scripts\activate

# 2. Run backend development server
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Health check endpoint: `http://127.0.0.1:8000/api/v1/health`

### Frontend Setup
```powershell
# 1. Start Vite development server
npm --prefix frontend run dev
```
Dashboard URL: `http://localhost:5173`

### Testing
```powershell
.\venv\Scripts\pytest -v tests
```
