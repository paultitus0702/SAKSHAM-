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

Planned future capability:

11. **Netlib Benchmark Validation**: Authentic Netlib LP benchmark verification (`afiro`, `adlittle`); external benchmark execution remains separate from the synthetic Phase 10 suite.

---

## Architecture Overview

- **Core Engine (`engine/`)**: Standalone Python scientific package (`saksham-engine`) using NumPy, SciPy sparse matrices, and HiGHS. The ISG currently uses its own deterministic node/edge representation.
- **Backend API (`backend/`)**: FastAPI health/system endpoints and a synchronous LP/MPS upload-and-solve endpoint. Job queues, persistence, and Server-Sent Events (SSE) are not implemented.
- **Frontend Dashboard (`frontend/`)**: React + Vite + Tailwind CSS dark-industrial cockpit.
- **Configuration (`config/`)**: Shared local defaults; separate Docker and cluster configurations are future work.

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
