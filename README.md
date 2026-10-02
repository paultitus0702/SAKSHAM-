# SAKSHAM (सक्षम)
### Sovereign Structure-Aware Sparse Optimization Engine for India's Industrial Planning and Decision Systems

SAKSHAM is a research-grade mathematical optimization platform designed to uncover and exploit macroscopic mathematical topologies (network flows, inventory balance equations, bordered block-angular couplings, and dynamic scale properties) before submitting problems to sparse numerical solvers.

---

## Key Capabilities (Planned Across Phases)

1. **Model Ingestion**: CPLEX LP and fixed/free MPS parsing into Canonical Internal Representation (CIR).
2. **Structural Fingerprinting**: Sparsity metrics, dynamic range, conservation rows, and block decomposition.
3. **Industrial Structure Graph (ISG)**: Multi-layer graph representation (Bipartite, Conflict, Network Flow).
4. **Strategy Selection**: Structure-aware algorithm and scaling recommendations with explainable technical rationale.
5. **Optimization Engine**: Integration with open-source HiGHS (`highspy`) with streaming iteration telemetry.
6. **Numerical Health Monitoring**: Dynamic range and condition estimation warnings before matrix factorization.
7. **Solve Certificate Generator**: Cryptographic SHA-256 sealed certificates with primal/dual residuals.
8. **Sakshi Independent Verifier**: Zero-trust validation auditing solutions against original raw constraints.
9. **Sovereign Industrial Scenarios**: Realistic synthetic industrial models (Refinery Blending, Power Grid, Logistics).
10. **Benchmark Validation**: Authentic Netlib LP benchmark verification (`afiro`, `adlittle`).

---

## Architecture Overview

- **Core Engine (`engine/`)**: Standalone Python scientific package (`saksham-engine`) using NumPy, SciPy sparse matrices, and NetworkX.
- **Backend API (`backend/`)**: FastAPI gateway with asynchronous job scheduling, SQLite persistence, and Server-Sent Events (SSE).
- **Frontend Dashboard (`frontend/`)**: React + Vite + Tailwind CSS dark-industrial cockpit.
- **Configuration (`config/`)**: Environment-specific configurations for local, Docker, and cluster execution.

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
