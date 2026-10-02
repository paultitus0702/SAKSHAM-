"""
FastAPI Main Application Entry Point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .api import api_router

app = FastAPI(
    title="SAKSHAM Optimization API",
    description="Sovereign Structure-Aware Sparse Optimization Engine API",
    version=settings.system.version
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.backend.cors_origins + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routers
app.include_router(api_router)


@app.get("/")
def root():
    return {
        "message": "SAKSHAM Sovereign Optimization Engine API is running.",
        "docs_url": "/docs",
        "health_url": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.backend.host, port=settings.backend.port, reload=True)
