"""Load the shared root ``config/default.json`` backend and engine settings."""
import json
import os
from pathlib import Path
from pydantic import BaseModel


class SystemConfig(BaseModel):
    app_name: str = "SAKSHAM"
    version: str = "0.1.0-alpha"
    environment: str = "development"
    log_level: str = "INFO"


class BackendConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    db_url: str = "sqlite:///./backend/saksham.db"


class EngineConfig(BaseModel):
    default_solver: str = "highs"
    feasibility_tolerance: float = 1e-7
    integrality_tolerance: float = 1e-6
    dynamic_range_warning_threshold: float = 1e7


class Settings(BaseModel):
    system: SystemConfig = SystemConfig()
    backend: BackendConfig = BackendConfig()
    engine: EngineConfig = EngineConfig()


def load_settings() -> Settings:
    root_dir = Path(__file__).resolve().parent.parent.parent
    config_path = root_dir / "config" / "default.json"
    
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Settings(**data)
        except Exception:
            return Settings()
    return Settings()


settings = load_settings()
