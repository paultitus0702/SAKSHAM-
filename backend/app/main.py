"""
FastAPI Main Application Entry Point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .config import settings
from .api import api_router


class _RequestBodyTooLarge(Exception):
    """Internal signal used to stop reading an oversized upload body."""


class ModelUploadBodyLimitMiddleware:
    """Bound raw solve-upload bytes before Starlette parses multipart data."""

    def __init__(self, app, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != "/api/v1/solve"
        ):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    await self._send_too_large(scope, receive, send)
                    return
            except ValueError:
                pass

        received_bytes = 0
        body_limit_exceeded = False

        async def receive_with_limit():
            nonlocal received_bytes, body_limit_exceeded
            message = await receive()
            if message.get("type") == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_bytes:
                    body_limit_exceeded = True
                    raise _RequestBodyTooLarge
            return message

        async def send_unless_limited(message):
            if not body_limit_exceeded:
                await send(message)

        try:
            await self.app(scope, receive_with_limit, send_unless_limited)
        except _RequestBodyTooLarge:
            await self._send_too_large(scope, receive, send)

    async def _send_too_large(self, scope, receive, send) -> None:
        response = JSONResponse(
            status_code=413,
            content={
                "success": False,
                "status": "FAILED",
                "stage": "INPUT",
                "errors": [
                    {
                        "stage": "INPUT",
                        "code": "UPLOAD_TOO_LARGE",
                        "message": "Uploaded request exceeds the configured size limit.",
                    }
                ],
            },
        )
        await response(scope, receive, send)


app = FastAPI(
    title="SAKSHAM Optimization API",
    description="Sovereign Structure-Aware Sparse Optimization Engine API",
    version=settings.system.version
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.backend.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    ModelUploadBodyLimitMiddleware,
    max_bytes=32 * 1024 * 1024 + 64 * 1024,
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
