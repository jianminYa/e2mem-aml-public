"""High-Performance AML HTTP API Service with FastAPI & Uvicorn.

Provides fully compliant AML Add and Search operations:
- GET /health: Health probe endpoint
- POST /add: Add conversation transcript messages with idempotent request deduplication
- POST /search: Retrieve top-k (<=100) high-fidelity grounded evidence blocks within token budgets

Supports official MEMORY_SYSTEM_KEY authentication (Bearer, Token, X-Api-Key headers)
and non-blocking asynchronous execution via asyncio thread-pool offloading.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from service.core import MemoryService


# Global service instance
_memory_service: MemoryService | None = None


def get_service() -> MemoryService:
    global _memory_service
    if _memory_service is None:
        db_path = os.getenv("E2MEM_DB", "data/memory.sqlite3")
        mode = os.getenv("E2MEM_MODE", "competition")
        enable_jev = os.getenv("ENABLE_JEV", "0").lower() in {"1", "true", "yes"}
        _memory_service = MemoryService(db_path, mode=mode, enable_jev=enable_jev)
    return _memory_service


def verify_api_key(
    authorization: str | None = Header(None, alias="Authorization"),
    x_api_key: str | None = Header(None, alias="X-Api-Key"),
) -> None:
    expected_key = os.getenv("MEMORY_SYSTEM_KEY", "").strip()
    is_competition = os.getenv("E2MEM_MODE", "simulator") == "competition"
    if not expected_key:
        if is_competition:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server configuration error: MEMORY_SYSTEM_KEY must be configured in competition mode",
            )
        return  # No authentication enforced in local simulator mode

    if x_api_key and x_api_key.strip() == expected_key:
        return

    if authorization:
        val = authorization.strip()
        if val == expected_key:
            return
        parts = val.split()
        if len(parts) == 2 and parts[0].lower() in {"bearer", "token"} and parts[1] == expected_key:
            return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized: invalid or missing MEMORY_SYSTEM_KEY",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize service during startup
    get_service()
    yield


app = FastAPI(
    title="E2Mem Evidence Anchor AML Service",
    description="Agent Memory Leaderboard (AML) Official Microservice Adapter",
    version="2.6.3",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    path = request.url.path
    print(f"[ACCESS] {client_ip} {method} {path} - incoming", flush=True)
    response = await call_next(request)
    print(f"[ACCESS] {client_ip} {method} {path} -> {response.status_code}", flush=True)
    return response


class AddPayload(BaseModel):
    request_id: str = Field(..., description="Unique idempotency identifier for the Add request")
    user_id: str = Field(..., description="Target user identifier")
    session_id: str = Field(..., description="Session identifier for multi-session dialogues")
    messages: list[dict[str, Any]] = Field(..., description="Transcript messages in chronologic order")


class SearchPayload(BaseModel):
    user_id: str = Field(..., description="Target user identifier")
    query: str = Field(..., description="Search query string")
    top_k: int = Field(default=100, ge=1, le=100, description="Maximum number of memories to return (<=100)")
    options: list[Any] | None = Field(default=None, description="Multiple-choice options (recorded in the search trace only)")


class MemoryRecord(BaseModel):
    id: str
    content: str
    score: float
    created_at: str = ""


class SearchResponse(BaseModel):
    data: list[MemoryRecord]


@app.api_route("/", methods=["GET", "HEAD", "OPTIONS"], status_code=status.HTTP_200_OK)
@app.api_route("/health", methods=["GET", "HEAD", "OPTIONS"], status_code=status.HTTP_200_OK)
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "e2mem-evidence-anchor", "version": "2.6.3"}


@app.api_route("/add", methods=["GET", "HEAD", "OPTIONS"], status_code=status.HTTP_200_OK)
async def add_probe() -> dict[str, str]:
    return {"status": "ok", "service": "e2mem-evidence-anchor", "endpoint": "add", "version": "2.6.3"}


@app.api_route("/search", methods=["GET", "HEAD", "OPTIONS"], status_code=status.HTTP_200_OK)
async def search_probe() -> dict[str, str]:
    return {"status": "ok", "service": "e2mem-evidence-anchor", "endpoint": "search", "version": "2.6.3"}


@app.post("/add", status_code=status.HTTP_200_OK, dependencies=[Depends(verify_api_key)])
async def add_memories(payload: AddPayload) -> dict[str, Any]:
    svc = get_service()
    try:
        raw_dict = payload.model_dump()
        result, stats = await asyncio.to_thread(svc.add, raw_dict)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except RuntimeError as e:
        code = status.HTTP_409_CONFLICT if "request_id reused" in str(e).lower() else status.HTTP_503_SERVICE_UNAVAILABLE
        raise HTTPException(status_code=code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal extraction error: {e}")


@app.post("/search", response_model=SearchResponse, status_code=status.HTTP_200_OK, dependencies=[Depends(verify_api_key)])
async def search_memories(payload: SearchPayload) -> dict[str, Any]:
    svc = get_service()
    try:
        raw_dict = payload.model_dump()
        result, stats = await asyncio.to_thread(svc.search, raw_dict)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal search error: {e}")


# Legacy Handler for backwards compatibility with ThreadingHTTPServer imports
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class Handler(BaseHTTPRequestHandler):
    service: MemoryService

    def _reply(self, status_code: int, data: dict) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path in {"/", "/health"}:
            self._reply(200, {"status": "ok", "service": "e2mem-evidence-anchor", "version": "2.6.3"})
        elif self.path in {"/add", "/search"}:
            self._reply(200, {"status": "ok", "service": "e2mem-evidence-anchor", "endpoint": self.path[1:], "version": "2.6.3"})
        else:
            self._reply(404, {"detail": "not found"})

    def do_HEAD(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:
        if self.path not in {"/add", "/search"}:
            self._reply(404, {"detail": "not found"})
            return

        key = os.getenv("MEMORY_SYSTEM_KEY", "").strip()
        if key:
            values = {f"Token {key}", f"Bearer {key}", key}
            if self.headers.get("Authorization") not in values and self.headers.get("X-Api-Key") != key:
                self._reply(401, {"detail": "unauthorized"})
                return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 8_000_000:
                raise ValueError("request size outside accepted range")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request must be an object")

            response, _ = self.service.add(payload) if self.path == "/add" else self.service.search(payload)
            self._reply(200, response)
        except ValueError as error:
            self._reply(422, {"detail": str(error)})
        except RuntimeError as error:
            code = 409 if "request_id reused" in str(error) else 503
            self._reply(code, {"detail": str(error)})
        except Exception as error:
            self._reply(500, {"detail": type(error).__name__})


def main() -> None:
    import uvicorn
    host = os.getenv("E2MEM_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    print(f"Starting E2Mem Evidence Anchor FastAPI server on {host}:{port}")
    uvicorn.run("service.api:app", host=host, port=port, log_level="info", access_log=True)


if __name__ == "__main__":
    main()
