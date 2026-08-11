from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from apps.api.dependencies import (
    get_artifact_service,
    get_diagnosis_service,
    get_experiment_service,
    get_search_service,
    get_settings,
)
from apps.api.schemas import (
    ExperimentListResponse,
    QueryListItem,
    QueryListResponse,
    SearchRequestModel,
    SearchResponse,
)
from apps.api.services.artifact_service import ArtifactNotFoundError, ArtifactService
from apps.api.services.diagnosis_service import DiagnosisService
from apps.api.services.experiment_service import ExperimentService
from apps.api.services.search_service import LiveRetrievalError, QueryNotFoundError, SearchService
from apps.api.settings import ApiSettings


def error_payload(code: str, message: str, details: Any = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details or {}}}


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    app = FastAPI(
        title="ScholarPath Competition API",
        description=(
            "科研场景复杂学术查询的可评测、可解释、成本可控论文搜索推荐服务。"
        ),
        version=(settings or get_settings()).service_version,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if settings is not None:
        artifacts = ArtifactService(settings)
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides[get_artifact_service] = lambda: artifacts
        app.dependency_overrides[get_search_service] = lambda: SearchService(artifacts, settings)
        app.dependency_overrides[get_diagnosis_service] = lambda: DiagnosisService(artifacts)
        app.dependency_overrides[get_experiment_service] = lambda: ExperimentService(artifacts, settings)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=error_payload("VALIDATION_ERROR", "Request validation failed.", exc.errors()),
        )

    @app.exception_handler(ArtifactNotFoundError)
    async def artifact_handler(_: Request, exc: ArtifactNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content=error_payload("ARTIFACT_NOT_FOUND", "Required ScholarPath artifact is unavailable.", {"path": str(exc)}),
        )

    @app.exception_handler(QueryNotFoundError)
    async def query_handler(_: Request, exc: QueryNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=error_payload("QUERY_NOT_FOUND", "Unknown benchmark query id or frozen result.", {"qid": str(exc).strip("'")}),
        )

    @app.exception_handler(LiveRetrievalError)
    async def live_handler(_: Request, exc: LiveRetrievalError) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content=error_payload("LIVE_RETRIEVAL_FAILED", str(exc)),
        )

    @app.get("/health")
    def health(settings_: ApiSettings = Depends(get_settings)) -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "scholarpath-api",
            "version": settings_.service_version,
            "environment": settings_.environment,
        }

    @app.get("/api/system")
    def system_info(
        artifacts: ArtifactService = Depends(get_artifact_service),
        settings_: ApiSettings = Depends(get_settings),
    ) -> dict[str, Any]:
        health = artifacts.artifact_health()
        return {
            "project": "ScholarPath",
            "mode": "competition",
            "available_modes": ["benchmark", "live"],
            "features": {
                "constraint_parsing": True,
                "anchor_planning": True,
                "rescue_retrieval": True,
                "citation_expansion": True,
                "semantic_reranking": True,
                "day8_e3_reranking": True,
                "constraint_evidence": True,
                "deterministic_recommendation_reason": True,
                "guard_aware_selector": True,
            },
            "artifacts": health,
            "benchmarks_loaded": health["benchmark"],
            "version": settings_.service_version,
        }

    @app.get("/api/queries", response_model=QueryListResponse)
    def list_queries(
        artifacts: ArtifactService = Depends(get_artifact_service),
    ) -> QueryListResponse:
        queries = artifacts.benchmark_queries()
        day4_targets = artifacts.day4_target_qids()
        day5_targets = set(artifacts.day5_seeds())
        items = [
            QueryListItem(
                qid=qid,
                question=str(row.get("question") or ""),
                day4_rescue_triggered=qid in day4_targets,
                day5_citation_triggered=qid in day5_targets,
            )
            for qid, row in sorted(queries.items())
        ]
        return QueryListResponse(total=len(items), items=items)

    @app.post("/api/search", response_model=SearchResponse)
    def search(
        request: SearchRequestModel,
        service: SearchService = Depends(get_search_service),
    ) -> SearchResponse:
        return service.search(request)

    @app.get("/api/queries/{qid}/diagnosis")
    def diagnosis(
        qid: str,
        service: DiagnosisService = Depends(get_diagnosis_service),
    ) -> dict[str, Any]:
        return service.diagnose(qid)

    @app.get("/api/experiments", response_model=ExperimentListResponse)
    def experiments(
        service: ExperimentService = Depends(get_experiment_service),
    ) -> ExperimentListResponse:
        return service.list_experiments()

    return app


app = create_app()
