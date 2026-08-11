from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class SearchRequestModel(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    qid: str | None = Field(default=None, max_length=128)
    mode: Literal["benchmark", "live"] = "benchmark"
    top_k: Literal[20, 50, 100] = 20
    enable_citation: bool = False

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        value = " ".join(value.split())
        if len(value) < 3:
            raise ValueError("query must contain at least 3 non-whitespace characters")
        return value


class CostSummary(BaseModel):
    api_calls: int = 0
    cache_hits: int = 0
    retries: int = 0
    estimated_cost_usd: float = 0.0


class CitationPathView(BaseModel):
    seed_openalex_id: str | None = None
    seed_title: str | None = None
    expanded_openalex_id: str | None = None
    expanded_title: str | None = None
    edge_type: str | None = None
    hop: int | None = None


class PaperResult(BaseModel):
    rank: int
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    openalex_id: str | None = None
    url: str | None = None
    score: float | None = None
    relevance_level: str | None = None
    reason_tags: list[str] = Field(default_factory=list)
    reason_text: str | None = None
    constraint_evidence: list[dict[str, Any]] = Field(default_factory=list)
    matched_constraints: list[str] = Field(default_factory=list)
    unmatched_constraints: list[str] = Field(default_factory=list)
    matched_count: int = 0
    constraint_count: int = 0
    retrieval_sources: list[str] = Field(default_factory=list)
    citation_path: CitationPathView | None = None


class PipelineSummary(BaseModel):
    stages: list[dict[str, Any]] = Field(default_factory=list)
    total_candidates: int = 0
    returned_results: int = 0


class SearchResponse(BaseModel):
    run_id: str
    query: str
    qid: str | None = None
    mode: Literal["benchmark", "live"]
    parsed_constraints: list[dict[str, Any]] = Field(default_factory=list)
    academic_anchors: list[dict[str, Any]] = Field(default_factory=list)
    query_plan: list[dict[str, Any]] = Field(default_factory=list)
    results: list[PaperResult] = Field(default_factory=list)
    pipeline: PipelineSummary
    cost: CostSummary
    latency_ms: int
    warnings: list[str] = Field(default_factory=list)


class QueryListItem(BaseModel):
    qid: str
    question: str
    day4_rescue_triggered: bool = False
    day5_citation_triggered: bool = False


class QueryListResponse(BaseModel):
    total: int
    items: list[QueryListItem]


class ExperimentItem(BaseModel):
    id: str
    label: str
    role: str
    description: str
    available: bool
    metrics: dict[str, Any] = Field(default_factory=dict)


class ExperimentListResponse(BaseModel):
    items: list[ExperimentItem]
