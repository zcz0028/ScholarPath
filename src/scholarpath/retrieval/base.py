from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from scholarpath.paper.schema import PaperRecord


@dataclass(slots=True)
class SearchRequest:
    query: str
    per_page: int = 100
    to_publication_date: str | None = None

    def __post_init__(self) -> None:
        self.query = str(self.query or "").strip()
        if not self.query:
            raise ValueError("Search query must be non-empty.")
        if not 1 <= int(self.per_page) <= 100:
            raise ValueError("per_page must be between 1 and 100.")
        self.per_page = int(self.per_page)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalStats:
    provider: str
    actual_api_calls: int = 0
    cache_hits: int = 0
    retries: int = 0
    http_status: int | None = None
    response_bytes: int = 0
    request_latency_ms: float = 0.0
    end_to_end_latency_ms: float = 0.0
    estimated_api_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    rate_limit_headers: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalResult:
    request: SearchRequest
    papers: list[PaperRecord]
    stats: RetrievalStats
    raw_result_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "papers": [paper.to_dict() for paper in self.papers],
            "stats": self.stats.to_dict(),
            "raw_result_count": self.raw_result_count,
        }
