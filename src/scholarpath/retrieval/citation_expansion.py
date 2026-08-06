from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

import requests

from scholarpath.paper.schema import PaperRecord
from scholarpath.rerank.constraint_rerank import paper_key
from scholarpath.retrieval.openalex import openalex_work_to_paper


class CitationBudgetExhausted(RuntimeError):
    """Raised when another uncached OpenAlex call would exceed the budget."""


class ResponseLike(Protocol):
    status_code: int
    content: bytes

    def json(self) -> Any: ...

    def raise_for_status(self) -> None: ...


class SessionLike(Protocol):
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: tuple[float, float],
    ) -> ResponseLike: ...


@dataclass(slots=True, frozen=True)
class CitationExpansionConfig:
    max_hops: int = 1
    max_references_per_seed: int = 15
    max_cited_by_per_seed: int = 15
    max_unique_candidates_per_query: int = 80
    max_api_calls: int = 60
    cache_dir: str = "data/cache/openalex_day5_citation"
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 45.0
    user_agent: str = "ScholarPath/0.1 (controlled citation expansion)"

    def __post_init__(self) -> None:
        if self.max_hops != 1:
            raise ValueError("Day 5 citation expansion only supports one hop")
        for name in (
            "max_references_per_seed",
            "max_cited_by_per_seed",
            "max_unique_candidates_per_query",
            "max_api_calls",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} cannot be negative")


@dataclass(slots=True)
class CitationBudget:
    max_api_calls: int
    actual_api_calls: int = 0
    cache_hits: int = 0

    def can_call(self) -> bool:
        return self.actual_api_calls < self.max_api_calls

    def record_api_call(self) -> None:
        if not self.can_call():
            raise CitationBudgetExhausted("Citation API budget exhausted")
        self.actual_api_calls += 1

    def record_cache_hit(self) -> None:
        self.cache_hits += 1


@dataclass(slots=True)
class CitationPath:
    qid: str
    seed_openalex_id: str
    seed_title: str
    expanded_openalex_id: str | None
    expanded_title: str
    edge_type: str
    seed_rank: int
    edge_rank: int
    hop: int = 1
    trigger_reason: str = "day4_zero_recall"

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "seed_openalex_id": self.seed_openalex_id,
            "seed_title": self.seed_title,
            "expanded_openalex_id": self.expanded_openalex_id,
            "expanded_title": self.expanded_title,
            "edge_type": self.edge_type,
            "seed_rank": self.seed_rank,
            "edge_rank": self.edge_rank,
            "hop": self.hop,
            "trigger_reason": self.trigger_reason,
        }


class OpenAlexCitationProvider:
    def __init__(
        self,
        config: CitationExpansionConfig,
        budget: CitationBudget,
        session: SessionLike | None = None,
    ) -> None:
        self.config = config
        self.budget = budget
        self.session = session or requests.Session()
        self.base_url = "https://api.openalex.org"
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def api_key(self) -> str | None:
        value = os.getenv("OPENALEX_API_KEY")
        return value.strip() if value and value.strip() else None

    def _cache_path(self, kind: str, work_id: str, limit: int) -> Path:
        key = hashlib.sha256(f"{kind}|{work_id}|{limit}".encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def has_cached_request(self, kind: str, work_id: str, limit: int) -> bool:
        return self._cache_path(kind, work_id, limit).exists()

    def _request(
        self,
        *,
        kind: str,
        work_id: str,
        limit: int,
        refresh_cache: bool = False,
    ) -> dict[str, Any]:
        cache_path = self._cache_path(kind, work_id, limit)
        if cache_path.exists() and not refresh_cache:
            self.budget.record_cache_hit()
            return json.loads(cache_path.read_text(encoding="utf-8"))

        if not self.api_key:
            raise RuntimeError("OPENALEX_API_KEY is not set")

        # Count the attempt before sending the request. A timeout still consumed
        # a real request attempt and therefore belongs in the budget.
        self.budget.record_api_call()

        if kind == "seed":
            url = f"{self.base_url}/works/{work_id}"
            params = {
                "api_key": self.api_key,
                "select": (
                    "id,doi,display_name,publication_year,publication_date,"
                    "authorships,primary_location,best_oa_location,ids,"
                    "cited_by_count,referenced_works"
                ),
            }
        elif kind == "cited_by":
            url = f"{self.base_url}/works"
            params = {
                "api_key": self.api_key,
                "filter": f"cites:{work_id}",
                "per-page": min(100, max(1, limit)),
                "select": (
                    "id,doi,display_name,publication_year,publication_date,"
                    "authorships,primary_location,best_oa_location,ids,"
                    "cited_by_count"
                ),
            }
        elif kind == "references":
            url = f"{self.base_url}/works"
            params = {
                "api_key": self.api_key,
                "filter": f"openalex_id:{work_id}",
                "per-page": min(100, max(1, limit)),
                "select": (
                    "id,doi,display_name,publication_year,publication_date,"
                    "authorships,primary_location,best_oa_location,ids,"
                    "cited_by_count"
                ),
            }
        else:
            raise ValueError(f"Unknown request kind: {kind}")

        started = time.perf_counter()
        response = self.session.get(
            url,
            params=params,
            headers={"User-Agent": self.config.user_agent},
            timeout=(
                self.config.connect_timeout_seconds,
                self.config.read_timeout_seconds,
            ),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("OpenAlex response must be a JSON object")

        payload["_latency_ms"] = (time.perf_counter() - started) * 1000.0
        cache_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        return payload

    def get_references(
        self,
        seed: PaperRecord,
        *,
        refresh_cache: bool = False,
    ) -> list[PaperRecord]:
        if not seed.openalex_id:
            return []

        seed_payload = self._request(
            kind="seed",
            work_id=seed.openalex_id,
            limit=1,
            refresh_cache=refresh_cache,
        )
        ids = [
            str(item).rsplit("/", 1)[-1]
            for item in seed_payload.get("referenced_works", [])
            if item
        ][: self.config.max_references_per_seed]
        if not ids:
            return []

        payload = self._request(
            kind="references",
            work_id="|".join(ids),
            limit=len(ids),
            refresh_cache=refresh_cache,
        )
        return [
            openalex_work_to_paper(item)
            for item in payload.get("results", [])
            if isinstance(item, Mapping)
        ]

    def get_cited_by(
        self,
        seed: PaperRecord,
        *,
        refresh_cache: bool = False,
    ) -> list[PaperRecord]:
        if not seed.openalex_id:
            return []

        payload = self._request(
            kind="cited_by",
            work_id=seed.openalex_id,
            limit=self.config.max_cited_by_per_seed,
            refresh_cache=refresh_cache,
        )
        return [
            openalex_work_to_paper(item)
            for item in payload.get("results", [])
            if isinstance(item, Mapping)
        ]


def annotate_citation_candidate(
    paper: PaperRecord,
    *,
    seed: PaperRecord,
    edge_type: str,
    seed_rank: int,
    edge_rank: int,
    seed_score: float,
) -> PaperRecord:
    raw = dict(paper.raw or {})
    paths = raw.get("day5_citation_paths")
    if not isinstance(paths, list):
        paths = []

    paths.append(
        {
            "seed_openalex_id": seed.openalex_id,
            "seed_title": seed.title,
            "edge_type": edge_type,
            "seed_rank": seed_rank,
            "edge_rank": edge_rank,
            "seed_score": seed_score,
            "hop": 1,
        }
    )
    raw["day5_citation_paths"] = paths
    raw["day5_citation_support_count"] = len(
        {str(item.get("seed_openalex_id")) for item in paths}
    )
    raw["day5_citation_edge_types"] = sorted(
        {str(item.get("edge_type")) for item in paths}
    )
    raw["day5_citation_bonus"] = min(
        0.15,
        0.06 * seed_score
        + 0.04 * min(raw["day5_citation_support_count"], 2),
    )
    paper.raw = raw
    return paper


def deduplicate_citation_candidates(
    papers: Iterable[PaperRecord],
    *,
    limit: int,
) -> list[PaperRecord]:
    merged: dict[str, PaperRecord] = {}
    for paper in papers:
        key = paper_key(paper.to_dict(include_raw=True))
        if key not in merged:
            merged[key] = paper
            continue

        base = merged[key]
        base_paths = list((base.raw or {}).get("day5_citation_paths") or [])
        incoming_paths = list((paper.raw or {}).get("day5_citation_paths") or [])
        raw = dict(base.raw or {})
        raw["day5_citation_paths"] = base_paths + incoming_paths
        raw["day5_citation_support_count"] = len(
            {
                str(item.get("seed_openalex_id"))
                for item in raw["day5_citation_paths"]
            }
        )
        raw["day5_citation_edge_types"] = sorted(
            {
                str(item.get("edge_type"))
                for item in raw["day5_citation_paths"]
            }
        )
        raw["day5_citation_bonus"] = min(
            0.15,
            max(
                float(raw.get("day5_citation_bonus") or 0.0),
                float((paper.raw or {}).get("day5_citation_bonus") or 0.0),
            )
            + 0.02,
        )
        base.raw = raw

    ordered = sorted(
        merged.values(),
        key=lambda paper: (
            -float((paper.raw or {}).get("day5_citation_bonus") or 0.0),
            -(paper.citation_count or 0),
            paper.title.casefold(),
        ),
    )
    return ordered[: max(0, limit)]
