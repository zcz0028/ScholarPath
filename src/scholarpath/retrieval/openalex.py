from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import requests

from scholarpath.paper.normalizers import normalize_arxiv_id
from scholarpath.paper.schema import PaperRecord

from .base import RetrievalResult, RetrievalStats, SearchRequest


class ResponseLike(Protocol):
    status_code: int
    content: bytes
    headers: Mapping[str, str]

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


@dataclass(slots=True)
class OpenAlexConfig:
    api_key: str | None = None
    base_url: str = "https://api.openalex.org"
    cache_dir: str = "data/cache/openalex"
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 45.0
    max_retries: int = 3
    bad_request_fallback_enabled: bool = True
    backoff_base_seconds: float = 1.0
    search_cost_per_1000_calls_usd: float = 1.0
    user_agent: str = "ScholarPath/0.1 (academic retrieval benchmark)"
    select_fields: tuple[str, ...] = (
        "id",
        "doi",
        "display_name",
        "publication_year",
        "publication_date",
        "authorships",
        "primary_location",
        "best_oa_location",
        "ids",
        "cited_by_count",
        "type",
    )

    def resolved_api_key(self) -> str | None:
        value = self.api_key or os.getenv("OPENALEX_API_KEY")
        return value.strip() if value and value.strip() else None

    @property
    def estimated_search_cost_per_call_usd(self) -> float:
        return float(self.search_cost_per_1000_calls_usd) / 1000.0


class OpenAlexError(RuntimeError):
    def __init__(
        self,
        message: str,
        stats: RetrievalStats | None = None,
    ) -> None:
        super().__init__(message)
        self.stats = stats


def _cache_key(request: SearchRequest, select_fields: tuple[str, ...]) -> str:
    payload = {
        "query": request.query,
        "per_page": request.per_page,
        "to_publication_date": request.to_publication_date,
        "select_fields": list(select_fields),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def sanitize_openalex_search_query(query: object | None) -> str:
    """Build a conservative fallback query for OpenAlex search.

    This is only used after OpenAlex rejects the raw query with HTTP 400.
    It preserves words and digits, but removes punctuation that can make a
    complex natural-language query hard for OpenAlex to parse.
    """
    text = str(query or "").strip()
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212/_]+", " ", text)

    cleaned_chars: list[str] = []
    for char in text:
        if char.isalnum():
            cleaned_chars.append(char)
        else:
            cleaned_chars.append(" ")

    cleaned = " ".join("".join(cleaned_chars).split())
    return cleaned or text


def _extract_arxiv_from_url(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value)
    match = re.search(
        r"arxiv\.org/(?:abs|pdf)/([^?#]+)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return normalize_arxiv_id(match.group(1))


def _location_value(work: Mapping[str, Any], field: str) -> Any:
    for location_key in ("primary_location", "best_oa_location"):
        location = work.get(location_key)
        if isinstance(location, Mapping) and location.get(field):
            return location.get(field)
    return None


def openalex_work_to_paper(work: Mapping[str, Any]) -> PaperRecord:
    ids = work.get("ids") if isinstance(work.get("ids"), Mapping) else {}
    primary_location = (
        work.get("primary_location")
        if isinstance(work.get("primary_location"), Mapping)
        else {}
    )
    source = (
        primary_location.get("source")
        if isinstance(primary_location.get("source"), Mapping)
        else {}
    )

    landing_page_url = _location_value(work, "landing_page_url")
    pdf_url = _location_value(work, "pdf_url")
    arxiv_id = (
        _extract_arxiv_from_url(landing_page_url)
        or _extract_arxiv_from_url(pdf_url)
    )

    authors: list[str] = []
    authorships = work.get("authorships")
    if isinstance(authorships, list):
        for authorship in authorships:
            if not isinstance(authorship, Mapping):
                continue
            author = authorship.get("author")
            if not isinstance(author, Mapping):
                continue
            name = str(author.get("display_name") or "").strip()
            if name:
                authors.append(name)

    openalex_id = work.get("id") or ids.get("openalex")
    url = landing_page_url or work.get("doi") or openalex_id

    raw = dict(work)
    relevance_score = work.get("relevance_score")
    if relevance_score is not None:
        raw["retrieval_score"] = relevance_score

    return PaperRecord(
        title=str(work.get("display_name") or work.get("title") or ""),
        source="openalex",
        source_record_id=str(openalex_id) if openalex_id else None,
        authors=authors,
        year=work.get("publication_year"),
        publication_date=(
            str(work.get("publication_date"))
            if work.get("publication_date")
            else None
        ),
        venue=str(source.get("display_name") or "").strip() or None,
        doi=work.get("doi") or ids.get("doi"),
        arxiv_id=arxiv_id,
        openalex_id=openalex_id,
        url=str(url) if url else None,
        citation_count=work.get("cited_by_count"),
        raw=raw,
    )


class OpenAlexRetriever:
    def __init__(
        self,
        config: OpenAlexConfig | None = None,
        session: SessionLike | None = None,
        sleep_fn: Any = time.sleep,
    ) -> None:
        self.config = config or OpenAlexConfig()
        self.session = session or requests.Session()
        self.sleep_fn = sleep_fn

    def search(
        self,
        request: SearchRequest,
        *,
        refresh_cache: bool = False,
    ) -> RetrievalResult:
        started = time.perf_counter()
        stats = RetrievalStats(provider="openalex")
        cache_path = self._cache_path(request)

        if cache_path.exists() and not refresh_cache:
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                response_payload = payload["response"]
                papers = self._parse_response(response_payload)
                stats.cache_hits = 1
                stats.end_to_end_latency_ms = (
                    time.perf_counter() - started
                ) * 1000.0
                return RetrievalResult(
                    request=request,
                    papers=papers,
                    stats=stats,
                    raw_result_count=len(response_payload.get("results", [])),
                )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                # Corrupt caches are ignored and replaced by a fresh response.
                pass

        api_key = self.config.resolved_api_key()
        if not api_key:
            raise OpenAlexError(
                "OPENALEX_API_KEY is not set. Create a free OpenAlex API key "
                "and set it in the current terminal environment."
            )

        params: dict[str, Any] = {
            "api_key": api_key,
            "search": request.query,
            "per_page": request.per_page,
            "select": ",".join(self.config.select_fields),
        }
        if request.to_publication_date:
            params["filter"] = (
                f"to_publication_date:{request.to_publication_date}"
            )

        headers = {
            "Accept": "application/json",
            "User-Agent": self.config.user_agent,
        }

        last_error: Exception | None = None
        successful_response: ResponseLike | None = None
        request_latency_ms = 0.0

        for attempt in range(self.config.max_retries + 2):
            call_started = time.perf_counter()
            stats.actual_api_calls += 1
            try:
                response = self.session.get(
                    f"{self.config.base_url.rstrip('/')}/works",
                    params=params,
                    headers=headers,
                    timeout=(
                        self.config.connect_timeout_seconds,
                        self.config.read_timeout_seconds,
                    ),
                )
            except requests.RequestException as exc:
                request_latency_ms += (
                    time.perf_counter() - call_started
                ) * 1000.0
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                stats.retries += 1
                self.sleep_fn(self._retry_delay(attempt, None))
                continue

            request_latency_ms += (
                time.perf_counter() - call_started
            ) * 1000.0
            stats.http_status = int(response.status_code)
            stats.response_bytes += len(response.content or b"")
            stats.rate_limit_headers = {
                str(key): str(value)
                for key, value in response.headers.items()
                if str(key).lower().startswith(
                    ("x-ratelimit", "ratelimit", "retry-after")
                )
            }

            if response.status_code in {429, 500, 502, 503, 504}:
                if attempt < self.config.max_retries:
                    stats.retries += 1
                    delay = self._retry_delay(
                        attempt,
                        response.headers.get("Retry-After"),
                    )
                    self.sleep_fn(delay)
                    continue

            if (
                response.status_code == 400
                and self.config.bad_request_fallback_enabled
            ):
                safe_query = sanitize_openalex_search_query(request.query)
                if safe_query and params.get("search") != safe_query:
                    params["search"] = safe_query
                    stats.retries += 1
                    last_error = requests.HTTPError(
                        "HTTP 400; retried once with sanitized OpenAlex query."
                    )
                    continue

            try:
                response.raise_for_status()
            except requests.RequestException as exc:
                last_error = exc
                break
            successful_response = response
            break

        stats.request_latency_ms = request_latency_ms
        stats.estimated_api_cost_usd = (
            stats.actual_api_calls
            * self.config.estimated_search_cost_per_call_usd
        )

        if successful_response is None:
            stats.error = str(last_error or "No successful HTTP response received.")
            stats.end_to_end_latency_ms = (
                time.perf_counter() - started
            ) * 1000.0
            raise OpenAlexError(stats.error, stats=stats)

        response = successful_response
        try:
            response_payload = response.json()
        except ValueError as exc:
            stats.error = "OpenAlex returned invalid JSON."
            raise OpenAlexError(stats.error, stats=stats) from exc

        if not isinstance(response_payload, dict):
            raise OpenAlexError("OpenAlex response must be a JSON object.", stats=stats)

        papers = self._parse_response(response_payload)
        self._write_cache(cache_path, request, response_payload)
        stats.end_to_end_latency_ms = (
            time.perf_counter() - started
        ) * 1000.0

        return RetrievalResult(
            request=request,
            papers=papers,
            stats=stats,
            raw_result_count=len(response_payload.get("results", [])),
        )

    def _cache_path(self, request: SearchRequest) -> Path:
        key = _cache_key(request, self.config.select_fields)
        return Path(self.config.cache_dir) / f"{key}.json"

    @staticmethod
    def _parse_response(payload: Mapping[str, Any]) -> list[PaperRecord]:
        results = payload.get("results", [])
        if not isinstance(results, list):
            raise OpenAlexError("OpenAlex response 'results' must be a list.")
        papers: list[PaperRecord] = []
        for item in results:
            if not isinstance(item, Mapping):
                continue
            paper = openalex_work_to_paper(item)
            if paper.title:
                papers.append(paper)
        return papers

    @staticmethod
    def _write_cache(
        path: Path,
        request: SearchRequest,
        response_payload: Mapping[str, Any],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(
                {
                    "request": request.to_dict(),
                    "response": response_payload,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary_path.replace(path)

    def _retry_delay(
        self,
        attempt: int,
        retry_after: str | None,
    ) -> float:
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        exponential = self.config.backoff_base_seconds * (2**attempt)
        return exponential + random.uniform(0.0, 0.25)
