from __future__ import annotations

import time
import uuid
from typing import Any

from apps.api.schemas import (
    CitationPathView,
    CostSummary,
    PaperResult,
    PipelineSummary,
    SearchRequestModel,
    SearchResponse,
)
from apps.api.services.artifact_service import ArtifactService
from apps.api.settings import ApiSettings

from scholarpath.query.academic_query_planner import AcademicQueryPlanner, PlannerConfig
from scholarpath.query.constraints import ConstraintDecomposer
from scholarpath.rerank.semantic_rerank import (
    annotate_and_semantic_rerank_record,
    make_semantic_config,
)
from scholarpath.retrieval.base import SearchRequest
from scholarpath.retrieval.openalex import OpenAlexConfig, OpenAlexError, OpenAlexRetriever


class QueryNotFoundError(KeyError):
    pass


class LiveRetrievalError(RuntimeError):
    pass


def _raw_meta(paper: dict[str, Any]) -> dict[str, Any]:
    raw = paper.get("raw")
    return raw if isinstance(raw, dict) else {}


def _float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _authors(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    for item in value:
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            nested = item.get("author") if isinstance(item.get("author"), dict) else item
            name = str(nested.get("display_name") or nested.get("name") or "").strip()
        else:
            name = ""
        if name:
            output.append(name)
    return output


def _sources(raw: dict[str, Any]) -> list[str]:
    candidates = (
        raw.get("b4_sources")
        or raw.get("fusion_sources")
        or raw.get("sources")
        or []
    )
    return [str(x) for x in candidates] if isinstance(candidates, list) else []


def _reason_tags(raw: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for key in (
        "b5_2_reason_tags",
        "b5_1_reason_tags",
        "selector_reason_tags",
        "day4_reason_tags",
        "b4_fusion_reason_tags",
    ):
        value = raw.get(key)
        if isinstance(value, list):
            tags.extend(str(x) for x in value if x)
    seen: set[str] = set()
    return [x for x in tags if not (x in seen or seen.add(x))]


def _reason_text(raw: dict[str, Any]) -> str | None:
    for key in (
        "b5_2_reason",
        "b5_1_guard_reason",
        "selector_reason",
        "b5_reason",
    ):
        value = str(raw.get(key) or "").strip()
        if value:
            return value
    tags = _reason_tags(raw)
    return "；".join(tags[:4]) if tags else None


def _constraint_evidence(raw: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    evidence = raw.get("b5_1_constraint_evidence") or raw.get("constraint_evidence")
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                output.append(dict(item))
    missing = raw.get("b5_1_missing_constraints")
    if isinstance(missing, list):
        for value in missing:
            output.append({"status": "missing", "constraint": value})
    return output


def _paper_openalex_id(paper: dict[str, Any]) -> str | None:
    value = paper.get("openalex_id")
    if value:
        return str(value).rsplit("/", 1)[-1]
    raw = _raw_meta(paper)
    value = raw.get("id") or raw.get("openalex_id")
    return str(value).rsplit("/", 1)[-1] if value else None


def _paper_score(raw: dict[str, Any]) -> float | None:
    for key in (
        "b5_2_final_score",
        "b5_1_final_score",
        "selector_score",
        "b3_final_score",
        "b4_pre_rerank_score",
        "day4_rescue_score",
        "retrieval_score",
    ):
        value = _float(raw.get(key))
        if value is not None:
            return value
    return None


class SearchService:
    def __init__(self, artifacts: ArtifactService, settings: ApiSettings) -> None:
        self.artifacts = artifacts
        self.settings = settings
        self.decomposer = ConstraintDecomposer()

    def search(self, request: SearchRequestModel) -> SearchResponse:
        started = time.perf_counter()
        if request.mode == "benchmark":
            response = self._benchmark_search(request)
        else:
            response = self._live_search(request)
        response.latency_ms = max(0, round((time.perf_counter() - started) * 1000))
        return response

    def _benchmark_search(self, request: SearchRequestModel) -> SearchResponse:
        if not request.qid:
            raise QueryNotFoundError("benchmark mode requires qid")
        queries = self.artifacts.benchmark_queries()
        query_row = queries.get(request.qid)
        if query_row is None:
            raise QueryNotFoundError(request.qid)

        canonical_question = str(query_row.get("question") or "")
        warnings: list[str] = []
        if request.query.strip() != canonical_question.strip():
            warnings.append(
                "Benchmark mode uses the qid-linked frozen result; request.query differs from the benchmark question."
            )

        prediction = self.artifacts.day4_predictions(request.top_k).get(request.qid)
        if prediction is None:
            raise QueryNotFoundError(f"No Day-4 prediction for {request.qid}")

        decomposition = self.decomposer.decompose(canonical_question)
        plan = self.artifacts.query_plans().get(request.qid) or {}
        anchors = self.artifacts.anchor_inventory().get(request.qid) or {}
        paths_by_qid = self.artifacts.day5_citation_paths()
        citation_paths = paths_by_qid.get(request.qid, []) if request.enable_citation else []
        path_by_paper: dict[str, dict[str, Any]] = {}
        for path in citation_paths:
            expanded = str(path.get("expanded_openalex_id") or "").rsplit("/", 1)[-1]
            if expanded and expanded not in path_by_paper:
                path_by_paper[expanded] = path

        papers = prediction.get("papers") if isinstance(prediction.get("papers"), list) else []
        results = [
            self._to_paper_result(
                paper,
                rank=index,
                citation_path=path_by_paper.get(_paper_openalex_id(paper) or ""),
            )
            for index, paper in enumerate(papers[: request.top_k], start=1)
            if isinstance(paper, dict)
        ]

        day4_summary = self.artifacts.day4_summary()
        day5_summary = self.artifacts.day5_summary()
        day5_seeds = self.artifacts.day5_seeds().get(request.qid) or {}
        stages = [
            {"name": "constraint_decomposition", "status": "completed", "count": len(decomposition.constraints)},
            {"name": "academic_query_planning", "status": "completed" if plan else "not_frozen", "count": len(plan.get("planned_queries") or []) if isinstance(plan, dict) else 0},
            {"name": "day4_rescue", "status": "triggered" if request.qid in self.artifacts.day4_target_qids() else "preserved"},
        ]
        if request.enable_citation:
            stages.append(
                {
                    "name": "day5_citation",
                    "status": "candidate_discovery_only" if request.qid in self.artifacts.day5_seeds() else "not_triggered",
                    "seed_count": len(day5_seeds.get("seeds") or []) if isinstance(day5_seeds, dict) else 0,
                    "path_count": len(citation_paths),
                }
            )
            if citation_paths:
                warnings.append(
                    "Day-5 citation artifacts are shown as discovery/explanation paths; final benchmark ranking remains the evaluated Day-4 Top-K."
                )

        retrieval_summary = day4_summary.get("retrieval") if isinstance(day4_summary.get("retrieval"), dict) else {}
        return SearchResponse(
            run_id=f"bench_{request.qid}_{uuid.uuid4().hex[:8]}",
            query=canonical_question,
            qid=request.qid,
            mode="benchmark",
            parsed_constraints=[item.to_dict() for item in decomposition.constraints],
            academic_anchors=self._extract_anchor_payload(plan, anchors),
            query_plan=list(plan.get("planned_queries") or []) if isinstance(plan, dict) else [],
            results=results,
            pipeline=PipelineSummary(
                stages=stages,
                total_candidates=len(papers),
                returned_results=len(results),
            ),
            cost=CostSummary(
                api_calls=0,
                cache_hits=0,
                retries=0,
                estimated_cost_usd=0.0,
            ),
            latency_ms=0,
            warnings=warnings,
        )

    def _live_search(self, request: SearchRequestModel) -> SearchResponse:
        qid = request.qid or f"live_{uuid.uuid4().hex[:10]}"
        decomposition = self.decomposer.decompose(request.query)
        planner = AcademicQueryPlanner(PlannerConfig(max_queries=self.settings.live_max_plans))
        plan = planner.plan(qid, request.query)
        planned_queries = plan.planned_queries[: self.settings.live_max_plans]
        if not planned_queries:
            raise LiveRetrievalError("Academic query planner produced no live retrieval plans")

        retriever = OpenAlexRetriever(OpenAlexConfig())
        all_papers: list[dict[str, Any]] = []
        seen: set[str] = set()
        api_calls = cache_hits = retries = 0
        estimated_cost = 0.0
        stage_rows: list[dict[str, Any]] = []

        for planned in planned_queries:
            try:
                result = retriever.search(
                    SearchRequest(query=planned.text, per_page=self.settings.live_per_plan)
                )
            except OpenAlexError as exc:
                stats = exc.stats
                if stats:
                    api_calls += stats.actual_api_calls
                    cache_hits += stats.cache_hits
                    retries += stats.retries
                    estimated_cost += stats.estimated_api_cost_usd
                stage_rows.append({"name": "openalex", "query": planned.text, "status": "failed", "error": str(exc)})
                continue

            api_calls += result.stats.actual_api_calls
            cache_hits += result.stats.cache_hits
            retries += result.stats.retries
            estimated_cost += result.stats.estimated_api_cost_usd
            stage_rows.append(
                {
                    "name": "openalex",
                    "query": planned.text,
                    "plan_type": planned.plan_type,
                    "status": "completed",
                    "raw_result_count": result.raw_result_count,
                }
            )
            for paper in result.papers:
                key = paper.canonical_id_v2
                if key in seen:
                    continue
                seen.add(key)
                payload = paper.to_dict(include_raw=True)
                payload_raw = dict(payload.get("raw") or {})
                payload_raw["live_plan_type"] = planned.plan_type
                payload_raw["live_plan_query"] = planned.text
                payload_raw["retrieval_sources"] = ["openalex_live"]
                payload["raw"] = payload_raw
                all_papers.append(payload)

        if not all_papers:
            raise LiveRetrievalError("All live OpenAlex retrieval plans failed or returned no candidates")

        record = {"qid": qid, "question": request.query, "papers": all_papers}
        reranked = annotate_and_semantic_rerank_record(
            record,
            decomposition,
            make_semantic_config("semantic_first"),
        )
        ranked_papers = list(reranked.get("papers") or [])[: min(request.top_k, self.settings.live_max_results)]
        results = [
            self._to_paper_result(paper, rank=index, citation_path=None)
            for index, paper in enumerate(ranked_papers, start=1)
            if isinstance(paper, dict)
        ]
        warnings = [
            "Live mode performs query planning, OpenAlex retrieval, deduplication and lightweight semantic reranking.",
            "Fresh live candidates do not yet have frozen Guard/Selector evidence; benchmark mode should be used for evaluated competition metrics.",
        ]
        if request.enable_citation:
            warnings.append("Online citation expansion is intentionally disabled in Day-6 live mode for latency and budget control.")

        return SearchResponse(
            run_id=f"live_{uuid.uuid4().hex[:12]}",
            query=request.query,
            qid=qid,
            mode="live",
            parsed_constraints=[item.to_dict() for item in decomposition.constraints],
            academic_anchors=[anchor.to_dict() for anchor in plan.anchors],
            query_plan=[item.to_dict() for item in planned_queries],
            results=results,
            pipeline=PipelineSummary(
                stages=[{"name": "query_planner", "status": "completed", "count": len(planned_queries)}, *stage_rows, {"name": "semantic_rerank", "status": "completed"}],
                total_candidates=len(all_papers),
                returned_results=len(results),
            ),
            cost=CostSummary(
                api_calls=api_calls,
                cache_hits=cache_hits,
                retries=retries,
                estimated_cost_usd=estimated_cost,
            ),
            latency_ms=0,
            warnings=warnings,
        )

    @staticmethod
    def _extract_anchor_payload(plan: dict[str, Any], anchors: dict[str, Any]) -> list[dict[str, Any]]:
        for source in (plan, anchors):
            value = source.get("anchors") if isinstance(source, dict) else None
            if isinstance(value, list):
                return [dict(x) for x in value if isinstance(x, dict)]
        return []

    @staticmethod
    def _to_paper_result(
        paper: dict[str, Any],
        *,
        rank: int,
        citation_path: dict[str, Any] | None,
    ) -> PaperResult:
        raw = _raw_meta(paper)
        relevance = (
            raw.get("b5_2_relevance_label")
            or raw.get("relevance_label")
            or raw.get("selector_label")
        )
        path_view = None
        if citation_path:
            path_view = CitationPathView(
                seed_openalex_id=citation_path.get("seed_openalex_id"),
                seed_title=citation_path.get("seed_title"),
                expanded_openalex_id=citation_path.get("expanded_openalex_id"),
                expanded_title=citation_path.get("expanded_title"),
                edge_type=citation_path.get("edge_type"),
                hop=citation_path.get("hop"),
            )
        return PaperResult(
            rank=rank,
            title=str(paper.get("title") or paper.get("display_name") or ""),
            authors=_authors(paper.get("authors") or paper.get("authorships")),
            year=paper.get("year") or paper.get("publication_year"),
            venue=str(paper.get("venue") or "").strip() or None,
            doi=paper.get("doi"),
            arxiv_id=paper.get("arxiv_id"),
            openalex_id=_paper_openalex_id(paper),
            url=paper.get("url"),
            score=_paper_score(raw),
            relevance_level=str(relevance) if relevance else None,
            reason_tags=_reason_tags(raw),
            reason_text=_reason_text(raw),
            constraint_evidence=_constraint_evidence(raw),
            retrieval_sources=_sources(raw) or list(raw.get("retrieval_sources") or []),
            citation_path=path_view,
        )
