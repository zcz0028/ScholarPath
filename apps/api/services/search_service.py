from __future__ import annotations

import time
import uuid
from typing import Any

from apps.api.schemas import (
    CitationPathView,
    CostSummary,
    PaperResult,
    PipelineSummary,
    SearchReasoning,
    SearchRequestModel,
    SearchResponse,
)
from apps.api.services.artifact_service import ArtifactService
from apps.api.settings import ApiSettings

from scholarpath.query.academic_query_planner import AcademicQueryPlanner, PlannerConfig
from scholarpath.query.constraints import ConstraintDecomposer
from scholarpath.rerank.constraint_evidence import (
    ConstraintEvidence,
    build_canonical_constraints,
)
from scholarpath.rerank.evidence_aware_rerank import (
    EvidenceAwareConfig,
    annotate_and_evidence_rerank_record,
)
from scholarpath.rerank.recommendation_reason import attach_recommendation_reason
from scholarpath.retrieval.base import SearchRequest
from scholarpath.retrieval.openalex import OpenAlexConfig, OpenAlexError, OpenAlexRetriever
from scholarpath.retrieval.citation_path_builder import (
    build_citation_path,
)
from scholarpath.retrieval.citation_resolver import (
    resolve_citation_path,
)

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
    day8 = raw.get("day8_reason_tags")
    if isinstance(day8, list):
        return [str(x) for x in day8 if x]

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
    day8 = str(raw.get("day8_reason_text") or "").strip()
    if day8:
        return day8

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
    return "，".join(tags[:4]) if tags else None


def _constraint_evidence(raw: dict[str, Any]) -> list[dict[str, Any]]:
    # Day8 evidence is the production truth layer. Do not mix legacy Guard
    # missing-constraint rows into it, otherwise the API may expose duplicate or
    # contradictory evidence for the same paper.
    day8 = raw.get("day8_constraint_evidence")
    if isinstance(day8, list):
        return [dict(item) for item in day8 if isinstance(item, dict)]

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
        "day8_final_score",
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


def _typed_day8_evidence(raw: dict[str, Any]) -> list[ConstraintEvidence]:
    rows = raw.get("day8_constraint_evidence")
    if not isinstance(rows, list):
        return []

    output: list[ConstraintEvidence] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        try:
            output.append(
                ConstraintEvidence(
                    constraint_id=str(item.get("constraint_id") or ""),
                    constraint_text=str(item.get("constraint_text") or ""),
                    constraint_type=str(item.get("constraint_type") or "topic"),
                    matched=bool(item.get("matched")),
                    match_type=str(item.get("match_type") or "none"),
                    evidence_field=(
                        str(item.get("evidence_field"))
                        if item.get("evidence_field") is not None
                        else None
                    ),
                    evidence_text=(
                        str(item.get("evidence_text"))
                        if item.get("evidence_text") is not None
                        else None
                    ),
                    confidence=float(item.get("confidence") or 0.0),
                    token_coverage=float(item.get("token_coverage") or 0.0),
                )
            )
        except (TypeError, ValueError):
            continue
    return output


def _apply_day8_stack(
    record: dict[str, Any],
    decomposition: Any,
) -> dict[str, Any]:
    """Apply the frozen Day8 production stack: E3 ranking + evidence + reason.

    Ranking is fixed to E3 alpha=1.0 / beta=0.0 from the accepted Day8-1C
    ablation. Constraint evidence remains an explanation signal and is not
    linearly fused into the production ranking score.
    """
    reranked = annotate_and_evidence_rerank_record(
        record,
        decomposition,
        EvidenceAwareConfig(variant="E3", alpha=1.0),
    )
    question = str(reranked.get("question") or decomposition.question or "")
    canonical = build_canonical_constraints(
        question=question,
        constraints=decomposition.constraints,
    )

    annotated: list[dict[str, Any]] = []
    for paper in reranked.get("papers") or []:
        if not isinstance(paper, dict):
            continue
        evidence = _typed_day8_evidence(_raw_meta(paper))
        annotated.append(
            attach_recommendation_reason(
                paper,
                constraints=canonical,
                evidence=evidence,
            )
        )

    output = dict(reranked)
    output["papers"] = annotated
    return output


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

        # E3 was evaluated on the frozen Day-4 Top100 candidate pool. Always
        # rerank that same pool, then slice to the requested top_k so API output
        # matches the accepted offline experiment.
        prediction = self.artifacts.day4_predictions(100).get(request.qid)
        if prediction is None:
            raise QueryNotFoundError(f"No Day-4 Top100 prediction for {request.qid}")

        decomposition = self.decomposer.decompose(canonical_question)
        day8_record = _apply_day8_stack(dict(prediction), decomposition)
        plan = self.artifacts.query_plans().get(request.qid) or {}
        anchors = self.artifacts.anchor_inventory().get(request.qid) or {}
        day9_path_by_paper = (
            self.artifacts
            .day9_citation_paths_by_paper(
                request.qid
            )
        )

        day5_path_by_paper = (
            self.artifacts
            .citation_paths_by_paper(
                request.qid
            )
        )

        path_by_paper = {
            **day5_path_by_paper,
            **day9_path_by_paper,
        }
        papers = day8_record.get("papers") if isinstance(day8_record.get("papers"), list) else []
        results = [
            self._to_paper_result(
                paper,
                rank=index,
                citation_path=resolve_citation_path(
                    paper=paper,
                    qid=request.qid,
                    seed_rank=index,
                    artifact_paths=path_by_paper,
                    fallback_builder=None,
                ),
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
            {"name": "day8_e3_rerank", "status": "completed", "variant": "E3", "alpha": 1.0, "beta": 0.0},
            {"name": "day8_constraint_evidence", "status": "completed"},
            {"name": "day8_recommendation_reason", "status": "completed"},
        ]
        if request.enable_citation:
            stages.append(
                {
                    "name": "day5_citation",
                    "status": "candidate_discovery_only"
                    if request.qid in self.artifacts.day5_seeds()
                    else "not_triggered",
                    "seed_count": len(day5_seeds.get("seeds") or [])
                    if isinstance(day5_seeds, dict)
                    else 0,
                    "path_count": len(path_by_paper),
                }
            )
            if path_by_paper:
                warnings.append(
                    "Day-5 citation artifacts are shown as discovery/explanation paths and do not alter the Day8 E3 benchmark ranking."
                )

        # Day10-1B: expose a structured, user-facing search reasoning trace.
        # This is an additive explanation layer only: it observes existing
        # benchmark artifacts and execution metadata and does not alter ranking.
        reasoning_constraints = [item.to_dict() for item in decomposition.constraints]
        reasoning_anchors = self._extract_anchor_payload(plan, anchors)
        reasoning_selected_plans = (
            list(plan.get("planned_queries") or [])
            if isinstance(plan, dict)
            else []
        )
        reasoning = SearchReasoning(
            original_query=canonical_question,
            cleaned_query=decomposition.cleaned_question,
            constraints=reasoning_constraints,
            candidate_subqueries=[item.to_dict() for item in decomposition.subqueries],
            academic_anchors=reasoning_anchors,
            derived_aliases=(
                [str(item) for item in plan.get("derived_aliases") or []]
                if isinstance(plan, dict)
                else []
            ),
            filters=(
                dict(plan.get("filters") or {})
                if isinstance(plan, dict)
                and isinstance(plan.get("filters"), dict)
                else {}
            ),
            selected_plans=reasoning_selected_plans,
            execution=[dict(stage) for stage in stages],
        )

        retrieval_summary = day4_summary.get("retrieval") if isinstance(day4_summary.get("retrieval"), dict) else {}
        return SearchResponse(
            run_id=f"bench_{request.qid}_{uuid.uuid4().hex[:8]}",
            query=canonical_question,
            qid=request.qid,
            mode="benchmark",
            reasoning=reasoning,
            parsed_constraints=reasoning_constraints,
            academic_anchors=reasoning_anchors,
            query_plan=reasoning_selected_plans,
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
        reranked = _apply_day8_stack(record, decomposition)
        ranked_papers = list(reranked.get("papers") or [])[: min(request.top_k, self.settings.live_max_results)]
      
        results = [
            self._to_paper_result(
                paper,
                rank=index,
                citation_path=build_citation_path(
                    paper=paper,
                    qid=qid,
                    seed_rank=index,
                ),
            )
            for index, paper in enumerate(ranked_papers, start=1)
            if isinstance(paper, dict)
        ]
        warnings = [
            "Live mode performs query planning, OpenAlex retrieval, deduplication, Day8 E3 pure-semantic reranking and deterministic evidence/reason generation.",
            "Fresh live candidates do not have frozen Guard/Selector labels; benchmark mode should be used for evaluated competition metrics.",
        ]
        if request.enable_citation:
            warnings.append("Online citation expansion is intentionally disabled in Day-6 live mode for latency and budget control.")

        reasoning_constraints = [item.to_dict() for item in decomposition.constraints]
        reasoning_anchors = [anchor.to_dict() for anchor in plan.anchors]
        reasoning_selected_plans = [item.to_dict() for item in planned_queries]
        reasoning_execution = [
            {"name": "query_planner", "status": "completed", "count": len(planned_queries)},
            *stage_rows,
            {"name": "day8_e3_rerank", "status": "completed", "variant": "E3", "alpha": 1.0, "beta": 0.0},
            {"name": "day8_constraint_evidence", "status": "completed"},
            {"name": "day8_recommendation_reason", "status": "completed"},
        ]
        reasoning = SearchReasoning(
            original_query=request.query,
            cleaned_query=decomposition.cleaned_question,
            constraints=reasoning_constraints,
            candidate_subqueries=[item.to_dict() for item in decomposition.subqueries],
            academic_anchors=reasoning_anchors,
            derived_aliases=list(plan.derived_aliases),
            filters=dict(plan.filters),
            selected_plans=reasoning_selected_plans,
            execution=[dict(stage) for stage in reasoning_execution],
        )

        return SearchResponse(
            run_id=f"live_{uuid.uuid4().hex[:12]}",
            query=request.query,
            qid=qid,
            mode="live",
            reasoning=reasoning,
            parsed_constraints=reasoning_constraints,
            academic_anchors=reasoning_anchors,
            query_plan=reasoning_selected_plans,
            results=results,
            pipeline=PipelineSummary(
                stages=reasoning_execution,
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
            matched_constraints=[str(x) for x in raw.get("day8_matched_constraints") or []],
            unmatched_constraints=[str(x) for x in raw.get("day8_unmatched_constraints") or []],
            matched_count=int(raw.get("day8_matched_count") or 0),
            constraint_count=int(raw.get("day8_constraint_count") or 0),
            retrieval_sources=_sources(raw) or list(raw.get("retrieval_sources") or []),
            citation_path=path_view,
        )
