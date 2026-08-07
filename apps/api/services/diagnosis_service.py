from __future__ import annotations

from typing import Any

from apps.api.services.artifact_service import ArtifactService
from apps.api.services.search_service import QueryNotFoundError
from scholarpath.query.constraints import ConstraintDecomposer


class DiagnosisService:
    def __init__(self, artifacts: ArtifactService) -> None:
        self.artifacts = artifacts
        self.decomposer = ConstraintDecomposer()

    def diagnose(self, qid: str) -> dict[str, Any]:
        query = self.artifacts.benchmark_queries().get(qid)
        if query is None:
            raise QueryNotFoundError(qid)
        question = str(query.get("question") or "")
        decomposition = self.decomposer.decompose(question)
        plan = self.artifacts.query_plans().get(qid) or {}
        anchors_record = self.artifacts.anchor_inventory().get(qid) or {}
        seeds = self.artifacts.day5_seeds().get(qid) or {}
        paths = self.artifacts.day5_citation_paths().get(qid, [])
        log = self.artifacts.day4_query_logs().get(qid) or {}
        day5_summary = self.artifacts.day5_summary()

        anchors = plan.get("anchors") if isinstance(plan.get("anchors"), list) else anchors_record.get("anchors")
        if not isinstance(anchors, list):
            anchors = []
        planned_queries = plan.get("planned_queries") if isinstance(plan.get("planned_queries"), list) else []

        return {
            "qid": qid,
            "question": question,
            "constraints": [item.to_dict() for item in decomposition.constraints],
            "subqueries": [item.to_dict() for item in decomposition.subqueries],
            "anchors": anchors,
            "query_plan": planned_queries,
            "pipeline": {
                "day4_rescue_triggered": qid in self.artifacts.day4_target_qids(),
                "day5_citation_triggered": qid in self.artifacts.day5_seeds(),
                "selected_seed_count": len(seeds.get("seeds") or []) if isinstance(seeds, dict) else 0,
                "citation_path_count": len(paths),
            },
            "retrieval_trace": {
                "day4": log,
                "citation_paths": paths[:100],
                "citation_paths_truncated": len(paths) > 100,
            },
            "cost": {
                "day5_actual_api_calls_global": int(day5_summary.get("actual_api_calls") or 0),
                "day5_cache_hits_global": int(day5_summary.get("cache_hits") or 0),
            },
            "notes": [
                "Reference answers are never returned by the diagnosis endpoint.",
                "Day-5 citation paths are discovery evidence and have not replaced the evaluated Day-4 final ranking.",
            ],
        }
