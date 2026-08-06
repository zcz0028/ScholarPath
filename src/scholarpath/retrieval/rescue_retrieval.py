from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from scholarpath.paper.schema import PaperRecord
from scholarpath.rerank.constraint_rerank import paper_key


FORBIDDEN_GOLD_KEYS = {
    "gold",
    "gold_papers",
    "gold_titles",
    "gold_ids",
    "gold_dois",
    "matched_gold",
}


@dataclass(slots=True, frozen=True)
class RescuePlanSpec:
    text: str
    plan_type: str
    priority: int = 50
    confidence: float = 0.5
    reason: str = ""
    anchor_texts: tuple[str, ...] = ()
    anchor_types: tuple[str, ...] = ()
    estimated_api_calls: int = 1

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RescuePlanSpec":
        text = str(value.get("text") or "").strip()
        if not text:
            raise ValueError("Rescue plan text must be non-empty.")
        priority = max(0, min(100, int(value.get("priority", 50))))
        confidence = max(0.0, min(1.0, float(value.get("confidence", 0.5))))
        estimated_calls = max(0, int(value.get("estimated_api_calls", 1)))
        return cls(
            text=text,
            plan_type=str(value.get("plan_type") or "anchor_query").strip() or "anchor_query",
            priority=priority,
            confidence=confidence,
            reason=str(value.get("reason") or "").strip(),
            anchor_texts=tuple(
                str(item).strip()
                for item in value.get("anchor_texts", [])
                if str(item).strip()
            ),
            anchor_types=tuple(
                str(item).strip()
                for item in value.get("anchor_types", [])
                if str(item).strip()
            ),
            estimated_api_calls=estimated_calls,
        )

    @property
    def weight(self) -> float:
        # Priority and confidence are both bounded, so the final channel weight
        # remains interpretable and does not overwhelm rank evidence.
        return 0.55 + 0.35 * (self.priority / 100.0) + 0.10 * self.confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "plan_type": self.plan_type,
            "priority": self.priority,
            "confidence": self.confidence,
            "reason": self.reason,
            "anchor_texts": list(self.anchor_texts),
            "anchor_types": list(self.anchor_types),
            "estimated_api_calls": self.estimated_api_calls,
            "weight": self.weight,
        }


@dataclass(slots=True)
class RescueCandidate:
    paper: PaperRecord
    score: float
    best_rank: int
    occurrence_count: int = 1
    plan_hits: list[dict[str, Any]] = field(default_factory=list)
    plan_types: set[str] = field(default_factory=set)
    anchor_texts: set[str] = field(default_factory=set)

    def add_hit(self, paper: PaperRecord, rank: int, plan: RescuePlanSpec) -> None:
        safe_rank = max(1, int(rank))
        self.score += rescue_rank_score(safe_rank, plan)
        self.best_rank = min(self.best_rank, safe_rank)
        self.occurrence_count += 1
        self.plan_types.add(plan.plan_type)
        self.anchor_texts.update(plan.anchor_texts)
        self.plan_hits.append(_plan_hit(plan, safe_rank))
        _merge_paper_record(self.paper, paper)


def _contains_forbidden_gold_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().casefold()
            if normalized in FORBIDDEN_GOLD_KEYS or normalized.startswith("gold_"):
                return True
            if _contains_forbidden_gold_key(nested):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_gold_key(item) for item in value)
    return False


def validate_query_plan_records(
    records: Sequence[Mapping[str, Any]],
    *,
    max_plans_per_query: int = 2,
    max_api_calls: int | None = None,
) -> dict[str, Any]:
    if max_plans_per_query < 1:
        raise ValueError("max_plans_per_query must be positive.")

    qids: set[str] = set()
    plan_count = 0
    estimated_api_calls = 0

    for index, record in enumerate(records, start=1):
        if _contains_forbidden_gold_key(record):
            raise ValueError(f"Query plan record {index} contains forbidden gold fields.")

        qid = str(record.get("qid") or "").strip()
        if not qid:
            raise ValueError(f"Query plan record {index} is missing qid.")
        if qid in qids:
            raise ValueError(f"Duplicate query plan qid: {qid}")
        qids.add(qid)

        plans = record.get("planned_queries")
        if not isinstance(plans, list) or not plans:
            raise ValueError(f"Query plan {qid} has no planned_queries.")
        if len(plans) > max_plans_per_query:
            raise ValueError(
                f"Query plan {qid} contains {len(plans)} plans; "
                f"maximum is {max_plans_per_query}."
            )

        for plan_value in plans:
            if not isinstance(plan_value, Mapping):
                raise ValueError(f"Query plan {qid} contains a non-object plan.")
            plan = RescuePlanSpec.from_mapping(plan_value)
            plan_count += 1
            estimated_api_calls += plan.estimated_api_calls

    if max_api_calls is not None and estimated_api_calls > int(max_api_calls):
        raise ValueError(
            f"Estimated API calls {estimated_api_calls} exceed budget {max_api_calls}."
        )

    return {
        "query_count": len(qids),
        "plan_count": plan_count,
        "estimated_api_calls": estimated_api_calls,
        "qids": sorted(qids),
        "production_retrieval_uses_gold": False,
    }


def rescue_rank_score(rank: int, plan: RescuePlanSpec) -> float:
    safe_rank = max(1, int(rank))
    return plan.weight / math.log2(safe_rank + 2)


def _plan_hit(plan: RescuePlanSpec, rank: int) -> dict[str, Any]:
    return {
        "query": plan.text,
        "plan_type": plan.plan_type,
        "priority": plan.priority,
        "confidence": plan.confidence,
        "rank": rank,
        "weight": plan.weight,
        "reason": plan.reason,
        "anchor_texts": list(plan.anchor_texts),
        "anchor_types": list(plan.anchor_types),
    }


def _merge_paper_record(base: PaperRecord, incoming: PaperRecord) -> None:
    for field_name in (
        "doi",
        "arxiv_id",
        "openalex_id",
        "semantic_scholar_id",
        "url",
        "publication_date",
        "venue",
        "abstract",
    ):
        if not getattr(base, field_name) and getattr(incoming, field_name):
            setattr(base, field_name, getattr(incoming, field_name))
    if len(incoming.authors) > len(base.authors):
        base.authors = list(incoming.authors)
    if base.year is None and incoming.year is not None:
        base.year = incoming.year
    if base.citation_count is None and incoming.citation_count is not None:
        base.citation_count = incoming.citation_count
    merged_raw = dict(base.raw or {})
    for key, value in (incoming.raw or {}).items():
        if key not in merged_raw or merged_raw[key] in (None, "", [], {}):
            merged_raw[key] = value
    base.raw = merged_raw


def aggregate_rescue_hits(
    hits: Iterable[tuple[PaperRecord, int, RescuePlanSpec]],
) -> list[PaperRecord]:
    candidates: dict[str, RescueCandidate] = {}

    for paper, rank, plan in hits:
        safe_rank = max(1, int(rank))
        key = paper_key(paper.to_dict(include_raw=True))
        if key not in candidates:
            candidates[key] = RescueCandidate(
                paper=paper,
                score=rescue_rank_score(safe_rank, plan),
                best_rank=safe_rank,
                occurrence_count=1,
                plan_hits=[_plan_hit(plan, safe_rank)],
                plan_types={plan.plan_type},
                anchor_texts=set(plan.anchor_texts),
            )
        else:
            candidates[key].add_hit(paper, safe_rank, plan)

    for item in candidates.values():
        if item.occurrence_count >= 2:
            item.score += min(0.20, 0.08 * (item.occurrence_count - 1))
        if item.paper.identity_keys_v2():
            item.score += 0.02

    ordered = sorted(
        candidates.values(),
        key=lambda item: (
            -item.score,
            -item.occurrence_count,
            item.best_rank,
            item.paper.title.casefold(),
        ),
    )

    output: list[PaperRecord] = []
    for item in ordered:
        raw = dict(item.paper.raw or {})
        raw.update(
            {
                "day4_rescue_score": item.score,
                "day4_occurrence_count": item.occurrence_count,
                "day4_best_rank": item.best_rank,
                "day4_plan_types": sorted(item.plan_types),
                "day4_anchor_texts": sorted(item.anchor_texts),
                "day4_plan_hits": item.plan_hits,
                "day4_reason_tags": _reason_tags(item),
            }
        )
        item.paper.raw = raw
        output.append(item.paper)
    return output


def _reason_tags(item: RescueCandidate) -> list[str]:
    tags = ["anchor_rescue_source"]
    if item.occurrence_count >= 2:
        tags.append("multi_plan_supported")
    if item.best_rank <= 10:
        tags.append("high_rank_rescue_hit")
    elif item.best_rank <= 20:
        tags.append("top20_rescue_hit")
    if item.paper.identity_keys_v2():
        tags.append("strong_identifier_available")
    return tags


def build_output_record(
    *,
    qid: str,
    question: str,
    papers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "qid": qid,
        "question": question,
        "papers": [dict(item) for item in papers],
    }


def preserve_or_replace_record(
    *,
    qid: str,
    target_qids: set[str],
    baseline_record: Mapping[str, Any],
    replacement_record: Mapping[str, Any] | None,
    top_k: int,
) -> dict[str, Any]:
    if qid not in target_qids or replacement_record is None:
        papers = baseline_record.get("papers")
        return {
            "qid": baseline_record.get("qid", qid),
            "question": baseline_record.get("question"),
            "papers": list(papers) if isinstance(papers, list) else [],
        }

    papers = replacement_record.get("papers")
    return {
        "qid": replacement_record.get("qid", qid),
        "question": replacement_record.get("question"),
        "papers": (list(papers) if isinstance(papers, list) else [])[:top_k],
    }
