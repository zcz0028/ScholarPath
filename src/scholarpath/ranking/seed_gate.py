from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from scholarpath.paper.schema import PaperRecord


FORBIDDEN_GOLD_KEYS = {
    "gold",
    "gold_papers",
    "gold_titles",
    "gold_ids",
    "matched_gold",
}

STOPWORDS = {
    "a",
    "about",
    "all",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "for",
    "from",
    "give",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "paper",
    "papers",
    "provide",
    "research",
    "show",
    "some",
    "that",
    "the",
    "their",
    "these",
    "to",
    "use",
    "using",
    "what",
    "which",
    "with",
}


@dataclass(slots=True, frozen=True)
class SeedGateConfig:
    max_seeds_per_query: int = 3
    max_seed_rank: int = 15
    require_openalex_id: bool = True
    reject_guard_failures: bool = True
    require_anchor_match: bool = True
    min_query_overlap_terms: int = 2
    min_query_overlap_score: float = 0.15

    def __post_init__(self) -> None:
        if self.max_seeds_per_query < 1:
            raise ValueError("max_seeds_per_query must be positive")
        if self.max_seed_rank < 1:
            raise ValueError("max_seed_rank must be positive")
        if self.min_query_overlap_terms < 1:
            raise ValueError("min_query_overlap_terms must be positive")
        if not 0.0 <= self.min_query_overlap_score <= 1.0:
            raise ValueError("min_query_overlap_score must be within [0, 1]")


@dataclass(slots=True)
class SeedDecision:
    qid: str
    paper: PaperRecord
    rank: int
    accepted: bool
    seed_score: float
    reasons: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "paper": self.paper.to_dict(include_raw=True),
            "rank": self.rank,
            "accepted": self.accepted,
            "seed_score": self.seed_score,
            "reasons": list(self.reasons),
            "rejection_reasons": list(self.rejection_reasons),
        }


def _raw(paper: Mapping[str, Any]) -> Mapping[str, Any]:
    value = paper.get("raw")
    return value if isinstance(value, Mapping) else {}


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _contains_gold(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).casefold().strip()
            if normalized in FORBIDDEN_GOLD_KEYS or normalized.startswith("gold_"):
                return True
            if _contains_gold(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_gold(item) for item in value)
    return False


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9.+_-]*", text.casefold())
        if len(token) >= 3 and token not in STOPWORDS
    }


def _query_overlap(
    question: str,
    paper: PaperRecord,
) -> tuple[float, list[str]]:
    query_tokens = _tokens(question)
    paper_tokens = _tokens(" ".join(filter(None, [paper.title, paper.abstract or ""])))
    if not query_tokens or not paper_tokens:
        return 0.0, []

    matched = sorted(query_tokens & paper_tokens)
    denominator = max(1, min(len(query_tokens), 8))
    return min(1.0, len(matched) / denominator), matched


def _metadata_anchor_signal(raw: Mapping[str, Any]) -> float:
    values = [
        _safe_float(raw.get("constraint_coverage")),
        _safe_float(raw.get("day4_rescue_score")),
    ]
    anchor_texts = raw.get("day4_anchor_texts")
    if isinstance(anchor_texts, list) and anchor_texts:
        values.append(min(1.0, 0.35 + 0.12 * len(anchor_texts)))
    return max(values, default=0.0)


def _selector_signal(raw: Mapping[str, Any]) -> float:
    return max(
        _safe_float(raw.get("b5_selector_score")),
        _safe_float(raw.get("b5_2_selector_score")),
        _safe_float(raw.get("selector_score")),
        0.0,
    )


def _semantic_signal(raw: Mapping[str, Any], rank: int) -> float:
    explicit = max(
        _safe_float(raw.get("semantic_score")),
        _safe_float(raw.get("b3_semantic_score")),
        _safe_float(raw.get("final_score")),
        0.0,
    )
    rank_signal = 1.0 / (1.0 + 0.12 * max(0, rank - 1))
    return max(explicit, rank_signal)


def evaluate_seed(
    *,
    qid: str,
    paper_mapping: Mapping[str, Any],
    rank: int,
    config: SeedGateConfig,
    question: str = "",
) -> SeedDecision:
    if _contains_gold(paper_mapping):
        raise ValueError("Production seed selection cannot consume gold fields")

    paper = PaperRecord.from_mapping(paper_mapping)
    raw = _raw(paper_mapping)
    rejection_reasons: list[str] = []
    reasons: list[str] = []

    if rank > config.max_seed_rank:
        rejection_reasons.append("rank_outside_gate")
    if config.require_openalex_id and not paper.openalex_id:
        rejection_reasons.append("missing_openalex_id")

    guard_decision = str(
        raw.get("b5_2_guard_decision")
        or raw.get("b5_1_guard_decision")
        or "pass"
    ).casefold()
    if config.reject_guard_failures and guard_decision == "reject":
        rejection_reasons.append("guard_reject")

    lexical_score, matched_terms = _query_overlap(question, paper)
    metadata_anchor_score = _metadata_anchor_signal(raw)

    # Production calls pass a non-empty question. In that case, a real query/title
    # or query/abstract overlap is mandatory. The metadata score may strengthen a
    # valid match, but cannot independently prove anchor relevance.
    if config.require_anchor_match:
        if question.strip():
            if len(matched_terms) < config.min_query_overlap_terms:
                rejection_reasons.append(
                    "insufficient_query_term_coverage"
                )
            elif lexical_score < config.min_query_overlap_score:
                rejection_reasons.append(
                    "low_query_overlap_score"
                )
        elif metadata_anchor_score <= 0.0:
            # Backward-compatible fallback for unit tests and legacy callers.
            rejection_reasons.append("missing_anchor_signal")

    semantic_score = _semantic_signal(raw, rank)
    selector_score = _selector_signal(raw)
    multi_plan = int(raw.get("day4_occurrence_count") or 1) >= 2
    source_names = raw.get("fusion_sources") or raw.get("sources") or []
    multi_source = (
        isinstance(source_names, list)
        and len(set(map(str, source_names))) >= 2
    )

    # Metadata-based anchor evidence is capped so it cannot dominate a weak or
    # irrelevant lexical match.
    anchor_score = max(lexical_score, min(0.50, metadata_anchor_score))
    seed_score = (
        0.45 * min(1.0, semantic_score)
        + 0.25 * min(1.0, anchor_score)
        + 0.20 * min(1.0, selector_score)
        + 0.05 * float(multi_plan)
        + 0.05 * float(multi_source)
    )

    if paper.openalex_id:
        reasons.append("openalex_resolvable")
    if matched_terms:
        reasons.append("query_terms:" + ",".join(matched_terms[:6]))
        reasons.append(f"query_term_coverage:{len(matched_terms)}")
    if metadata_anchor_score > 0.0:
        reasons.append("metadata_anchor_support")
    if guard_decision in {"pass", "soft_pass"}:
        reasons.append(f"guard_{guard_decision}")
    if multi_plan:
        reasons.append("multi_plan_support")
    if multi_source:
        reasons.append("multi_source_support")

    return SeedDecision(
        qid=qid,
        paper=paper,
        rank=rank,
        accepted=not rejection_reasons,
        seed_score=max(0.0, min(1.0, seed_score)),
        reasons=reasons,
        rejection_reasons=rejection_reasons,
    )


def select_seed_papers(
    *,
    qid: str,
    papers: Sequence[Mapping[str, Any]],
    config: SeedGateConfig,
    question: str = "",
) -> tuple[list[SeedDecision], list[SeedDecision]]:
    decisions = [
        evaluate_seed(
            qid=qid,
            question=question,
            paper_mapping=paper,
            rank=rank,
            config=config,
        )
        for rank, paper in enumerate(papers, start=1)
        if isinstance(paper, Mapping)
    ]

    accepted = sorted(
        (item for item in decisions if item.accepted),
        key=lambda item: (
            -item.seed_score,
            item.rank,
            item.paper.title.casefold(),
        ),
    )[: config.max_seeds_per_query]

    accepted_ids = {id(item) for item in accepted}
    rejected = [
        item
        for item in decisions
        if not item.accepted or id(item) not in accepted_ids
    ]
    for item in rejected:
        if item.accepted and id(item) not in accepted_ids:
            item.accepted = False
            item.rejection_reasons.append("seed_budget_exceeded")

    return accepted, rejected
