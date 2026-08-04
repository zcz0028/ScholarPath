from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class SelectorConfig:
    mode: str
    semantic_weight: float
    rank_weight: float
    b2_weight: float
    core_weight: float
    title_weight: float
    high_threshold: float
    partial_threshold: float
    weak_threshold: float
    min_keep_per_query: int
    max_output_per_query: int


MODE_PRESETS = {
    "ranking": {
        "semantic_weight": 0.40,
        "rank_weight": 0.30,
        "b2_weight": 0.15,
        "core_weight": 0.10,
        "title_weight": 0.05,
        "high_threshold": 0.70,
        "partial_threshold": 0.48,
        "weak_threshold": 0.28,
        "min_keep_per_query": 100,
        "max_output_per_query": 100,
    },
    "balanced": {
        "semantic_weight": 0.45,
        "rank_weight": 0.25,
        "b2_weight": 0.15,
        "core_weight": 0.10,
        "title_weight": 0.05,
        "high_threshold": 0.68,
        "partial_threshold": 0.44,
        "weak_threshold": 0.26,
        "min_keep_per_query": 20,
        "max_output_per_query": 100,
    },
    "precision": {
        "semantic_weight": 0.52,
        "rank_weight": 0.18,
        "b2_weight": 0.14,
        "core_weight": 0.11,
        "title_weight": 0.05,
        "high_threshold": 0.72,
        "partial_threshold": 0.52,
        "weak_threshold": 0.34,
        "min_keep_per_query": 8,
        "max_output_per_query": 60,
    },
    "recall_safe": {
        "semantic_weight": 0.38,
        "rank_weight": 0.36,
        "b2_weight": 0.14,
        "core_weight": 0.08,
        "title_weight": 0.04,
        "high_threshold": 0.70,
        "partial_threshold": 0.46,
        "weak_threshold": 0.22,
        "min_keep_per_query": 50,
        "max_output_per_query": 100,
    },
}


def make_selector_config(mode: str) -> SelectorConfig:
    if mode not in MODE_PRESETS:
        raise ValueError(f"Unknown selector mode: {mode}")
    return SelectorConfig(mode=mode, **MODE_PRESETS[mode])


def normalize_text(value: object | None) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.casefold()
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212-]", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def raw_meta(paper: dict[str, Any]) -> dict[str, Any]:
    raw = paper.get("raw")
    return raw if isinstance(raw, dict) else {}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def b3_features(paper: dict[str, Any]) -> dict[str, Any]:
    features = raw_meta(paper).get("b3_features")
    return features if isinstance(features, dict) else {}


def rank_score(paper: dict[str, Any]) -> float:
    raw = raw_meta(paper)
    rank = safe_int(raw.get("b3_original_rank") or raw.get("b2_1_original_rank"), 999)
    if rank <= 0:
        rank = 999
    return 1.0 / (1.0 + (rank / 10.0))


def semantic_evidence_score(paper: dict[str, Any]) -> float:
    raw = raw_meta(paper)
    features = b3_features(paper)
    semantic = safe_float(raw.get("b3_semantic_score"), safe_float(features.get("semantic_score"), 0.0))
    title = safe_float(features.get("title_overlap"), 0.0)
    expanded = safe_float(features.get("expanded_query_overlap"), 0.0)
    abstract = safe_float(features.get("abstract_signal"), 0.0)
    phrase = safe_float(features.get("constraint_phrase_coverage"), 0.0)

    return clamp(
        0.42 * semantic
        + 0.22 * title
        + 0.16 * expanded
        + 0.10 * abstract
        + 0.10 * phrase
    )


def b2_constraint_score(paper: dict[str, Any]) -> float:
    raw = raw_meta(paper)
    b2_score = safe_float(raw.get("b2_1_constraint_score"), 0.0)
    coverage_count = safe_int(raw.get("b2_1_coverage_count"), 0)
    coverage_weight = safe_float(raw.get("b2_1_coverage_weight"), 0.0)

    # b2_score was not normalized; combine it with bounded coverage signals.
    return clamp(0.55 * b2_score + 0.25 * min(1.0, coverage_count / 4.0) + 0.20 * min(1.0, coverage_weight / 5.0))


def core_coverage_score(paper: dict[str, Any]) -> float:
    raw = raw_meta(paper)
    features = b3_features(paper)
    b2_core = safe_int(raw.get("b2_1_core_coverage_count"), 0)
    b3_core = safe_float(features.get("core_constraint_coverage"), 0.0)
    return clamp(0.60 * min(1.0, b2_core / 2.0) + 0.40 * b3_core)


def title_match_score(paper: dict[str, Any]) -> float:
    features = b3_features(paper)
    return clamp(safe_float(features.get("title_overlap"), 0.0))


def has_strong_identifier(paper: dict[str, Any]) -> bool:
    return any(
        str(paper.get(field) or "").strip()
        for field in ("doi", "arxiv_id", "openalex_id", "semantic_scholar_id")
    )


def compute_selector_score(paper: dict[str, Any], config: SelectorConfig) -> dict[str, Any]:
    semantic = semantic_evidence_score(paper)
    rank = rank_score(paper)
    b2 = b2_constraint_score(paper)
    core = core_coverage_score(paper)
    title = title_match_score(paper)

    score = (
        config.semantic_weight * semantic
        + config.rank_weight * rank
        + config.b2_weight * b2
        + config.core_weight * core
        + config.title_weight * title
    )

    if has_strong_identifier(paper):
        score += 0.015

    # Penalize papers with little evidence despite high original rank.
    if semantic < 0.04 and core == 0:
        score -= 0.06
    elif semantic < 0.08 and core == 0:
        score -= 0.03

    score = clamp(score)

    if score >= config.high_threshold:
        label = "highly_relevant"
    elif score >= config.partial_threshold:
        label = "partially_relevant"
    elif score >= config.weak_threshold:
        label = "weakly_relevant"
    else:
        label = "low_relevance"

    return {
        "selector_score": score,
        "semantic_evidence": semantic,
        "rank_evidence": rank,
        "b2_constraint_evidence": b2,
        "core_constraint_evidence": core,
        "title_evidence": title,
        "relevance_label": label,
    }


def build_reason_tags(paper: dict[str, Any], evidence: dict[str, Any]) -> tuple[list[str], list[str]]:
    tags: list[str] = []
    missing: list[str] = []

    if evidence["semantic_evidence"] >= 0.18:
        tags.append("semantic_score_high")
    elif evidence["semantic_evidence"] >= 0.10:
        tags.append("semantic_score_medium")
    else:
        missing.append("low_semantic_evidence")

    if evidence["title_evidence"] >= 0.20:
        tags.append("title_matches_query")
    elif evidence["title_evidence"] <= 0.04:
        missing.append("weak_title_match")

    if evidence["core_constraint_evidence"] >= 0.50:
        tags.append("core_constraints_covered")
    elif evidence["core_constraint_evidence"] > 0:
        tags.append("partial_core_constraint_coverage")
    else:
        missing.append("missing_core_constraints")

    if evidence["b2_constraint_evidence"] >= 0.30:
        tags.append("constraint_coverage_supported")

    if evidence["rank_evidence"] >= 0.40:
        tags.append("ranked_high_by_previous_reranker")

    if has_strong_identifier(paper):
        tags.append("strong_identifier_available")

    return tags, missing


def build_reason_text(label: str, tags: list[str], missing: list[str]) -> str:
    label_text = {
        "highly_relevant": "高度相关",
        "partially_relevant": "部分相关",
        "weakly_relevant": "弱相关",
        "low_relevance": "低相关",
    }.get(label, label)

    positive_map = {
        "semantic_score_high": "语义相似度较高",
        "semantic_score_medium": "语义相似度中等",
        "title_matches_query": "标题与查询关键词/短语匹配",
        "core_constraints_covered": "覆盖核心约束",
        "partial_core_constraint_coverage": "覆盖部分核心约束",
        "constraint_coverage_supported": "约束覆盖分提供支持",
        "ranked_high_by_previous_reranker": "前序Reranker排序较靠前",
        "strong_identifier_available": "具有强标识符",
    }
    missing_map = {
        "low_semantic_evidence": "语义证据偏弱",
        "weak_title_match": "标题匹配较弱",
        "missing_core_constraints": "核心约束覆盖不足",
    }

    positives = [positive_map.get(item, item) for item in tags[:4]]
    negatives = [missing_map.get(item, item) for item in missing[:2]]

    if positives and negatives:
        return f"{label_text}：{'; '.join(positives)}；但{'; '.join(negatives)}。"
    if positives:
        return f"{label_text}：{'; '.join(positives)}。"
    if negatives:
        return f"{label_text}：{'; '.join(negatives)}。"
    return f"{label_text}：证据不足。"


def annotate_paper(paper: dict[str, Any], config: SelectorConfig) -> dict[str, Any]:
    evidence = compute_selector_score(paper, config)
    tags, missing = build_reason_tags(paper, evidence)
    label = evidence["relevance_label"]
    reason_text = build_reason_text(label, tags, missing)

    annotated = dict(paper)
    raw = dict(raw_meta(annotated))
    raw["b5_selector_mode"] = config.mode
    raw["b5_selector_score"] = evidence["selector_score"]
    raw["b5_relevance_label"] = label
    raw["b5_evidence"] = {
        key: value for key, value in evidence.items() if key != "relevance_label"
    }
    raw["b5_reason_tags"] = tags
    raw["b5_missing_tags"] = missing
    raw["b5_reason_text"] = reason_text
    annotated["raw"] = raw
    return annotated


def select_papers(papers: list[dict[str, Any]], config: SelectorConfig) -> list[dict[str, Any]]:
    annotated = [annotate_paper(paper, config) for paper in papers if isinstance(paper, dict)]

    annotated.sort(
        key=lambda item: (
            -safe_float(raw_meta(item).get("b5_selector_score"), 0.0),
            -safe_float(raw_meta(item).get("b3_final_score"), 0.0),
            safe_int(raw_meta(item).get("b3_original_rank"), 999),
            normalize_text(item.get("title")),
        )
    )

    if config.mode == "ranking":
        return annotated[: config.max_output_per_query]

    if config.mode == "precision":
        selected = [
            item for item in annotated
            if raw_meta(item).get("b5_relevance_label") in {"highly_relevant", "partially_relevant"}
        ]
    elif config.mode == "balanced":
        selected = [
            item for item in annotated
            if raw_meta(item).get("b5_relevance_label") in {"highly_relevant", "partially_relevant", "weakly_relevant"}
        ]
    elif config.mode == "recall_safe":
        selected = [
            item for item in annotated
            if raw_meta(item).get("b5_relevance_label") != "low_relevance"
        ]
    else:
        selected = annotated

    # Avoid empty or extremely short result lists; fill with best remaining candidates.
    if len(selected) < config.min_keep_per_query:
        selected_keys = {id(item) for item in selected}
        for item in annotated:
            if id(item) not in selected_keys:
                selected.append(item)
                selected_keys.add(id(item))
            if len(selected) >= config.min_keep_per_query:
                break

    return selected[: config.max_output_per_query]


def annotate_and_select_record(
    prediction_record: dict[str, Any],
    config: SelectorConfig,
) -> dict[str, Any]:
    papers = prediction_record.get("papers") or []
    if not isinstance(papers, list):
        papers = []

    selected = select_papers(papers, config)
    return {
        "qid": prediction_record.get("qid"),
        "question": prediction_record.get("question"),
        "papers": selected,
        "selector": {
            "strategy": "b5_selector_with_relevance_reason_tags",
            "config": asdict(config),
        },
    }
