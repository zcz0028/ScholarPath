from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

from scholarpath.query.constraints import ConstraintDecomposition, QueryConstraint
from scholarpath.rerank.constraint_rerank import (
    paper_key,
    raw_meta,
)


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can",
    "could", "for", "from", "give", "has", "have", "having", "in", "into",
    "is", "it", "its", "me", "of", "on", "or", "paper", "papers", "result",
    "results", "show", "shows", "shown", "that", "the", "their", "there",
    "these", "this", "to", "using", "was", "were", "what", "which", "with",
    "work", "works",
}

SYNONYM_MAP = {
    "large language model": ["llm", "language model", "transformer"],
    "large language models": ["llm", "language models", "transformer"],
    "pre training": ["pretraining", "pre train", "pretrained"],
    "pretraining": ["pre training", "pre train", "pretrained"],
    "smaller dataset": ["small dataset", "less data", "fewer data", "data efficient"],
    "bigger dataset": ["larger dataset", "more data", "large dataset"],
    "better models": ["better performance", "outperform", "improved performance"],
    "scaling law": ["scaling laws", "scaling behavior"],
    "scaling laws": ["scaling law", "scaling behavior"],
    "video text": ["video language", "video-text", "video and text"],
    "image text": ["vision language", "image-text", "image and text"],
    "retrieval augmented generation": ["rag", "retrieval generation"],
    "text to sql": ["text-to-sql", "nl2sql", "natural language to sql"],
}


@dataclass(slots=True)
class SemanticRerankConfig:
    mode: str = "hybrid"
    rank_weight: float = 0.55
    b2_weight: float = 0.15
    semantic_weight: float = 0.30
    title_boost: float = 0.08
    core_constraint_bonus: float = 0.05
    low_evidence_penalty: float = 0.04
    strong_id_bonus: float = 0.01


@dataclass(slots=True)
class SemanticRerankFeatures:
    original_rank_score: float
    b2_constraint_score: float
    query_overlap: float
    expanded_query_overlap: float
    title_overlap: float
    constraint_phrase_coverage: float
    core_constraint_coverage: float
    abstract_signal: float
    semantic_score: float
    final_score: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


MODE_PRESETS = {
    "conservative": {
        "rank_weight": 0.70,
        "b2_weight": 0.15,
        "semantic_weight": 0.15,
        "title_boost": 0.05,
        "core_constraint_bonus": 0.03,
        "low_evidence_penalty": 0.02,
    },
    "hybrid": {
        "rank_weight": 0.55,
        "b2_weight": 0.15,
        "semantic_weight": 0.30,
        "title_boost": 0.08,
        "core_constraint_bonus": 0.05,
        "low_evidence_penalty": 0.04,
    },
    "semantic_first": {
        "rank_weight": 0.35,
        "b2_weight": 0.20,
        "semantic_weight": 0.45,
        "title_boost": 0.10,
        "core_constraint_bonus": 0.07,
        "low_evidence_penalty": 0.06,
    },
}


def make_semantic_config(mode: str) -> SemanticRerankConfig:
    if mode not in MODE_PRESETS:
        raise ValueError(f"Unknown semantic rerank mode: {mode}")
    return SemanticRerankConfig(mode=mode, **MODE_PRESETS[mode])


def normalize_text(value: object | None) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.casefold()
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212-]", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: object | None) -> list[str]:
    normalized = normalize_text(text)
    tokens = re.findall(r"[a-z][a-z0-9]*|\d+[a-z]*", normalized)
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def token_set(text: object | None) -> set[str]:
    return set(tokenize(text))


def reconstruct_openalex_abstract(raw: dict[str, Any]) -> str:
    inverted = raw.get("abstract_inverted_index")
    if not isinstance(inverted, dict):
        return ""

    positions: list[tuple[int, str]] = []
    for word, indices in inverted.items():
        if not isinstance(indices, list):
            continue
        for index in indices:
            try:
                positions.append((int(index), str(word)))
            except (TypeError, ValueError):
                continue
    if not positions:
        return ""
    return " ".join(word for _, word in sorted(positions))


def paper_text_parts(paper: dict[str, Any]) -> dict[str, str]:
    raw = raw_meta(paper)
    title = str(paper.get("title") or "")

    abstract = str(paper.get("abstract") or "")
    if not abstract:
        abstract = reconstruct_openalex_abstract(raw)

    venue = str(paper.get("venue") or "")
    concepts = raw.get("concepts")
    concept_text = ""
    if isinstance(concepts, list):
        concept_text = " ".join(
            str(item.get("display_name", ""))
            for item in concepts
            if isinstance(item, dict)
        )

    return {
        "title": title,
        "abstract": abstract,
        "venue": venue,
        "concepts": concept_text,
        "all": " ".join([title, abstract, venue, concept_text]),
    }


def expand_query_phrases(question: str, constraints: list[QueryConstraint]) -> set[str]:
    phrases: set[str] = {normalize_text(question)}
    for constraint in constraints:
        phrase = normalize_text(constraint.text)
        if phrase:
            phrases.add(phrase)
        for synonym in SYNONYM_MAP.get(phrase, []):
            phrases.add(normalize_text(synonym))

    normalized_question = normalize_text(question)
    for key, values in SYNONYM_MAP.items():
        if key in normalized_question:
            phrases.add(key)
            for value in values:
                phrases.add(normalize_text(value))

    return {item for item in phrases if item}


def expanded_query_tokens(question: str, constraints: list[QueryConstraint]) -> set[str]:
    phrases = expand_query_phrases(question, constraints)
    tokens: set[str] = set()
    for phrase in phrases:
        tokens.update(tokenize(phrase))
    return tokens


def overlap_score(query_tokens: set[str], doc_tokens: set[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    matched = query_tokens & doc_tokens
    return len(matched) / max(1, len(query_tokens))


def soft_overlap_score(query_tokens: set[str], doc_tokens: set[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    exact = query_tokens & doc_tokens
    soft = 0
    for query_token in query_tokens - exact:
        if any(
            doc_token.startswith(query_token) or query_token.startswith(doc_token)
            for doc_token in doc_tokens
            if len(query_token) >= 4 and len(doc_token) >= 4
        ):
            soft += 1
    return (len(exact) + 0.5 * soft) / max(1, len(query_tokens))


def phrase_in_text(phrase: str, text: str) -> bool:
    normalized_phrase = normalize_text(phrase)
    normalized_text = normalize_text(text)
    if not normalized_phrase or not normalized_text:
        return False
    escaped = re.escape(normalized_phrase)
    if normalized_phrase.endswith("s"):
        pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    else:
        pattern = rf"(?<![a-z0-9]){escaped}s?(?![a-z0-9])"
    return re.search(pattern, normalized_text) is not None


def constraint_phrase_score(
    text: str,
    constraints: list[QueryConstraint],
) -> tuple[float, float]:
    if not constraints:
        return 0.0, 0.0

    covered = 0
    core_covered = 0
    core_total = 0
    for constraint in constraints:
        if constraint.constraint_type in {
            "performance_relation",
            "data_condition",
            "method_or_property",
        }:
            core_total += 1

        if phrase_in_text(constraint.text, text):
            covered += 1
            if constraint.constraint_type in {
                "performance_relation",
                "data_condition",
                "method_or_property",
            }:
                core_covered += 1
            continue

        terms = [term for term in constraint.terms if normalize_text(term)]
        if not terms:
            continue
        doc_tokens = token_set(text)
        matched = sum(1 for term in terms if soft_overlap_score({normalize_text(term)}, doc_tokens) > 0)
        required = len(terms) if len(terms) <= 2 else max(2, len(terms) - 1)
        if matched >= required:
            covered += 1
            if constraint.constraint_type in {
                "performance_relation",
                "data_condition",
                "method_or_property",
            }:
                core_covered += 1

    return covered / max(1, len(constraints)), core_covered / max(1, core_total)


def original_rank_score(rank: int) -> float:
    return 1.0 / math.log2(max(2, int(rank) + 1))


def existing_b2_score(paper: dict[str, Any]) -> float:
    raw = raw_meta(paper)
    value = raw.get("b2_1_constraint_score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def has_strong_identifier(paper: dict[str, Any]) -> bool:
    return any(
        str(paper.get(field) or "").strip()
        for field in ("doi", "arxiv_id", "openalex_id", "semantic_scholar_id")
    )


def compute_semantic_features(
    *,
    paper: dict[str, Any],
    question: str,
    decomposition: ConstraintDecomposition,
    original_rank: int,
    config: SemanticRerankConfig,
) -> SemanticRerankFeatures:
    parts = paper_text_parts(paper)
    constraints = decomposition.constraints

    query_tokens = token_set(question)
    expanded_tokens = expanded_query_tokens(question, constraints)
    doc_tokens = token_set(parts["all"])
    title_tokens = token_set(parts["title"])
    abstract_tokens = token_set(parts["abstract"])

    query_overlap = soft_overlap_score(query_tokens, doc_tokens)
    expanded_overlap = soft_overlap_score(expanded_tokens, doc_tokens)
    title_overlap = soft_overlap_score(query_tokens | expanded_tokens, title_tokens)
    abstract_signal = soft_overlap_score(query_tokens | expanded_tokens, abstract_tokens)

    phrase_coverage, core_coverage = constraint_phrase_score(parts["all"], constraints)

    semantic_score = (
        0.28 * query_overlap
        + 0.25 * expanded_overlap
        + 0.22 * title_overlap
        + 0.15 * phrase_coverage
        + 0.07 * core_coverage
        + 0.03 * abstract_signal
    )

    rank_score = original_rank_score(original_rank)
    b2_score = existing_b2_score(paper)

    final_score = (
        config.rank_weight * rank_score
        + config.b2_weight * b2_score
        + config.semantic_weight * semantic_score
    )

    if title_overlap > 0:
        final_score += config.title_boost * title_overlap

    if core_coverage > 0:
        final_score += config.core_constraint_bonus * core_coverage

    if semantic_score < 0.08 and core_coverage == 0:
        final_score -= config.low_evidence_penalty

    if has_strong_identifier(paper):
        final_score += config.strong_id_bonus

    return SemanticRerankFeatures(
        original_rank_score=rank_score,
        b2_constraint_score=b2_score,
        query_overlap=query_overlap,
        expanded_query_overlap=expanded_overlap,
        title_overlap=title_overlap,
        constraint_phrase_coverage=phrase_coverage,
        core_constraint_coverage=core_coverage,
        abstract_signal=abstract_signal,
        semantic_score=semantic_score,
        final_score=final_score,
    )


def semantic_rerank_score(features: SemanticRerankFeatures) -> float:
    return features.final_score


def annotate_paper(
    paper: dict[str, Any],
    *,
    question: str,
    decomposition: ConstraintDecomposition,
    original_rank: int,
    config: SemanticRerankConfig,
) -> dict[str, Any]:
    features = compute_semantic_features(
        paper=paper,
        question=question,
        decomposition=decomposition,
        original_rank=original_rank,
        config=config,
    )
    annotated = dict(paper)
    raw = dict(raw_meta(annotated))
    raw["b3_original_rank"] = original_rank
    raw["b3_rerank_mode"] = config.mode
    raw["b3_features"] = features.to_dict()
    raw["b3_semantic_score"] = features.semantic_score
    raw["b3_final_score"] = features.final_score
    annotated["raw"] = raw
    return annotated


def annotate_and_semantic_rerank_record(
    prediction_record: dict[str, Any],
    decomposition: ConstraintDecomposition,
    config: SemanticRerankConfig,
) -> dict[str, Any]:
    question = str(prediction_record.get("question") or decomposition.question or "")
    papers = prediction_record.get("papers") or []
    if not isinstance(papers, list):
        papers = []

    annotated: list[dict[str, Any]] = []
    seen: set[str] = set()

    for rank, paper in enumerate(papers, start=1):
        if not isinstance(paper, dict):
            continue
        key = paper_key(paper)
        if key in seen:
            continue
        seen.add(key)
        annotated.append(
            annotate_paper(
                paper,
                question=question,
                decomposition=decomposition,
                original_rank=rank,
                config=config,
            )
        )

    annotated.sort(
        key=lambda item: (
            -float(raw_meta(item).get("b3_final_score", 0.0)),
            -float(raw_meta(item).get("b3_semantic_score", 0.0)),
            int(raw_meta(item).get("b3_original_rank", 999999)),
            normalize_text(item.get("title")),
        )
    )

    return {
        "qid": prediction_record.get("qid"),
        "question": question,
        "papers": annotated,
        "rerank": {
            "strategy": "b3_lightweight_semantic_rerank",
            "config": asdict(config),
            "constraint_count": len(decomposition.constraints),
        },
    }
