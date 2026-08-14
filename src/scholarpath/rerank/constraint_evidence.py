from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from scholarpath.query.constraints import QueryConstraint


# Day8-1A V2 design rules:
# 1. Never compare raw strings directly.
# 2. Normalize Unicode/case/hyphens/punctuation/whitespace before comparison.
# 3. Use only conservative deterministic morphology and auditable aliases.
# 4. Clean connector noise only at phrase boundaries; never strip internal words.
# 5. Merge established aliases/equivalents before token-similarity deduplication.
# 6. Token-similarity merging requires the same constraint type.
# 7. Merged weights use max(weight), never sum(weight).
# 8. Recover meaningful uncovered query spans as low-risk topic constraints.
# 9. No LLM, embedding API or network call is used in this module.


_LIGHT_SINGULAR_MAP: dict[str, str] = {
    "papers": "paper",
    "models": "model",
    "networks": "network",
    "datasets": "dataset",
    "methods": "method",
    "systems": "system",
    "agents": "agent",
    "factors": "factor",
    "signals": "signal",
    "documents": "document",
    "properties": "property",
    "images": "image",
    "videos": "video",
    "tasks": "task",
    "benchmarks": "benchmark",
    "representations": "representation",
    "embeddings": "embedding",
}


# Leading cleanup is intentionally narrower than trailing cleanup. Phrases such
# as "in context learning" must remain intact.
_LEADING_BOUNDARY_CONNECTORS = {
    "for",
    "on",
    "about",
    "using",
    "via",
}
_TRAILING_BOUNDARY_CONNECTORS = {
    "for",
    "on",
    "about",
    "with",
    "using",
    "via",
    "of",
    "in",
}


# Tokens that delimit or do not constitute a residual research constraint.
# These are applied only during residual recovery, not normal evidence matching.
_RESIDUAL_BOUNDARY_TOKENS = {
    "a",
    "an",
    "and",
    "any",
    "about",
    "for",
    "from",
    "give",
    "in",
    "me",
    "of",
    "on",
    "or",
    "paper",
    "recent",
    "related",
    "relevant",
    "research",
    "show",
    "some",
    "study",
    "that",
    "the",
    "to",
    "use",
    "using",
    "via",
    "which",
    "with",
    "work",
    # Day14-1C prompt/discourse residue. These tokens are boundaries only during
    # residual recovery; normal evidence matching is unaffected.
    "can",
    "claim",
    "could",
    "explaining",
    "help",
    "know",
    "list",
    "please",
    "provide",
    "supporting",
    "want",
    "why",
}


_ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
    "large language model": (
        "large language model",
        "large language models",
        "llm",
        "llms",
    ),
    "graph neural network": (
        "graph neural network",
        "graph neural networks",
        "gnn",
        "gnns",
    ),
    "retrieval augmented generation": (
        "retrieval augmented generation",
        "retrieval-augmented generation",
        "rag",
    ),
    "natural language to sql": (
        "natural language to sql",
        "natural-language-to-sql",
        "text to sql",
        "text-to-sql",
        "nl2sql",
    ),
    "pre training": (
        "pre training",
        "pre-training",
        "pretraining",
        "pre train",
    ),
    "vision language": (
        "vision language",
        "vision-language",
    ),
    "video text": (
        "video text",
        "video-text",
    ),
    "image text": (
        "image text",
        "image-text",
    ),
}


@dataclass(slots=True, frozen=True)
class CanonicalConstraint:
    id: str
    canonical_text: str
    constraint_type: str
    terms: tuple[str, ...]
    aliases: tuple[str, ...]
    weight: float
    source_constraint_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["terms"] = list(self.terms)
        data["aliases"] = list(self.aliases)
        data["source_constraint_ids"] = list(self.source_constraint_ids)
        return data


@dataclass(slots=True, frozen=True)
class ConstraintDedupConfig:
    same_type_jaccard_threshold: float = 0.80
    recover_residual_constraints: bool = True
    residual_min_tokens: int = 2
    residual_max_tokens: int = 8
    max_residual_constraints: int = 3
    residual_weight: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.same_type_jaccard_threshold <= 1.0:
            raise ValueError(
                "same_type_jaccard_threshold must be within [0, 1]"
            )
        if self.residual_min_tokens < 1:
            raise ValueError("residual_min_tokens must be positive")
        if self.residual_max_tokens < self.residual_min_tokens:
            raise ValueError(
                "residual_max_tokens must be >= residual_min_tokens"
            )
        if self.max_residual_constraints < 0:
            raise ValueError("max_residual_constraints cannot be negative")
        if self.residual_weight <= 0:
            raise ValueError("residual_weight must be positive")


def normalize_evidence_text(value: object | None) -> str:
    """Normalize arbitrary English-oriented evidence text.

    Boundary connectors are *not* removed here because this function is also
    suitable for titles/abstracts. Constraint-only cleanup is handled by
    normalize_constraint_text().
    """
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.casefold()
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212-]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    tokens = [_normalize_token(token) for token in text.split()]
    return " ".join(token for token in tokens if token)


def normalize_constraint_text(value: object | None) -> str:
    """Normalize a constraint phrase and remove boundary connector noise only."""
    normalized = normalize_evidence_text(value)
    if not normalized:
        return ""
    tokens = normalized.split()

    # Remove only when at least two meaningful tokens remain. This prevents
    # collapsing short meaningful phrases to a single generic token.
    while len(tokens) >= 3 and tokens[0] in _LEADING_BOUNDARY_CONNECTORS:
        tokens.pop(0)
    while len(tokens) >= 3 and tokens[-1] in _TRAILING_BOUNDARY_CONNECTORS:
        tokens.pop()
    return " ".join(tokens)


def normalized_tokens(value: object | None) -> tuple[str, ...]:
    normalized = normalize_evidence_text(value)
    if not normalized:
        return ()
    return tuple(normalized.split())


def token_jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = {item for item in left if item}
    right_set = {item for item in right if item}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def canonical_alias_text(value: object | None) -> str:
    normalized = normalize_constraint_text(value)
    if not normalized:
        return ""
    for canonical, aliases in _NORMALIZED_ALIAS_GROUPS.items():
        if normalized == canonical or normalized in aliases:
            return canonical
    return normalized


def deduplicate_constraints(
    constraints: Sequence[QueryConstraint],
    *,
    config: ConstraintDedupConfig | None = None,
) -> list[CanonicalConstraint]:
    """Deduplicate planner constraints without recovering query residuals.

    Merge order:
      1. normalized/boundary-cleaned exact equivalence;
      2. established alias equivalence;
      3. high token Jaccard similarity within the same constraint type.

    Exact/declared alias equivalence may merge across planner type drift; raw
    token similarity may not. Weights always use max(), never sum().
    """
    cfg = config or ConstraintDedupConfig()
    canonical: list[_MutableCanonicalConstraint] = []

    for constraint in constraints:
        raw_normalized = normalize_evidence_text(constraint.text)
        normalized = normalize_constraint_text(constraint.text)
        if not normalized:
            continue

        alias_canonical = canonical_alias_text(normalized)
        tokens = normalized_tokens(alias_canonical)
        if not tokens:
            continue

        target = _find_merge_target(
            canonical,
            constraint=constraint,
            alias_canonical=alias_canonical,
            tokens=tokens,
            config=cfg,
        )

        if target is None:
            aliases = {normalized, alias_canonical}
            if raw_normalized:
                aliases.add(raw_normalized)
            canonical.append(
                _MutableCanonicalConstraint(
                    canonical_text=alias_canonical,
                    constraint_type=constraint.constraint_type,
                    terms=set(tokens),
                    aliases=aliases,
                    weight=float(constraint.weight),
                    source_constraint_ids=[constraint.id],
                )
            )
            continue

        if raw_normalized:
            target.aliases.add(raw_normalized)
        target.aliases.add(normalized)
        target.aliases.add(alias_canonical)
        target.terms.update(tokens)
        target.weight = max(target.weight, float(constraint.weight))
        if constraint.id not in target.source_constraint_ids:
            target.source_constraint_ids.append(constraint.id)
        target.canonical_text = _preferred_canonical_text(
            target.canonical_text,
            alias_canonical,
        )

    return _freeze_constraints(canonical)


def build_canonical_constraints(
    *,
    question: str,
    constraints: Sequence[QueryConstraint],
    config: ConstraintDedupConfig | None = None,
) -> list[CanonicalConstraint]:
    """Build canonical constraints and recover uncovered query spans.

    Residual recovery is deliberately deterministic: it marks query token spans
    already explained by canonical constraints, then considers only remaining
    contiguous non-noise spans. Recovered spans are re-run through the same
    deduplicator, so they cannot silently bypass normal merge rules.
    """
    cfg = config or ConstraintDedupConfig()
    base = deduplicate_constraints(constraints, config=cfg)
    if (
        not cfg.recover_residual_constraints
        or cfg.max_residual_constraints == 0
        or not question.strip()
    ):
        return base

    spans = recover_residual_constraint_spans(
        question=question,
        canonical_constraints=base,
        config=cfg,
    )
    if not spans:
        return base

    residuals = [
        QueryConstraint(
            id=f"residual_query_span_{index}",
            text=span,
            constraint_type="topic",
            terms=list(normalized_tokens(span)),
            weight=cfg.residual_weight,
        )
        for index, span in enumerate(spans, start=1)
    ]
    return deduplicate_constraints([*constraints, *residuals], config=cfg)



_RESIDUAL_DANGLING_ENDINGS = {
    "a", "an", "and", "as", "between", "by", "for", "from", "have", "has",
    "in", "inductive", "into", "knowledgeable", "multiple", "of", "on", "or",
    "such", "systematically", "the", "to", "using", "via", "with",
}

_RESIDUAL_GENERIC_TOKENS = {
    "analysis", "approach", "data", "document", "learning", "method", "model",
    "network", "paper", "prediction", "research", "result", "study", "system",
    "task", "relationship", "relationships",
}

_RESIDUAL_PROTECTED_SHORT_TERMS = {
    "ai", "ee", "gnn", "gnns", "hotpotqa", "imo", "llm", "llms", "ner",
    "qat", "rag", "re", "rlhf", "sft",
}


def _residual_span_is_meaningful(value: object | None) -> bool:
    """Quality guard for recovered query spans only.

    Residual recovery is useful, but without a guard it can re-introduce the
    same discourse/dangling fragments filtered by the planner. Short academic
    entities remain explicitly protected.
    """
    normalized = normalize_constraint_text(value)
    tokens = list(normalized_tokens(normalized))
    if not tokens:
        return False

    if normalized in _RESIDUAL_PROTECTED_SHORT_TERMS:
        return True

    if len(tokens) == 1:
        return False

    if tokens[-1] in _RESIDUAL_DANGLING_ENDINGS:
        return False

    if "between" in tokens and tokens[-1] in {"between", "multiple"}:
        return False

    if len(tokens) <= 3 and all(token in _RESIDUAL_GENERIC_TOKENS for token in tokens):
        return False

    return True

def recover_residual_constraint_spans(
    *,
    question: str,
    canonical_constraints: Sequence[CanonicalConstraint],
    config: ConstraintDedupConfig | None = None,
) -> list[str]:
    """Recover meaningful query spans not covered by canonical constraints."""
    cfg = config or ConstraintDedupConfig()
    query_tokens = list(normalized_tokens(question))
    if not query_tokens:
        return []

    covered = [False] * len(query_tokens)
    for constraint in canonical_constraints:
        phrases = {
            constraint.canonical_text,
            *constraint.aliases,
        }
        for phrase in phrases:
            phrase_tokens = list(normalized_tokens(normalize_constraint_text(phrase)))
            if not phrase_tokens:
                continue
            _mark_sequence_occurrences(
                query_tokens=query_tokens,
                phrase_tokens=phrase_tokens,
                covered=covered,
            )

    candidate_spans: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        if cfg.residual_min_tokens <= len(current) <= cfg.residual_max_tokens:
            text = normalize_constraint_text(" ".join(current))
            if (
                text
                and len(normalized_tokens(text)) >= cfg.residual_min_tokens
                and _residual_span_is_meaningful(text)
            ):
                candidate_spans.append(text)
        current = []

    for index, token in enumerate(query_tokens):
        if covered[index] or token in _RESIDUAL_BOUNDARY_TOKENS:
            flush()
            continue
        current.append(token)
    flush()

    output: list[str] = []
    seen: set[str] = set()
    for span in candidate_spans:
        canonical = canonical_alias_text(span)
        if (
            not canonical
            or canonical in seen
            or not _residual_span_is_meaningful(canonical)
        ):
            continue
        # Residuals must add genuinely new content, not a differently chunked
        # restatement of an existing canonical concept.
        residual_tokens = set(normalized_tokens(canonical))
        if any(
            token_jaccard(residual_tokens, set(item.terms))
            >= cfg.same_type_jaccard_threshold
            for item in canonical_constraints
            if item.terms
        ):
            continue
        output.append(canonical)
        seen.add(canonical)
        if len(output) >= cfg.max_residual_constraints:
            break
    return output


def canonical_constraints_to_dicts(
    constraints: Sequence[QueryConstraint],
    *,
    question: str | None = None,
    config: ConstraintDedupConfig | None = None,
) -> list[dict]:
    if question is None:
        items = deduplicate_constraints(constraints, config=config)
    else:
        items = build_canonical_constraints(
            question=question,
            constraints=constraints,
            config=config,
        )
    return [item.to_dict() for item in items]


@dataclass(slots=True)
class _MutableCanonicalConstraint:
    canonical_text: str
    constraint_type: str
    terms: set[str]
    aliases: set[str]
    weight: float
    source_constraint_ids: list[str]


def _normalize_token(token: str) -> str:
    return _LIGHT_SINGULAR_MAP.get(token, token)


def _normalize_alias_group(values: Iterable[str]) -> set[str]:
    return {
        normalized
        for value in values
        if (normalized := normalize_constraint_text(value))
    }


_NORMALIZED_ALIAS_GROUPS: dict[str, set[str]] = {
    normalize_constraint_text(canonical): _normalize_alias_group(aliases)
    for canonical, aliases in _ALIAS_GROUPS.items()
}


def _find_merge_target(
    canonical: Sequence[_MutableCanonicalConstraint],
    *,
    constraint: QueryConstraint,
    alias_canonical: str,
    tokens: tuple[str, ...],
    config: ConstraintDedupConfig,
) -> _MutableCanonicalConstraint | None:
    for item in canonical:
        if item.canonical_text == alias_canonical:
            return item
        if alias_canonical in item.aliases:
            return item

    for item in canonical:
        if item.constraint_type != constraint.constraint_type:
            continue
        similarity = token_jaccard(item.terms, tokens)
        if similarity >= config.same_type_jaccard_threshold:
            return item
    return None


def _preferred_canonical_text(left: str, right: str) -> str:
    left_alias = canonical_alias_text(left)
    right_alias = canonical_alias_text(right)
    if left_alias != left:
        return left_alias
    if right_alias != right:
        return right_alias
    left_tokens = normalized_tokens(left)
    right_tokens = normalized_tokens(right)
    if len(right_tokens) < len(left_tokens):
        return right
    if len(left_tokens) < len(right_tokens):
        return left
    return min(left, right)


def _freeze_constraints(
    canonical: Sequence[_MutableCanonicalConstraint],
) -> list[CanonicalConstraint]:
    return [
        CanonicalConstraint(
            id=f"cc{index}_{_slugify(item.canonical_text)}",
            canonical_text=item.canonical_text,
            constraint_type=item.constraint_type,
            terms=tuple(sorted(item.terms)),
            aliases=tuple(sorted(item.aliases)),
            weight=item.weight,
            source_constraint_ids=tuple(item.source_constraint_ids),
        )
        for index, item in enumerate(canonical, start=1)
    ]


def _mark_sequence_occurrences(
    *,
    query_tokens: Sequence[str],
    phrase_tokens: Sequence[str],
    covered: list[bool],
) -> None:
    width = len(phrase_tokens)
    if width == 0 or width > len(query_tokens):
        return
    for start in range(0, len(query_tokens) - width + 1):
        if list(query_tokens[start : start + width]) == list(phrase_tokens):
            for index in range(start, start + width):
                covered[index] = True


def _slugify(text: str) -> str:
    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        normalize_evidence_text(text),
    ).strip("_")
    return value[:48] or "constraint"

# ---------------------------------------------------------------------------
# Day8-1B: deterministic Constraint Evidence Matcher
# ---------------------------------------------------------------------------

from math import ceil
from typing import Any, Mapping

from scholarpath.rerank.constraint_rerank import raw_meta
from scholarpath.rerank.semantic_rerank import reconstruct_openalex_abstract


# A single generic word is too weak to constitute evidence on its own.
_GENERIC_SINGLE_TOKENS = {
    "analysis",
    "approach",
    "data",
    "document",
    "learning",
    "method",
    "model",
    "network",
    "paper",
    "prediction",
    "result",
    "system",
    "task",
}


@dataclass(slots=True, frozen=True)
class ConstraintEvidence:
    constraint_id: str
    constraint_text: str
    constraint_type: str
    matched: bool
    match_type: str
    evidence_field: str | None
    evidence_text: str | None
    confidence: float
    token_coverage: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class EvidenceMatcherConfig:
    title_phrase_confidence: float = 1.00
    title_alias_confidence: float = 0.95
    title_token_confidence: float = 0.90
    abstract_phrase_confidence: float = 0.90
    abstract_alias_confidence: float = 0.86
    abstract_token_confidence: float = 0.82
    concept_phrase_confidence: float = 0.75
    concept_alias_confidence: float = 0.72
    concept_token_confidence: float = 0.68
    long_constraint_token_coverage: float = 0.80

    def __post_init__(self) -> None:
        for field_name in (
            "title_phrase_confidence",
            "title_alias_confidence",
            "title_token_confidence",
            "abstract_phrase_confidence",
            "abstract_alias_confidence",
            "abstract_token_confidence",
            "concept_phrase_confidence",
            "concept_alias_confidence",
            "concept_token_confidence",
            "long_constraint_token_coverage",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be within [0, 1]")


def build_constraint_evidence(
    constraints: Sequence[CanonicalConstraint],
    paper: Mapping[str, Any],
    *,
    config: EvidenceMatcherConfig | None = None,
) -> list[ConstraintEvidence]:
    """Build one best evidence record for every canonical constraint.

    Field priority is deterministic: title > abstract > individual OpenAlex
    concept display_name. Within one field the order is canonical phrase >
    alias phrase > token coverage. Unmatched constraints are retained.

    Empty constraints return [] so upper layers can safely define coverage=0.
    """
    if not constraints:
        return []

    cfg = config or EvidenceMatcherConfig()
    paper_dict = dict(paper)
    title = str(paper_dict.get("title") or "")
    abstract = _paper_abstract(paper_dict)
    concepts = _concept_display_names(paper_dict)

    return [
        match_constraint_evidence(
            constraint,
            title=title,
            abstract=abstract,
            concepts=concepts,
            config=cfg,
        )
        for constraint in constraints
    ]


def match_constraint_evidence(
    constraint: CanonicalConstraint,
    *,
    title: object | None = None,
    abstract: object | None = None,
    concepts: Sequence[object] | None = None,
    config: EvidenceMatcherConfig | None = None,
) -> ConstraintEvidence:
    """Return the highest-priority deterministic evidence for one constraint."""
    cfg = config or EvidenceMatcherConfig()
    canonical = normalize_constraint_text(constraint.canonical_text)
    phrases = _ordered_constraint_phrases(constraint)

    title_result = _match_one_field(
        constraint=constraint,
        canonical=canonical,
        phrases=phrases,
        field_name="title",
        value=title,
        phrase_confidence=cfg.title_phrase_confidence,
        alias_confidence=cfg.title_alias_confidence,
        token_confidence=cfg.title_token_confidence,
        config=cfg,
    )
    if title_result is not None:
        return title_result

    abstract_result = _match_one_field(
        constraint=constraint,
        canonical=canonical,
        phrases=phrases,
        field_name="abstract",
        value=abstract,
        phrase_confidence=cfg.abstract_phrase_confidence,
        alias_confidence=cfg.abstract_alias_confidence,
        token_confidence=cfg.abstract_token_confidence,
        config=cfg,
    )
    if abstract_result is not None:
        return abstract_result

    # Concepts are intentionally matched one display_name at a time. Never join
    # separate OpenAlex concepts, otherwise unrelated concept tokens could be
    # combined into false evidence.
    for concept in concepts or ():
        concept_result = _match_one_field(
            constraint=constraint,
            canonical=canonical,
            phrases=phrases,
            field_name="concept",
            value=concept,
            phrase_confidence=cfg.concept_phrase_confidence,
            alias_confidence=cfg.concept_alias_confidence,
            token_confidence=cfg.concept_token_confidence,
            config=cfg,
        )
        if concept_result is not None:
            return concept_result

    return _unmatched_evidence(constraint)


def _ordered_constraint_phrases(
    constraint: CanonicalConstraint,
) -> tuple[tuple[str, bool], ...]:
    """Return canonical first, then stable unique aliases.

    The boolean marks whether the phrase is an alias. We never rely on the
    stored aliases tuple order for canonical priority.
    """
    canonical = normalize_constraint_text(constraint.canonical_text)
    output: list[tuple[str, bool]] = []
    seen: set[str] = set()
    if canonical:
        output.append((canonical, False))
        seen.add(canonical)

    for alias in constraint.aliases:
        normalized = normalize_constraint_text(alias)
        if not normalized or normalized in seen:
            continue
        output.append((normalized, True))
        seen.add(normalized)
    return tuple(output)


def _match_one_field(
    *,
    constraint: CanonicalConstraint,
    canonical: str,
    phrases: Sequence[tuple[str, bool]],
    field_name: str,
    value: object | None,
    phrase_confidence: float,
    alias_confidence: float,
    token_confidence: float,
    config: EvidenceMatcherConfig,
) -> ConstraintEvidence | None:
    normalized_field = normalize_evidence_text(value)
    if not normalized_field or not canonical:
        return None

    # Canonical phrase is always tested before aliases.
    for phrase, is_alias in phrases:
        if (
            _phrase_is_eligible_as_evidence(phrase)
            and _normalized_phrase_in_text(phrase, normalized_field)
        ):
            match_kind = "alias" if is_alias else "phrase"
            confidence = alias_confidence if is_alias else phrase_confidence
            return _matched_evidence(
                constraint=constraint,
                match_type=f"{field_name}_{match_kind}_match",
                evidence_field=field_name,
                evidence_text=phrase,
                confidence=confidence,
                token_coverage=1.0,
            )

    matched, coverage, matched_tokens = _token_coverage_match(
        constraint_text=canonical,
        field_text=normalized_field,
        config=config,
    )
    if matched:
        return _matched_evidence(
            constraint=constraint,
            match_type=f"{field_name}_token_match",
            evidence_field=field_name,
            evidence_text=" ".join(matched_tokens),
            confidence=token_confidence,
            token_coverage=coverage,
        )
    return None



def _phrase_is_eligible_as_evidence(phrase: object | None) -> bool:
    tokens = tuple(dict.fromkeys(normalized_tokens(phrase)))
    if not tokens:
        return False
    if len(tokens) == 1 and tokens[0] in _GENERIC_SINGLE_TOKENS:
        return False
    return True


def _normalized_phrase_in_text(phrase: object | None, text: object | None) -> bool:
    """Match a normalized phrase on token boundaries without regex.

    This is intentionally *not* raw-string contains matching:
    both operands are normalized first, punctuation/hyphens have already
    become token separators, and explicit surrounding spaces enforce whole
    token-sequence boundaries.

    The padded-string check is implemented in CPython's optimized substring
    search and is substantially faster than compiling/executing a regex for
    every constraint-paper-field pair during full Top100 evaluation.
    """
    normalized_phrase = normalize_constraint_text(phrase)
    normalized_text = normalize_evidence_text(text)
    if not normalized_phrase or not normalized_text:
        return False

    needle = f" {normalized_phrase} "
    haystack = f" {normalized_text} "
    return needle in haystack


def _token_coverage_match(
    *,
    constraint_text: object | None,
    field_text: object | None,
    config: EvidenceMatcherConfig,
) -> tuple[bool, float, tuple[str, ...]]:
    constraint_tokens = tuple(dict.fromkeys(normalized_tokens(constraint_text)))
    field_tokens = set(normalized_tokens(field_text))
    if not constraint_tokens or not field_tokens:
        return False, 0.0, ()

    matched_tokens = tuple(
        token for token in constraint_tokens if token in field_tokens
    )
    matched_count = len(matched_tokens)
    total = len(constraint_tokens)
    coverage = matched_count / total

    if total == 1:
        token = constraint_tokens[0]
        accepted = matched_count == 1 and token not in _GENERIC_SINGLE_TOKENS
        return accepted, coverage, matched_tokens

    if total == 2:
        return matched_count == 2, coverage, matched_tokens

    # Convert the threshold to an integer requirement to avoid floating-point
    # boundary ambiguity. Examples at 0.8: 3->3, 4->4, 5->4.
    required = max(2, ceil(config.long_constraint_token_coverage * total))
    return matched_count >= required, coverage, matched_tokens


def _paper_abstract(paper: dict[str, Any]) -> str:
    abstract = str(paper.get("abstract") or "").strip()
    if abstract:
        return abstract
    raw = raw_meta(paper)
    return reconstruct_openalex_abstract(raw)


def _concept_display_names(paper: dict[str, Any]) -> tuple[str, ...]:
    raw = raw_meta(paper)
    concepts = raw.get("concepts")
    if not isinstance(concepts, list):
        return ()

    output: list[str] = []
    for item in concepts:
        if not isinstance(item, dict):
            continue
        display_name = str(item.get("display_name") or "").strip()
        if display_name:
            output.append(display_name)
    return tuple(output)


def _matched_evidence(
    *,
    constraint: CanonicalConstraint,
    match_type: str,
    evidence_field: str,
    evidence_text: str,
    confidence: float,
    token_coverage: float,
) -> ConstraintEvidence:
    return ConstraintEvidence(
        constraint_id=constraint.id,
        constraint_text=constraint.canonical_text,
        constraint_type=constraint.constraint_type,
        matched=True,
        match_type=match_type,
        evidence_field=evidence_field,
        evidence_text=evidence_text,
        confidence=max(0.0, min(1.0, float(confidence))),
        token_coverage=max(0.0, min(1.0, float(token_coverage))),
    )


def _unmatched_evidence(
    constraint: CanonicalConstraint,
) -> ConstraintEvidence:
    return ConstraintEvidence(
        constraint_id=constraint.id,
        constraint_text=constraint.canonical_text,
        constraint_type=constraint.constraint_type,
        matched=False,
        match_type="none",
        evidence_field=None,
        evidence_text=None,
        confidence=0.0,
        token_coverage=0.0,
    )
