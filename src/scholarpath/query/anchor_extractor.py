from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass(slots=True, frozen=True)
class AnchorEvidence:
    text: str
    normalized_text: str
    anchor_type: str
    source_span: str
    start: int
    end: int
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnchorExtractionResult:
    question: str
    anchors: list[AnchorEvidence]

    def to_dict(self) -> dict[str, Any]:
        by_type: dict[str, list[str]] = {}
        for anchor in self.anchors:
            by_type.setdefault(anchor.anchor_type, []).append(anchor.text)
        return {
            "question": self.question,
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "anchor_types": sorted(by_type),
            "anchors_by_type": by_type,
        }


KNOWN_ANCHORS: dict[str, tuple[str, ...]] = {
    "benchmark": (
        "HumanEval",
        "MBPP",
        "CodeContests",
        "code_contests",
        "HotpotQA",
        "MMLU",
        "GSM8K",
        "ImageNet",
        "MS COCO",
        "COCO",
        "SQuAD",
        "GLUE",
        "SuperGLUE",
        "TruthfulQA",
        "HellaSwag",
        "ARC-Challenge",
        "BigBench",
        "BIG-Bench",
    ),
    "method": (
        "RLHF",
        "DPO",
        "SFT",
        "RAG",
        "LoRA",
        "QLoRA",
        "CoT",
        "chain-of-thought",
        "chain of thought",
        "in-context learning",
        "instruction tuning",
        "test-time training",
        "retrieval-augmented generation",
        "retrieval augmented generation",
        "knowledge distillation",
        "contrastive learning",
        "self-supervised learning",
        "reinforcement learning",
        "direct preference optimization",
    ),
    "model": (
        "GPT-4",
        "GPT-3",
        "BERT",
        "T5",
        "LLaMA",
        "Llama",
        "LLaVA",
        "CLIP",
        "Stable Diffusion",
        "Transformer",
        "ViT",
        "VLM",
        "LLM",
    ),
    "task": (
        "text-to-SQL",
        "text to SQL",
        "question answering",
        "code generation",
        "event extraction",
        "document-level event extraction",
        "semantic segmentation",
        "object detection",
        "machine translation",
        "information retrieval",
        "paper recommendation",
        "video generation",
        "robot task planning",
        "task planning",
        "robot decision making",
        "game playing",
        "computer control",
    ),
    "modality": (
        "vision-language",
        "vision language",
        "video-language",
        "video language",
        "multimodal",
        "multi-modal",
        "audio-visual",
        "audio visual",
        "3D",
        "video",
        "image",
        "audio",
        "text",
    ),
}

NOISE_PHRASES = (
    "find papers",
    "find me papers",
    "show me",
    "give me papers",
    "help me find",
    "can you help me",
    "share insights",
    "i want to know",
    "recommend papers",
    "please find",
    "papers which",
    "papers that",
)

NEGATIVE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(?:exclude|excluding|without|do not include|don't include|not)\s+(?:any\s+)?(?:survey|surveys|review papers?|review articles?)\b", "exclude_survey"),
    (r"\b(?:non[- ]survey|non[- ]review)\b", "exclude_survey"),
    (r"\b(?:exclude|without|do not include|don't include)\s+(?:any\s+)?(?:preprints?|arxiv papers?)\b", "exclude_preprint"),
)

TEMPORAL_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(?:after|since|from)\s+(19\d{2}|20\d{2})\b", "year_from"),
    (r"\b(?:before|until|up to)\s+(19\d{2}|20\d{2})\b", "year_to"),
    (r"\bbetween\s+(19\d{2}|20\d{2})\s+(?:and|to)\s+(19\d{2}|20\d{2})\b", "year_range"),
    (r"\b(?:recent|latest|newest)\s+(?:papers?|work|research)\b", "recent_only"),
)

COMPARISON_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        r"\b(?:harder|more difficult)\s+than\s+([^,.;]+?)\s+(?:but|and)\s+(?:easier|less difficult)\s+than\s+([^,.;?]+)",
        "difficulty_between",
    ),
    (
        r"\b(?:better|stronger|more accurate)\s+than\s+([^,.;?]+)",
        "outperforms_anchor",
    ),
    (
        r"\b(?:smaller|less|fewer)\s+([^,.;]+?)\s+than\s+([^,.;?]+)",
        "smaller_than",
    ),
)

QUOTED_TEXT_RE = re.compile(r"[\"“”']([^\"“”']{5,180})[\"“”']")
TITLE_TRIGGER_RE = re.compile(
    r"\b(?:paper|work|article)\s+(?:titled|called|named)\s+[\"“”']?([^\"“”',.;?]{5,180})",
    flags=re.IGNORECASE,
)
CAMEL_OR_ACRONYM_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Z]{2,}[A-Za-z0-9_-]*|[A-Z][a-z]+(?:[A-Z][A-Za-z0-9_-]+)+|[A-Za-z]+_[A-Za-z0-9_]+)(?![A-Za-z0-9_])"
)


def _normalize(value: str) -> str:
    value = value.casefold().replace("_", " ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _source_span(question: str, start: int, end: int, context: int = 28) -> str:
    left = max(0, start - context)
    right = min(len(question), end + context)
    return question[left:right].strip()


def _overlaps_noise(question: str, start: int, end: int) -> bool:
    candidate = _normalize(question[start:end])
    if not candidate:
        return True
    return any(candidate == _normalize(noise) for noise in NOISE_PHRASES)


def _add_anchor(
    output: list[AnchorEvidence],
    seen: set[tuple[str, str, int, int]],
    *,
    question: str,
    text: str,
    anchor_type: str,
    start: int,
    end: int,
    confidence: float,
    metadata: dict[str, Any] | None = None,
) -> None:
    cleaned = text.strip(" \t\r\n,.;:?!()[]{}\"'“”")
    normalized = _normalize(cleaned)
    if not cleaned or not normalized:
        return
    if _overlaps_noise(question, start, end):
        return
    key = (anchor_type, normalized, start, end)
    if key in seen:
        return
    seen.add(key)
    output.append(
        AnchorEvidence(
            text=cleaned,
            normalized_text=normalized,
            anchor_type=anchor_type,
            source_span=_source_span(question, start, end),
            start=start,
            end=end,
            confidence=round(float(confidence), 3),
            metadata=dict(metadata or {}),
        )
    )


def _iter_literal_matches(question: str, value: str) -> Iterable[re.Match[str]]:
    escaped = re.escape(value).replace(r"\ ", r"[\s_-]+")
    pattern = re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)
    return pattern.finditer(question)


def extract_anchors(question: str) -> AnchorExtractionResult:
    question = str(question or "")
    anchors: list[AnchorEvidence] = []
    seen: set[tuple[str, str, int, int]] = set()

    # 1. Known academic entities and phrases.
    for anchor_type, values in KNOWN_ANCHORS.items():
        for value in sorted(values, key=len, reverse=True):
            for match in _iter_literal_matches(question, value):
                _add_anchor(
                    anchors,
                    seen,
                    question=question,
                    text=match.group(0),
                    anchor_type=anchor_type,
                    start=match.start(),
                    end=match.end(),
                    confidence=1.0 if anchor_type in {"benchmark", "method", "model"} else 0.92,
                    metadata={"rule": "known_anchor"},
                )

    # 2. Negative constraints.
    for pattern, normalized_value in NEGATIVE_PATTERNS:
        for match in re.finditer(pattern, question, flags=re.IGNORECASE):
            _add_anchor(
                anchors,
                seen,
                question=question,
                text=match.group(0),
                anchor_type="negative_constraint",
                start=match.start(),
                end=match.end(),
                confidence=1.0,
                metadata={"normalized_value": normalized_value, "rule": "negative_constraint"},
            )

    # 3. Temporal constraints. Check ranges before single-year rules to retain relation.
    for pattern, relation in TEMPORAL_PATTERNS:
        for match in re.finditer(pattern, question, flags=re.IGNORECASE):
            years = [int(value) for value in match.groups() if value]
            _add_anchor(
                anchors,
                seen,
                question=question,
                text=match.group(0),
                anchor_type="temporal_constraint",
                start=match.start(),
                end=match.end(),
                confidence=1.0,
                metadata={"relation": relation, "years": years, "rule": "temporal_constraint"},
            )

    # 4. Comparison constraints and their endpoints.
    for pattern, relation in COMPARISON_PATTERNS:
        for match in re.finditer(pattern, question, flags=re.IGNORECASE):
            groups = [group.strip() for group in match.groups() if group and group.strip()]
            _add_anchor(
                anchors,
                seen,
                question=question,
                text=match.group(0),
                anchor_type="comparison",
                start=match.start(),
                end=match.end(),
                confidence=0.96,
                metadata={"relation": relation, "arguments": groups, "rule": "comparison"},
            )

    # 5. Explicit paper-title anchors.
    for match in TITLE_TRIGGER_RE.finditer(question):
        title = match.group(1).strip()
        title_start = match.start(1)
        title_end = match.end(1)
        _add_anchor(
            anchors,
            seen,
            question=question,
            text=title,
            anchor_type="paper_title",
            start=title_start,
            end=title_end,
            confidence=0.98,
            metadata={"rule": "title_trigger"},
        )

    for match in QUOTED_TEXT_RE.finditer(question):
        quoted = match.group(1).strip()
        if len(quoted.split()) >= 3:
            _add_anchor(
                anchors,
                seen,
                question=question,
                text=quoted,
                anchor_type="paper_title",
                start=match.start(1),
                end=match.end(1),
                confidence=0.82,
                metadata={"rule": "quoted_phrase"},
            )

    # 6. Unknown named entities: acronyms, CamelCase names and underscore identifiers.
    known_normalized = {anchor.normalized_text for anchor in anchors}
    for match in CAMEL_OR_ACRONYM_RE.finditer(question):
        text = match.group(0)
        normalized = _normalize(text)
        if normalized in known_normalized:
            continue
        if normalized in {"api", "pdf", "doi", "url", "ai", "ml", "nlp"}:
            continue
        _add_anchor(
            anchors,
            seen,
            question=question,
            text=text,
            anchor_type="named_entity",
            start=match.start(),
            end=match.end(),
            confidence=0.72,
            metadata={"rule": "camel_or_acronym"},
        )

    anchors.sort(key=lambda item: (item.start, -item.end, item.anchor_type))
    return AnchorExtractionResult(question=question, anchors=anchors)


def anchor_types(result: AnchorExtractionResult) -> set[str]:
    return {anchor.anchor_type for anchor in result.anchors}
