from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

from scholarpath.query.rewriter import (
    clean_question_text,
    compact_query_terms,
    expand_terms,
    extract_salient_phrases,
    normalize_space,
    split_hyphenated_terms,
)


@dataclass(slots=True)
class QueryConstraint:
    id: str
    text: str
    constraint_type: str
    terms: list[str]
    weight: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class SubQuery:
    text: str
    subquery_type: str
    constraint_ids: list[str]
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class ConstraintDecomposition:
    question: str
    cleaned_question: str
    constraints: list[QueryConstraint]
    subqueries: list[SubQuery]

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "cleaned_question": self.cleaned_question,
            "constraints": [item.to_dict() for item in self.constraints],
            "subqueries": [item.to_dict() for item in self.subqueries],
        }


PROMPT_PREFIX_PATTERNS = (
    r"^\s*give\s+me\s+(?:papers?|works?)\s+(?:which|that|about|on)\s+",
    r"^\s*find\s+(?:me\s+)?(?:papers?|works?)\s+(?:which|that|about|on)\s+",
    r"^\s*show\s+me\s+(?:papers?|works?)\s+(?:which|that|about|on)\s+",
    r"^\s*what\s+are\s+(?:some\s+)?(?:papers?|works?)\s+(?:which|that|about|on)\s+",
    r"^\s*is\s+there\s+any\s+(?:work|paper)\s+(?:which|that)\s+",
    r"^\s*are\s+there\s+any\s+(?:works|papers)\s+(?:which|that)\s+",
    r"^\s*could\s+you\s+(?:please\s+)?(?:list|find|show|provide)\s+(?:me\s+)?(?:some\s+)?(?:papers?|works?|research)\s+(?:which|that|about|on)?\s*",
    r"^\s*can\s+you\s+(?:please\s+)?(?:help\s+me\s+)?(?:list|find|show|provide)\s+(?:me\s+)?(?:some\s+)?(?:papers?|works?|research)\s+(?:which|that|about|on)?\s*",
    r"^\s*i\s+(?:would\s+like|want)\s+to\s+know\s+(?:about|how|whether|if)?\s*",
    r"^\s*please\s+(?:list|find|show|provide)\s+(?:me\s+)?(?:some\s+)?(?:papers?|works?|research)\s+(?:which|that|about|on)?\s*",
)

PROMPT_NOISE_WORDS = {
    "give", "me", "paper", "papers", "work", "works", "which", "that", "show",
    "shows", "showing", "find", "result", "results", "using", "use", "used",
    "can", "could", "than", "any", "some", "there", "about", "related",
}


# Day14-1C: deterministic constraint-quality repair.
#
# These rules are deliberately conservative:
# - protect short academic entities/acronyms;
# - reject obvious discourse residue and dangling phrase fragments;
# - extract a few relation/capability spans from the full cleaned query before
#   accepting generic salient-phrase chunks.
#
# No network, LLM, embedding service, or gold label is used.
ACADEMIC_SHORT_TERMS: dict[str, str] = {
    "ai": "model_or_entity",
    "ee": "task_or_modality",
    "gnn": "method_or_property",
    "gnns": "method_or_property",
    "hotpotqa": "task_or_modality",
    "imo": "task_or_modality",
    "llm": "model_or_entity",
    "llms": "model_or_entity",
    "ner": "task_or_modality",
    "qat": "method_or_property",
    "rag": "method_or_property",
    "re": "task_or_modality",
    "rlhf": "method_or_property",
    "sft": "method_or_property",
}

_CANDIDATE_LEADING_NOISE_PATTERNS = (
    r"^(?:supporting|support|providing|provide)\s+(?:the\s+)?claim(?:\s+that)?\s+",
    r"^(?:evidence|papers?|works?)\s+(?:supporting|showing)\s+(?:that\s+)?",
    r"^explaining\s+why\s+",
    r"^(?:i\s+)?(?:am\s+)?looking\s+for\s+(?:research\s+)?(?:papers?|works?)?\s*",
    r"^(?:i\s+)?(?:would\s+like|want)\s+to\s+(?:know|find)\s+(?:how|whether|if|about|some)?\s*",
    r"^(?:could|can)\s+you\s+(?:please\s+)?(?:help\s+me\s+)?",
    r"^(?:help\s+me\s+)?search\s+for\s+(?:the\s+)?(?:work|works|papers?|research)\s+(?:related\s+to)?\s*",
    r"^(?:list|find|show|provide)\s+(?:me\s+)?(?:all\s+|some\s+)?(?:papers?|works?|research)\s*",
    r"^(?:what\s+are\s+the\s+)?(?:researches|research)\s+(?:that\s+)?",
    r"^(?:do\s+you\s+)?know\s+(?:some\s+)?",
    r"^(?:note\s+that\s+)?i\s+am\s+referring\s+to\s+",
    r"^is\s+there\s+(?:any\s+)?",
)

_DANGLING_FINAL_TOKENS = {
    "a", "an", "and", "as", "between", "by", "claim", "for", "from",
    "have", "has", "in", "inductive", "into", "knowledgeable", "multiple",
    "of", "on", "or", "such", "supporting", "systematically", "the", "to",
    "using", "via", "with",
}

_GENERIC_TOPIC_TOKENS = {
    "analysis", "approach", "data", "document", "learning", "method", "model",
    "network", "paper", "prediction", "research", "result", "study", "system",
    "task", "relationship", "relationships",
}

METHOD_PATTERNS = (
    "scaling law",
    "scaling laws",
    "chain of thought",
    "retrieval augmented generation",
    "reinforcement learning",
    "in context learning",
    "instruction tuning",
    "fine tuning",
    "parameter efficient",
    "knowledge distillation",
    "contrastive learning",
    "self supervised",
    "graph neural network",
    "causal inference",
    "pre training",
    "pretraining",
    "pre train",
)

TASK_OR_MODALITY_PATTERNS = (
    "video text",
    "image text",
    "vision language",
    "video language",
    "multi module",
    "multi modal",
    "multimodal",
    "text to sql",
    "natural language to sql",
    "question answering",
    "semantic segmentation",
    "object detection",
    "information retrieval",
    "recommendation",
    "machine translation",
    "speech recognition",
    "code generation",
)

MODEL_OR_ENTITY_PATTERNS = (
    "large language model",
    "large language models",
    "language model",
    "language models",
    "llm",
    "bert",
    "gpt",
    "clip",
    "transformer",
    "diffusion model",
    "stable diffusion",
    "resnet",
    "yolo",
    "rtdetr",
    "rt detr",
)

DATA_SCALE_PATTERNS = (
    "smaller dataset",
    "small dataset",
    "less data",
    "fewer data",
    "bigger dataset",
    "larger dataset",
    "more data",
    "dataset size",
    "data efficient",
    "data efficiency",
    "sample efficiency",
)

PERFORMANCE_PATTERNS = (
    "better model",
    "better models",
    "better performance",
    "improved performance",
    "outperform",
    "outperforms",
)


def normalize_for_match(text: object | None) -> str:
    value = normalize_space(text)
    value = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212-]", " ", value)
    value = value.casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def strip_prompt_prefix(text: str) -> str:
    cleaned = normalize_space(text)
    for pattern in PROMPT_PREFIX_PATTERNS:
        if re.match(pattern, cleaned, flags=re.IGNORECASE):
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
            break

    cleaned = re.sub(r"^\s*(?:that\s+)?(?:using|use|used)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(" ?。？!！")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def slugify(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")
    return value[:40] or "constraint"


def unique_list(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_space(value)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        output.append(normalized)
        seen.add(key)
    return output


def phrase_terms(phrase: str) -> list[str]:
    cleaned = normalize_for_match(phrase)
    terms = re.findall(r"[A-Za-z][A-Za-z0-9]*|\d+[A-Za-z]*", cleaned)
    return unique_list(term for term in terms if term not in PROMPT_NOISE_WORDS)


def find_patterns(text: str, patterns: Iterable[str]) -> list[str]:
    lowered = normalize_for_match(text)
    hits: list[str] = []
    for pattern in patterns:
        pattern_norm = normalize_for_match(pattern)
        if re.search(rf"(?<![a-z0-9]){re.escape(pattern_norm)}s?(?![a-z0-9])", lowered):
            hits.append(pattern)
    return unique_list(hits)


def extract_clean_keywords(text: str, *, max_terms: int = 12) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9]*|\d+[A-Za-z]*", normalize_for_match(text))
    keywords: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in PROMPT_NOISE_WORDS:
            continue
        if len(token) < 2:
            continue
        if token not in seen:
            keywords.append(token)
            seen.add(token)
        if len(keywords) >= max_terms:
            break
    return keywords



def extract_protected_academic_terms(text: str) -> list[tuple[str, str]]:
    """Return short academic terms that must survive generic short-phrase filters."""
    normalized = normalize_for_match(text)
    hits: list[tuple[str, str]] = []
    for term, constraint_type in ACADEMIC_SHORT_TERMS.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", normalized):
            hits.append((term.upper() if term.isalpha() and len(term) <= 6 else term, constraint_type))
    return hits


def clean_candidate_phrase(text: str) -> str:
    """Remove prompt/discourse residue from one extracted candidate phrase."""
    value = normalize_space(text).strip(" ?。？!！,，;；:：()[]{}<>")
    if not value:
        return ""

    for pattern in _CANDIDATE_LEADING_NOISE_PATTERNS:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE).strip()

    value = re.sub(r"^(?:that|which)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
    value = _strip_candidate_trailing_noise(value) if "_strip_candidate_trailing_noise" in globals() else value
    return value



_CANDIDATE_TRAILING_NOISE_PATTERNS = (
    r"\s+note$",
    r"\s+do$",
    r"\s+it$",
)

_HARD_FRAGMENT_EXACT = {
    "do not",
    "demonstrate how",
    "have explored",
    "is there",
    "i would like",
    "i am referring",
    "do you",
    "cutting edge",
}

_HARD_FRAGMENT_PREFIXES = (
    "am looking for research",
    "what are the researches",
)

_HARD_FRAGMENT_SUFFIXES = (
    " method do",
    " supervised fine",
    " based agent",
    " financial task note",
)

def _strip_candidate_trailing_noise(text: str) -> str:
    value = normalize_space(text)
    for pattern in _CANDIDATE_TRAILING_NOISE_PATTERNS:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE).strip()
    return value

def is_hard_fragment(text: str) -> bool:
    normalized = normalize_for_match(text)
    if not normalized:
        return True
    if normalized in _HARD_FRAGMENT_EXACT:
        return True
    if any(normalized.startswith(prefix) for prefix in _HARD_FRAGMENT_PREFIXES):
        return True
    if any(normalized.endswith(suffix) for suffix in _HARD_FRAGMENT_SUFFIXES):
        return True
    tokens = normalized.split()
    if tokens[-1] in {"between", "multiple", "inductive"}:
        return True
    if "between" in tokens and tokens[-1] in {"multiple", "several", "different"}:
        return True
    return False

def suppress_redundant_candidates(
    values: list[tuple[str, str | None]],
) -> list[tuple[str, str | None]]:
    cleaned: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    for text, ctype in values:
        repaired = clean_candidate_phrase(text)
        repaired = _strip_candidate_trailing_noise(repaired)
        key = normalize_for_match(repaired)
        if not key or key in seen or is_hard_fragment(repaired):
            continue
        cleaned.append((repaired, ctype))
        seen.add(key)

    keys = [normalize_for_match(text) for text, _ in cleaned]
    output: list[tuple[str, str | None]] = []
    for idx, (text, ctype) in enumerate(cleaned):
        key = keys[idx]
        tokens = set(key.split())
        redundant = False
        for j, other_key in enumerate(keys):
            if j == idx or not other_key or other_key == key:
                continue
            other_tokens = set(other_key.split())
            if not other_tokens or not other_tokens < tokens:
                continue
            remainder = tokens - other_tokens
            if not remainder:
                continue
            for k, third_key in enumerate(keys):
                if k in {idx, j} or not third_key:
                    continue
                third_tokens = set(third_key.split())
                if remainder <= third_tokens:
                    redundant = True
                    break
            if redundant:
                break
        if not redundant:
            output.append((text, ctype))
    return output


def is_semantically_complete_candidate(
    text: str,
    constraint_type: str | None = None,
) -> bool:
    """Reject obvious fragments while protecting concise academic constraints.

    IMPORTANT:
    Semantic-completeness checks must use the actual normalized phrase tokens,
    not phrase_terms(). phrase_terms() intentionally removes prompt/noise words
    such as "paper"/"papers", but those words can be legitimate relation
    objects in constraints like:

        analyze relationships between multiple papers

    Removing "papers" would incorrectly turn that complete relation into the
    dangling fragment:

        analyze relationships between multiple
    """
    cleaned = clean_candidate_phrase(text)
    normalized = normalize_for_match(cleaned)
    tokens = normalized.split()

    if not tokens:
        return False

    if is_hard_fragment(cleaned):
        return False

    # Protect known concise academic entities / abbreviations.
    if normalized in ACADEMIC_SHORT_TERMS:
        return True

    ctype = constraint_type or infer_constraint_type(cleaned)

    # Named entities, methods and explicit tasks may legitimately be short.
    if ctype in {
        "model_or_entity",
        "method_or_property",
        "task_or_modality",
    }:
        return True

    # A single generic topic token is not a useful research constraint.
    if len(tokens) == 1:
        return False

    # Detect genuinely dangling phrase endings using the complete phrase,
    # without deleting legitimate nouns such as paper/document first.
    if tokens[-1] in _DANGLING_FINAL_TOKENS:
        return False

    # Relation phrases such as "relationships between multiple" are
    # incomplete until their semantic object is present.
    if "between" in tokens and tokens[-1] in {
        "between",
        "multiple",
        "several",
        "different",
    }:
        return False

    if "among" in tokens and tokens[-1] in {
        "among",
        "multiple",
        "several",
        "different",
    }:
        return False

    # Short topic chunks composed entirely of broad/generic research words
    # remain too weak to promote to standalone constraints.
    if (
        len(tokens) <= 3
        and all(token in _GENERIC_TOPIC_TOKENS for token in tokens)
    ):
        return False

    return True



def extract_relation_constraints(text: str) -> list[tuple[str, str]]:
    normalized = normalize_for_match(text)
    output: list[tuple[str, str]] = []

    patterns: tuple[tuple[str, str], ...] = (
        (
            r"\b(?:llms?|large language models?)\s+"
            r"(?:have|has|possess|possesses|demonstrate|demonstrates)\s+"
            r"(?:sufficient\s+)?(?:inductive\s+)?"
            r"(?:capacity|capability|capabilities|ability|abilities|reasoning)\b",
            "method_or_property",
        ),
        (
            r"\b(?:analyze|analyse|understand|model|capture)\s+"
            r"(?:the\s+)?relationships?\s+(?:between|among)\s+"
            r"(?:multiple|several|different)\s+"
            r"(?:papers?|documents?|studies?|works?)\b",
            "topic",
        ),
        (
            r"\brelationships?\s+(?:between|among)\s+"
            r"(?:multiple|several|different)\s+"
            r"(?:papers?|documents?|studies?|works?)\b",
            "topic",
        ),
        (
            r"\b(?:systematically|automatically)\s+"
            r"(?:write|generate|produce|compose)\s+"
            r"(?:a\s+|an\s+)?"
            r"(?:survey|review|literature review|systematic review)\b",
            "topic",
        ),

        # Day14-1C-2: preserve discriminative academic task/topic phrases
        # that generic salient-phrase extraction can otherwise split apart.
        (
            r"\bmining\s+factors?\b",
            "topic",
        ),
        (
            r"\b(?:llm\s+generated|generated)\s+text"
            r"(?:\s+(?:detection|detector|classification))?\b",
            "task_or_modality",
        ),
    )

    for pattern, ctype in patterns:
        for match in re.finditer(pattern, normalized):
            phrase = clean_candidate_phrase(match.group(0))
            if phrase and is_semantically_complete_candidate(
                phrase,
                ctype,
            ):
                output.append((phrase, ctype))

    return unique_list_pairs(output)

def unique_list_pairs(values: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for text, ctype in values:
        key = (normalize_for_match(text), ctype)
        if not key[0] or key in seen:
            continue
        output.append((text, ctype))
        seen.add(key)
    return output

def infer_constraint_type(text: str) -> str:
    lowered = normalize_for_match(text)
    if lowered in ACADEMIC_SHORT_TERMS:
        return ACADEMIC_SHORT_TERMS[lowered]
    if any(token in lowered for token in ("smaller", "small", "less", "fewer", "bigger", "larger", "more", "dataset size", "data efficient")):
        return "data_condition"
    if any(token in lowered for token in ("better", "outperform", "performance", "improved")):
        return "performance_relation"
    if any(token in lowered for token in ("scaling", "retrieval", "generation", "learning", "tuning", "distillation", "pretraining", "pre training")):
        return "method_or_property"
    if any(token in lowered for token in ("video", "image", "vision", "language", "sql", "question", "segmentation", "detection")):
        return "task_or_modality"
    if any(token in lowered for token in ("llm", "gpt", "bert", "clip", "transformer", "diffusion", "yolo", "rtdetr")):
        return "model_or_entity"
    return "topic"


def build_constraint(index: int, text: str, constraint_type: str | None = None) -> QueryConstraint:
    ctype = constraint_type or infer_constraint_type(text)
    terms = phrase_terms(text)
    weight = {
        "data_condition": 1.30,
        "performance_relation": 1.30,
        "method_or_property": 1.25,
        "task_or_modality": 1.15,
        "model_or_entity": 1.10,
        "topic": 1.0,
    }.get(ctype, 1.0)
    return QueryConstraint(
        id=f"c{index}_{slugify(text)}",
        text=normalize_space(text),
        constraint_type=ctype,
        terms=terms,
        weight=weight,
    )


def extract_comparative_constraints(text: str) -> list[tuple[str, str]]:
    lowered = normalize_for_match(text)
    constraints: list[tuple[str, str]] = []

    if re.search(r"(smaller|small|less|fewer).{0,25}(dataset|data)", lowered):
        constraints.append(("smaller dataset", "data_condition"))
    if re.search(r"(bigger|larger|more).{0,25}(dataset|data)", lowered):
        constraints.append(("bigger dataset", "data_condition"))
    if re.search(r"(better|outperform|outperforms|improved).{0,35}(model|performance)", lowered) or "better models" in lowered:
        constraints.append(("better models", "performance_relation"))
    if (
        re.search(r"(smaller|small|less|fewer).{0,40}(better|outperform|improved)", lowered)
        or re.search(r"better.{0,40}(bigger|larger|more)", lowered)
    ):
        constraints.append(("smaller dataset better than bigger dataset", "performance_relation"))
    return constraints


class ConstraintDecomposer:
    def __init__(self, *, max_constraints: int = 6, max_subqueries: int = 6) -> None:
        self.max_constraints = max_constraints
        self.max_subqueries = max_subqueries

    def decompose(self, question: str) -> ConstraintDecomposition:
        original = normalize_space(question)
        cleaned = self._clean_for_constraints(original)
        constraints = self._extract_constraints(original, cleaned)
        subqueries = self._build_subqueries(original, cleaned, constraints)
        return ConstraintDecomposition(
            question=original,
            cleaned_question=cleaned,
            constraints=constraints,
            subqueries=subqueries,
        )

    def _clean_for_constraints(self, question: str) -> str:
        cleaned = clean_question_text(question)
        cleaned = strip_prompt_prefix(cleaned)
        cleaned = re.sub(
            r"^\s*(?:supporting|support|providing)\s+(?:the\s+)?claim\s+(?:that\s+)?",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^\s*(?:i\s+)?(?:want|would\s+like)\s+to\s+know\s+(?:how|whether|if|about)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212-]", " ", cleaned)
        cleaned = re.sub(r"[?？!！,，;；:：()\[\]{}<>]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _extract_constraints(self, original: str, cleaned: str) -> list[QueryConstraint]:
        candidates: list[tuple[str, str | None]] = []

        # Comparative/data-scale constraints are central for queries like
        # "smaller dataset can be better than bigger datasets".
        candidates.extend(extract_comparative_constraints(original))

        # High-precision known phrases.
        # Put training/method constraints before broad model aliases so that
        # important constraints such as "pre training" are not pushed out by
        # redundant phrases like "language model" when max_constraints is small.
        for phrase in find_patterns(original, METHOD_PATTERNS):
            candidates.append((phrase, "method_or_property"))

        model_phrases = find_patterns(original, MODEL_OR_ENTITY_PATTERNS)
        model_phrase_keys = {normalize_for_match(item) for item in model_phrases}
        if "large language model" in model_phrase_keys or "large language models" in model_phrase_keys:
            model_phrases = [
                item for item in model_phrases
                if normalize_for_match(item) not in {"language model", "language models"}
            ]
        for phrase in model_phrases:
            candidates.append((phrase, "model_or_entity"))

        for phrase in find_patterns(original, DATA_SCALE_PATTERNS):
            candidates.append((phrase, "data_condition"))
        for phrase in find_patterns(original, PERFORMANCE_PATTERNS):
            candidates.append((phrase, "performance_relation"))
        for phrase in find_patterns(original, TASK_OR_MODALITY_PATTERNS):
            candidates.append((phrase, "task_or_modality"))

        # Recover complete relation/capability spans before generic salient chunks.
        candidates.extend(extract_relation_constraints(cleaned))

        # Preserve short but meaningful academic acronyms/entities such as RLHF,
        # QAT, SFT, NER and HotPotQA.
        candidates.extend(extract_protected_academic_terms(original))

        # Reuse salient phrases from B1 only after deterministic quality repair.
        # Obvious fragments are skipped rather than promoted to canonical constraints.
        for phrase in extract_salient_phrases(cleaned):
            repaired = clean_candidate_phrase(phrase)
            if not repaired:
                continue
            ctype = infer_constraint_type(repaired)
            if not is_semantically_complete_candidate(repaired, ctype):
                continue
            candidates.append((repaired, ctype))

        # Add compact keyword groups only when phrase extraction is sparse.
        keywords = extract_clean_keywords(cleaned, max_terms=12)
        if len(candidates) < 3 and keywords:
            primary = compact_query_terms(keywords[:4], limit=4)
            if primary and is_semantically_complete_candidate(primary, "topic"):
                candidates.append((primary, "topic"))
        if len(candidates) < 4 and len(keywords) >= 6:
            secondary = compact_query_terms(keywords[4:8], limit=4)
            if secondary and is_semantically_complete_candidate(secondary, "topic"):
                candidates.append((secondary, "topic"))

        # Add expansion phrases as lower-priority constraints only if sparse.
        if len(candidates) < 4:
            for expansion in expand_terms(original):
                candidates.append((expansion, None))
                if len(candidates) >= 4:
                    break

        candidates = suppress_redundant_candidates(candidates)

        unique_candidates: list[tuple[str, str | None]] = []
        seen: set[str] = set()
        for text, ctype in candidates:
            repaired = clean_candidate_phrase(text)
            if not repaired:
                continue
            effective_type = ctype or infer_constraint_type(repaired)
            if not is_semantically_complete_candidate(repaired, effective_type):
                continue
            key = normalize_for_match(repaired)
            if not key or key in seen:
                continue
            unique_candidates.append((repaired, effective_type))
            seen.add(key)
            if len(unique_candidates) >= self.max_constraints:
                break

        return [
            build_constraint(index + 1, text, ctype)
            for index, (text, ctype) in enumerate(unique_candidates)
        ]

    def _build_subqueries(
        self,
        original: str,
        cleaned: str,
        constraints: list[QueryConstraint],
    ) -> list[SubQuery]:
        subqueries: list[SubQuery] = []

        if cleaned:
            subqueries.append(
                SubQuery(
                    text=cleaned,
                    subquery_type="full_cleaned",
                    constraint_ids=[item.id for item in constraints],
                    reason="Full cleaned query preserving the original intent.",
                )
            )

        keywords = extract_clean_keywords(cleaned, max_terms=10)
        if keywords:
            keyword_query = compact_query_terms(keywords, limit=10)
            subqueries.append(
                SubQuery(
                    text=keyword_query,
                    subquery_type="keyword_fallback",
                    constraint_ids=[item.id for item in constraints[:3]],
                    reason="Fallback compact keyword query.",
                )
            )

        if constraints:
            primary = self._choose_primary_constraint(constraints)
            secondary = [item for item in constraints if item.id != primary.id]

            # Core query: primary + most important secondary constraints.
            core_terms = [primary.text] + [item.text for item in secondary[:3]]
            core_query = compact_query_terms(core_terms, limit=14)
            if core_query:
                subqueries.append(
                    SubQuery(
                        text=core_query,
                        subquery_type="constraint_core",
                        constraint_ids=[primary.id] + [item.id for item in secondary[:3]],
                        reason="Primary constraint combined with major secondary constraints.",
                    )
                )

            # Pairwise queries help avoid overly long OpenAlex search strings.
            for item in secondary[:4]:
                pair_query = compact_query_terms([primary.text, item.text], limit=8)
                if pair_query:
                    subqueries.append(
                        SubQuery(
                            text=pair_query,
                            subquery_type="pairwise_constraint",
                            constraint_ids=[primary.id, item.id],
                            reason="Pairwise query for a primary-secondary constraint pair.",
                        )
                    )

            # If no secondary constraints exist, search the primary alone.
            if not secondary:
                subqueries.append(
                    SubQuery(
                        text=primary.text,
                        subquery_type="single_constraint",
                        constraint_ids=[primary.id],
                        reason="Single primary constraint query.",
                    )
                )

        return self._deduplicate_subqueries(subqueries)[: self.max_subqueries]

    @staticmethod
    def _choose_primary_constraint(constraints: list[QueryConstraint]) -> QueryConstraint:
        priority = {
            "performance_relation": 0,
            "data_condition": 1,
            "method_or_property": 2,
            "task_or_modality": 3,
            "model_or_entity": 4,
            "topic": 5,
        }
        return sorted(
            constraints,
            key=lambda item: (priority.get(item.constraint_type, 9), -item.weight, item.id),
        )[0]

    @staticmethod
    def _deduplicate_subqueries(subqueries: list[SubQuery]) -> list[SubQuery]:
        output: list[SubQuery] = []
        seen: set[str] = set()
        for item in subqueries:
            key = normalize_for_match(item.text)
            if not key or key in seen:
                continue
            output.append(item)
            seen.add(key)
        return output
