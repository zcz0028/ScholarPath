from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(slots=True)
class QueryVariant:
    text: str
    variant_type: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(slots=True)
class RewriteConfig:
    max_variants: int = 5
    max_keyword_terms: int = 12
    min_token_length: int = 2


QUESTION_PREFIX_PATTERNS = (
    r"^\s*is\s+there\s+any\s+work\s+that\s+",
    r"^\s*is\s+there\s+any\s+paper\s+that\s+",
    r"^\s*are\s+there\s+any\s+works?\s+that\s+",
    r"^\s*are\s+there\s+any\s+papers?\s+that\s+",
    r"^\s*find\s+(?:me\s+)?(?:papers?|works?)\s+(?:that|about|on)\s+",
    r"^\s*give\s+me\s+(?:papers?|works?)\s+(?:that|about|on)\s+",
    r"^\s*what\s+are\s+(?:some\s+)?(?:papers?|works?)\s+(?:that|about|on)\s+",
    r"^\s*which\s+(?:papers?|works?)\s+(?:study|analyze|focus\s+on|about|on)\s+",
    r"^\s*show\s+me\s+(?:papers?|works?)\s+(?:that|about|on)\s+",
)

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "with",
    "without", "by", "from", "into", "over", "under", "between", "among",
    "using", "use", "uses", "used", "based", "via", "through", "that",
    "which", "what", "when", "where", "who", "whose", "there", "their",
    "these", "those", "this", "such", "as", "any", "some", "work", "works",
    "paper", "papers", "study", "studies", "analyze", "analyzes",
    "analysis", "method", "methods", "model", "models", "approach",
    "approaches", "task", "tasks", "can", "could", "would", "should",
    "does", "do", "did", "be", "is", "are", "was", "were", "been",
    "have", "has", "had", "about", "also", "including", "include",
    "includes", "related", "research",
}

PHRASE_EXPANSIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("scaling law", ("scaling laws", "scaling behavior")),
    ("multi module", ("multi module model", "multimodal model", "multi modal model")),
    ("multi modal", ("multimodal", "multimodal model")),
    ("multimodal", ("multi modal", "multimodal model")),
    ("image text", ("image language", "vision language", "vision language model")),
    ("video text", ("video language", "video language model", "multimodal video")),
    ("text to sql", ("text-to-sql", "nl2sql", "natural language to sql")),
    ("retrieval augmented generation", ("RAG", "retrieval augmented generation")),
    ("large language model", ("LLM", "large language models")),
    ("vision language", ("vision-language", "vision language model")),
    ("chain of thought", ("CoT", "chain-of-thought")),
    ("reinforcement learning", ("RL", "reinforcement learning")),
)


def normalize_space(text: object | None) -> str:
    value = str(text or "")
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def clean_question_text(question: object | None) -> str:
    text = normalize_space(question)
    text = text.strip(" \t\r\n?。？")
    lowered = text.casefold()

    for pattern in QUESTION_PREFIX_PATTERNS:
        if re.match(pattern, lowered, flags=re.IGNORECASE):
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
            break

    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", text)
    text = text.replace("/", " ")
    text = re.sub(r"[?？!！,，;；:：()\[\]{}<>]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_hyphenated_terms(text: str) -> str:
    return re.sub(r"(?<=\w)-(?=\w)", " ", text)


def tokenize_for_keywords(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text)
    text = split_hyphenated_terms(text)
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9]*|\d+[A-Za-z]*", text)
    return tokens


def extract_keywords(
    question: str,
    *,
    max_terms: int = 12,
    min_token_length: int = 2,
) -> list[str]:
    tokens = tokenize_for_keywords(question)
    keywords: list[str] = []
    seen: set[str] = set()

    for token in tokens:
        normalized = token.casefold()
        if normalized in STOPWORDS:
            continue
        if len(normalized) < min_token_length:
            continue

        # Keep original uppercase acronyms, otherwise use lower case.
        value = token if token.isupper() and len(token) <= 8 else normalized
        if value not in seen:
            keywords.append(value)
            seen.add(value)
        if len(keywords) >= max_terms:
            break
    return keywords


def extract_salient_phrases(question: str) -> list[str]:
    cleaned = clean_question_text(question).casefold()
    cleaned = split_hyphenated_terms(cleaned)
    phrases: list[str] = []

    for phrase, _expansions in PHRASE_EXPANSIONS:
        if phrase in cleaned and phrase not in phrases:
            phrases.append(phrase)

    # Preserve common technical forms containing numbers or capitals.
    raw = normalize_space(question)
    for match in re.findall(r"\b[A-Za-z]+-\d+[A-Za-z]*\b|\b[A-Z]{2,}[A-Za-z0-9]*\b", raw):
        value = match.strip()
        if value and value not in phrases:
            phrases.append(value)

    return phrases


def expand_terms(text: str) -> list[str]:
    lowered = split_hyphenated_terms(clean_question_text(text).casefold())
    expansions: list[str] = []
    seen: set[str] = set()

    for phrase, candidates in PHRASE_EXPANSIONS:
        if phrase not in lowered:
            continue
        for candidate in candidates:
            key = candidate.casefold()
            if key not in seen:
                expansions.append(candidate)
                seen.add(key)
    return expansions


def compact_query_terms(terms: Iterable[str], limit: int = 16) -> str:
    output: list[str] = []
    seen: set[str] = set()
    for term in terms:
        value = normalize_space(term)
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        output.append(value)
        seen.add(key)
        if len(output) >= limit:
            break
    return " ".join(output)


class QueryRewriter:
    def __init__(self, config: RewriteConfig | None = None) -> None:
        self.config = config or RewriteConfig()

    def rewrite(self, question: str) -> list[QueryVariant]:
        original = normalize_space(question)
        cleaned = clean_question_text(original)
        split_cleaned = split_hyphenated_terms(cleaned)
        keywords = extract_keywords(
            split_cleaned,
            max_terms=self.config.max_keyword_terms,
            min_token_length=self.config.min_token_length,
        )
        phrases = extract_salient_phrases(original)
        expansions = expand_terms(original)

        candidates: list[QueryVariant] = [
            QueryVariant(
                text=original,
                variant_type="raw",
                reason="Original benchmark query.",
            )
        ]

        if cleaned and cleaned.casefold() != original.casefold():
            candidates.append(
                QueryVariant(
                    text=cleaned,
                    variant_type="cleaned",
                    reason="Removed question-style prefix and punctuation.",
                )
            )

        if keywords:
            keyword_query = compact_query_terms(phrases + keywords, limit=12)
            if keyword_query:
                candidates.append(
                    QueryVariant(
                        text=keyword_query,
                        variant_type="keyword_core",
                        reason="Core technical keywords extracted from query.",
                    )
                )

        if expansions:
            expanded_query = compact_query_terms(
                phrases + keywords[:8] + expansions,
                limit=16,
            )
            if expanded_query:
                candidates.append(
                    QueryVariant(
                        text=expanded_query,
                        variant_type="expanded",
                        reason="Added deterministic synonym and alias expansions.",
                    )
                )

        if phrases:
            phrase_query = compact_query_terms(phrases + expansions, limit=10)
            if phrase_query:
                candidates.append(
                    QueryVariant(
                        text=phrase_query,
                        variant_type="phrase_focus",
                        reason="Focused search using salient phrases and aliases.",
                    )
                )

        return self._deduplicate(candidates)[: self.config.max_variants]

    @staticmethod
    def _deduplicate(candidates: list[QueryVariant]) -> list[QueryVariant]:
        output: list[QueryVariant] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = normalize_space(candidate.text).casefold()
            if not key or key in seen:
                continue
            output.append(candidate)
            seen.add(key)
        return output
