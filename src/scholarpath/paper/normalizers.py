from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>]+", re.I)
_ARXIV_NEW_RE = re.compile(r"\d{4}\.\d{4,5}", re.I)
_ARXIV_OLD_RE = re.compile(r"[a-z\-]+(?:\.[a-z\-]+)?/\d{7}", re.I)
_ARXIV_DOI_RE = re.compile(
    r"10\.48550/arxiv\.(?P<identifier>(?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[a-z\-]+)?/\d{7})(?:v\d+)?)",
    re.I,
)
_OPENALEX_RE = re.compile(r"(?:https?://(?:api\.)?openalex\.org/)?(W\d+)", re.I)
_S2_HEX_RE = re.compile(r"^[0-9a-f]{40}$", re.I)
_CORPUS_RE = re.compile(r"^(?:corpusid:?)?(\d+)$", re.I)
_TRAILING = " \t\r\n.,;:)]}>\"'"
_TRACKING_QUERY_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "ref",
    "source",
}


def _text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def extract_doi(value: object | None) -> str | None:
    """Extract and normalize a DOI from a DOI string, URL, or free text."""
    text = _text(value)
    if not text:
        return None
    text = unquote(html.unescape(text)).strip().rstrip(_TRAILING)
    doi_url_match = re.search(
        r"https?://(?:dx\.)?doi\.org/(?P<doi>10\.\d{4,9}/[^\s\"<>?#]+)",
        text,
        flags=re.I,
    )
    if doi_url_match:
        return doi_url_match.group("doi").rstrip(_TRAILING).lower()
    match = _DOI_RE.search(text)
    return match.group(0).rstrip(_TRAILING).lower() if match else None


def normalize_doi(value: object | None) -> str | None:
    text = _text(value)
    if not text:
        return None
    text = unquote(html.unescape(text))
    text = re.sub(r"^doi:\s*", "", text, flags=re.I)
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.I)
    return extract_doi(text)


def arxiv_id_from_doi(value: object | None) -> str | None:
    """Return the arXiv identifier encoded by a 10.48550/arXiv DOI."""
    doi = normalize_doi(value)
    if not doi:
        return None
    match = _ARXIV_DOI_RE.fullmatch(doi)
    if not match:
        return None
    return normalize_arxiv_id(match.group("identifier"))


def extract_arxiv_id(value: object | None) -> str | None:
    """Extract an arXiv ID without mistaking arbitrary DOI digits for arXiv."""
    text = _text(value)
    if not text:
        return None
    text = unquote(html.unescape(text)).strip()

    from_doi = arxiv_id_from_doi(text)
    if from_doi:
        return from_doi

    direct = re.sub(r"^arxiv:\s*", "", text, flags=re.I)
    direct = re.sub(r"\.pdf$", "", direct, flags=re.I)
    direct = direct.strip().rstrip(_TRAILING)
    direct = re.sub(r"v\d+$", "", direct, flags=re.I).lower()
    if _ARXIV_NEW_RE.fullmatch(direct) or _ARXIV_OLD_RE.fullmatch(direct):
        return direct

    url_match = re.match(
        r"^https?://(?:(?:export\.)?arxiv\.org/(?:abs|pdf|html)/|"
        r"ar5iv\.labs\.arxiv\.org/html/)(?P<identifier>[^?#]+)",
        text,
        flags=re.I,
    )
    if url_match:
        candidate = re.sub(r"\.pdf$", "", url_match.group("identifier"), flags=re.I)
        candidate = re.sub(r"v\d+$", "", candidate.strip().rstrip(_TRAILING), flags=re.I).lower()
        if _ARXIV_NEW_RE.fullmatch(candidate) or _ARXIV_OLD_RE.fullmatch(candidate):
            return candidate

    contextual = re.search(
        r"arxiv(?:\s*[:.]|\s+)(?P<identifier>"
        r"(?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[a-z\-]+)?/\d{7})(?:v\d+)?)",
        text,
        flags=re.I,
    )
    if contextual:
        candidate = re.sub(r"v\d+$", "", contextual.group("identifier"), flags=re.I).lower()
        return candidate
    return None


def normalize_arxiv_id(value: object | None) -> str | None:
    return extract_arxiv_id(value)


def extract_openalex_id(value: object | None) -> str | None:
    text = _text(value)
    if not text:
        return None
    match = _OPENALEX_RE.search(unquote(text))
    return match.group(1).upper() if match else None


def normalize_openalex_id(value: object | None) -> str | None:
    return extract_openalex_id(value)


def normalize_semantic_scholar_id(value: object | None) -> str | None:
    text = _text(value)
    if not text:
        return None
    text = unquote(text).strip().rstrip(_TRAILING)
    text = re.sub(
        r"^https?://www\.semanticscholar\.org/paper/(?:[^/]+/)?",
        "",
        text,
        flags=re.I,
    )
    if _S2_HEX_RE.fullmatch(text):
        return text.lower()
    match = _CORPUS_RE.fullmatch(text)
    if match:
        return f"CorpusID:{match.group(1)}"
    return text


def normalize_url(value: object | None) -> str | None:
    """Canonicalize common scholarly URLs without performing network access."""
    text = _text(value)
    if not text:
        return None
    text = html.unescape(text).strip()
    doi = extract_doi(text)
    if doi and re.match(r"^https?://(?:dx\.)?doi\.org/", text, flags=re.I):
        return f"https://doi.org/{doi}"
    arxiv_id = extract_arxiv_id(text)
    if arxiv_id and re.match(
        r"^https?://(?:(?:export\.)?arxiv\.org|ar5iv\.labs\.arxiv\.org)/",
        text,
        flags=re.I,
    ):
        return f"https://arxiv.org/abs/{arxiv_id}"
    openalex_id = extract_openalex_id(text)
    if openalex_id and re.match(r"^https?://(?:api\.)?openalex\.org/", text, flags=re.I):
        return f"https://openalex.org/{openalex_id}"

    try:
        parts = urlsplit(text)
    except ValueError:
        return text.rstrip("/")
    if not parts.scheme or not parts.netloc:
        return text.rstrip("/")
    scheme = parts.scheme.lower()
    host = parts.netloc.lower()
    path = re.sub(r"/{2,}", "/", parts.path).rstrip("/")
    query_items = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_QUERY_KEYS
    ]
    query = urlencode(sorted(query_items))
    return urlunsplit((scheme, host, path, query, ""))


def normalize_title(value: object | None) -> str:
    text = _text(value)
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", html.unescape(text)).casefold()
    text = re.sub(r"\\(?:textit|textbf|mathrm|mathbf|emph)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"[$\\{}]", " ", text)
    out: list[str] = []
    for char in text:
        if char.isalnum():
            out.append(char)
        elif unicodedata.category(char).startswith(("P", "S", "Z")):
            out.append(" ")
        else:
            out.append(char)
    return " ".join("".join(out).split())


def compact_title_key(value: object | None) -> str:
    return "".join(c for c in normalize_title(value) if c.isalnum())


def normalize_author_name(value: object | None) -> str:
    text = _text(value)
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", html.unescape(text)).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def first_author_key(authors: list[str] | tuple[str, ...] | None) -> str:
    return normalize_author_name(authors[0]) if authors else ""


def title_fingerprint(value: object | None, length: int = 16) -> str:
    return hashlib.sha1(normalize_title(value).encode("utf-8")).hexdigest()[:length]
