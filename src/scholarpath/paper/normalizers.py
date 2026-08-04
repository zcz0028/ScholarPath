from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from urllib.parse import unquote

_DOI_RE = re.compile(r"10\.\d{4,9}/\S+", re.I)
_ARXIV_NEW_RE = re.compile(r"^\d{4}\.\d{4,5}$")
_ARXIV_OLD_RE = re.compile(r"^[a-z\-]+(?:\.[a-z\-]+)?/\d{7}$", re.I)
_OPENALEX_RE = re.compile(r"W\d+", re.I)
_S2_HEX_RE = re.compile(r"^[0-9a-f]{40}$", re.I)
_CORPUS_RE = re.compile(r"^(?:corpusid:?)?(\d+)$", re.I)
_TRAILING = " \t\r\n.,;:)]}>\"'"


def _text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_doi(value: object | None) -> str | None:
    text = _text(value)
    if not text:
        return None
    text = unquote(html.unescape(text))
    text = re.sub(r"^doi:\s*", "", text, flags=re.I)
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.I)
    match = _DOI_RE.search(text.rstrip(_TRAILING))
    return match.group(0).rstrip(_TRAILING).lower() if match else None


def normalize_arxiv_id(value: object | None) -> str | None:
    text = _text(value)
    if not text:
        return None
    text = unquote(html.unescape(text))
    text = re.sub(r"^arxiv:\s*", "", text, flags=re.I)
    text = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", text, flags=re.I)
    text = re.sub(r"\.pdf$", "", text, flags=re.I)
    text = re.sub(r"v\d+$", "", text.strip().rstrip(_TRAILING), flags=re.I).lower()
    return text if _ARXIV_NEW_RE.fullmatch(text) or _ARXIV_OLD_RE.fullmatch(text) else None


def normalize_openalex_id(value: object | None) -> str | None:
    text = _text(value)
    if not text:
        return None
    match = _OPENALEX_RE.search(unquote(text))
    return match.group(0).upper() if match else None


def normalize_semantic_scholar_id(value: object | None) -> str | None:
    text = _text(value)
    if not text:
        return None
    text = unquote(text).strip().rstrip(_TRAILING)
    if _S2_HEX_RE.fullmatch(text):
        return text.lower()
    match = _CORPUS_RE.fullmatch(text)
    if match:
        return f"CorpusID:{match.group(1)}"
    return text


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
