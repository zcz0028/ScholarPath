from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .normalizers import (
    arxiv_id_from_doi,
    extract_arxiv_id,
    extract_doi,
    extract_openalex_id,
    normalize_arxiv_id,
    normalize_doi,
    normalize_openalex_id,
    normalize_semantic_scholar_id,
    normalize_title,
    title_fingerprint,
)


def _first(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _year(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        year = int(str(value)[:4])
    except (TypeError, ValueError):
        return None
    return year if 1500 <= year <= 2200 else None


def _authors(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        name = ""
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, Mapping):
            nested = item.get("author")
            source = nested if isinstance(nested, Mapping) else item
            name = str(_first(source, ("display_name", "name", "authorName")) or "").strip()
        if name:
            result.append(name)
    return result


@dataclass(slots=True)
class PaperRecord:
    title: str
    source: str = "unknown"
    source_record_id: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    publication_date: str | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    openalex_id: str | None = None
    semantic_scholar_id: str | None = None
    url: str | None = None
    abstract: str | None = None
    citation_count: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.title = str(self.title or "").strip()
        self.source = str(self.source or "unknown").strip() or "unknown"
        self.authors = [str(x).strip() for x in self.authors if str(x).strip()]
        self.year = _year(self.year)
        self.doi = normalize_doi(self.doi)
        self.arxiv_id = normalize_arxiv_id(self.arxiv_id)
        self.openalex_id = normalize_openalex_id(self.openalex_id)
        self.semantic_scholar_id = normalize_semantic_scholar_id(self.semantic_scholar_id)
        if self.citation_count is not None:
            try:
                self.citation_count = int(self.citation_count)
            except (TypeError, ValueError):
                self.citation_count = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], default_source: str = "unknown") -> "PaperRecord":
        ext = data.get("externalIds") or data.get("external_ids") or {}
        if not isinstance(ext, Mapping):
            ext = {}
        source = str(data.get("source") or default_source)
        direct_id = data.get("id")
        openalex_from_id = normalize_openalex_id(direct_id)
        venue_value = _first(data, ("venue", "host_venue", "journal"))
        if isinstance(venue_value, Mapping):
            venue = str(_first(venue_value, ("display_name", "name")) or "").strip() or None
        else:
            venue = str(venue_value).strip() if venue_value else None
        s2_id = _first(data, ("semantic_scholar_id", "semanticScholarId", "paperId"))
        if s2_id is None:
            s2_id = _first(ext, ("CorpusId", "CorpusID", "S2PaperId"))
        source_id = _first(data, ("source_record_id", "sourceRecordId"))
        if source_id is None:
            source_id = data.get("paperId") or direct_id
        return cls(
            title=str(_first(data, ("title", "display_name", "paperTitle", "name")) or ""),
            source=source,
            source_record_id=str(source_id) if source_id else None,
            authors=_authors(_first(data, ("authors", "authorships"))),
            year=_year(_first(data, ("year", "publication_year", "publicationYear"))),
            publication_date=str(_first(data, ("publication_date", "publicationDate")) or "") or None,
            venue=venue,
            doi=_first(data, ("doi", "DOI")) or _first(ext, ("DOI", "doi")),
            arxiv_id=_first(data, ("arxiv_id", "arxivId", "arxiv")) or _first(ext, ("ArXiv", "arXiv", "arxiv")),
            openalex_id=_first(data, ("openalex_id", "openalexId")) or openalex_from_id,
            semantic_scholar_id=s2_id,
            url=str(_first(data, ("url", "landing_page_url", "pdf_url")) or "") or None,
            abstract=str(_first(data, ("abstract", "abstractText")) or "") or None,
            citation_count=_first(data, ("citation_count", "cited_by_count", "citationCount")),
            raw=dict(data),
        )

    @property
    def normalized_title(self) -> str:
        return normalize_title(self.title)

    @property
    def canonical_id(self) -> str:
        if self.doi:
            return f"doi:{self.doi}"
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        if self.openalex_id:
            return f"openalex:{self.openalex_id}"
        if self.semantic_scholar_id:
            return f"s2:{self.semantic_scholar_id}"
        return f"title:{title_fingerprint(self.title)}"

    def identity_keys(self) -> set[str]:
        result: set[str] = set()
        for prefix, value in (
            ("doi", self.doi),
            ("arxiv", self.arxiv_id),
            ("openalex", self.openalex_id),
            ("s2", self.semantic_scholar_id),
        ):
            if value:
                result.add(f"{prefix}:{value}")
        return result

    def inferred_identifiers(self) -> dict[str, str]:
        """Infer identifiers from DOI, URL, and source record IDs.

        These inferred values are used only by the Day-2 identity-aware
        matching path. Existing ``strict`` evaluation remains unchanged.
        """
        values: dict[str, str] = {}
        doi = self.doi or extract_doi(self.url) or extract_doi(self.source_record_id)
        arxiv_id = (
            self.arxiv_id
            or arxiv_id_from_doi(doi)
            or extract_arxiv_id(self.url)
            or extract_arxiv_id(self.source_record_id)
        )
        openalex_id = (
            self.openalex_id
            or extract_openalex_id(self.url)
            or extract_openalex_id(self.source_record_id)
        )
        if doi:
            values["doi"] = doi
        if arxiv_id:
            values["arxiv_id"] = arxiv_id
        if openalex_id:
            values["openalex_id"] = openalex_id
        if self.semantic_scholar_id:
            values["semantic_scholar_id"] = self.semantic_scholar_id
        return values

    def identity_keys_v2(self) -> set[str]:
        inferred = self.inferred_identifiers()
        result: set[str] = set()
        for prefix, field_name in (
            ("doi", "doi"),
            ("arxiv", "arxiv_id"),
            ("openalex", "openalex_id"),
            ("s2", "semantic_scholar_id"),
        ):
            value = inferred.get(field_name)
            if value:
                result.add(f"{prefix}:{value}")
        return result

    @property
    def canonical_id_v2(self) -> str:
        inferred = self.inferred_identifiers()
        arxiv_id = inferred.get("arxiv_id")
        doi = inferred.get("doi")
        if arxiv_id and (not doi or arxiv_id_from_doi(doi)):
            return f"arxiv:{arxiv_id}"
        if doi:
            return f"doi:{doi}"
        if arxiv_id:
            return f"arxiv:{arxiv_id}"
        if inferred.get("openalex_id"):
            return f"openalex:{inferred['openalex_id']}"
        if inferred.get("semantic_scholar_id"):
            return f"s2:{inferred['semantic_scholar_id']}"
        return f"title:{title_fingerprint(self.title)}"

    def to_dict(self, include_raw: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        if not include_raw:
            payload.pop("raw", None)
        payload["normalized_title"] = self.normalized_title
        payload["canonical_id"] = self.canonical_id
        payload["inferred_identifiers"] = self.inferred_identifiers()
        payload["canonical_id_v2"] = self.canonical_id_v2
        return payload
