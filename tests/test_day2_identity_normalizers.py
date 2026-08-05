from scholarpath.paper.normalizers import (
    arxiv_id_from_doi,
    extract_arxiv_id,
    extract_doi,
    extract_openalex_id,
    normalize_url,
)


def test_extract_doi_from_url_and_text() -> None:
    assert extract_doi("See https://doi.org/10.1145/ABC.123?x=1") == "10.1145/abc.123"
    assert extract_doi("doi:10.1000/XYZ.") == "10.1000/xyz"


def test_arxiv_id_from_arxiv_doi() -> None:
    assert arxiv_id_from_doi("https://doi.org/10.48550/arXiv.2401.01234v3") == "2401.01234"


def test_extract_arxiv_from_common_urls() -> None:
    assert extract_arxiv_id("https://export.arxiv.org/abs/2401.01234v2") == "2401.01234"
    assert extract_arxiv_id("https://ar5iv.labs.arxiv.org/html/cs/9901001") == "cs/9901001"


def test_extract_openalex_id() -> None:
    assert extract_openalex_id("https://api.openalex.org/works/W123") == "W123"


def test_normalize_common_urls() -> None:
    assert normalize_url("https://dx.doi.org/10.1000/ABC") == "https://doi.org/10.1000/abc"
    assert normalize_url("https://arxiv.org/pdf/2401.01234v2.pdf") == "https://arxiv.org/abs/2401.01234"
