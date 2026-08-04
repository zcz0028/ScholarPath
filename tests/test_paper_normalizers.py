from scholarpath.paper.normalizers import (
    compact_title_key,
    normalize_arxiv_id,
    normalize_author_name,
    normalize_doi,
    normalize_openalex_id,
    normalize_semantic_scholar_id,
    normalize_title,
)


def test_normalize_doi() -> None:
    assert normalize_doi("https://doi.org/10.1000/ABC.1") == "10.1000/abc.1"
    assert normalize_doi("doi: 10.1000/ABC.1.") == "10.1000/abc.1"


def test_normalize_arxiv_id() -> None:
    assert normalize_arxiv_id("arXiv:2401.01234v3") == "2401.01234"
    assert normalize_arxiv_id("https://arxiv.org/pdf/cs/9901001v2.pdf") == "cs/9901001"


def test_normalize_openalex_and_s2() -> None:
    assert normalize_openalex_id("https://openalex.org/w123") == "W123"
    assert normalize_semantic_scholar_id("A" * 40) == "a" * 40
    assert normalize_semantic_scholar_id("CorpusId:123") == "CorpusID:123"


def test_title_and_author_normalization() -> None:
    assert normalize_title("GPT-4: A Study") == "gpt 4 a study"
    assert compact_title_key("GPT-4: A Study") == "gpt4astudy"
    assert normalize_author_name("Smith, Alice") == "smith alice"
