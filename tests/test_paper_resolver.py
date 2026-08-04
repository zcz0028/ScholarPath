from scholarpath.paper.resolver import PaperResolver, ResolutionConfig
from scholarpath.paper.schema import PaperRecord


def test_merge_by_doi() -> None:
    records = [
        PaperRecord(title="A Paper", doi="https://doi.org/10.1000/Test", source="a"),
        PaperRecord(title="A Paper Extended Metadata", doi="10.1000/test", source="b"),
    ]
    result = PaperResolver().resolve(records)
    assert len(result.entities) == 1
    assert result.exact_id_merges == 1


def test_merge_arxiv_and_formal_version_by_title_author_year() -> None:
    records = [
        PaperRecord(
            title="A Reliable Retrieval Method",
            authors=["Alice Smith"],
            year=2024,
            arxiv_id="2401.01234v2",
            source="arxiv",
        ),
        PaperRecord(
            title="A Reliable Retrieval Method",
            authors=["Alice Smith", "Bob Lee"],
            year=2025,
            doi="10.1000/reliable",
            source="crossref",
        ),
    ]
    result = PaperResolver().resolve(records)
    assert len(result.entities) == 1
    assert result.title_merges == 1
    assert result.entities[0].record.arxiv_id == "2401.01234"
    assert result.entities[0].record.doi == "10.1000/reliable"


def test_conflicting_doi_does_not_merge() -> None:
    records = [
        PaperRecord(
            title="Identical Title for Different Records",
            authors=["Alice Smith"],
            year=2024,
            doi="10.1000/one",
        ),
        PaperRecord(
            title="Identical Title for Different Records",
            authors=["Alice Smith"],
            year=2024,
            doi="10.1000/two",
        ),
    ]
    assert len(PaperResolver().resolve(records).entities) == 2


def test_same_title_without_supporting_metadata_does_not_merge() -> None:
    records = [
        PaperRecord(title="A Long Enough Shared Paper Title"),
        PaperRecord(title="A Long Enough Shared Paper Title"),
    ]
    assert len(PaperResolver().resolve(records).entities) == 2


def test_near_title_goes_to_review_not_auto_merge() -> None:
    records = [
        PaperRecord(title="Efficient Scholarly Retrieval Agents", authors=["Carol Wang"], year=2024),
        PaperRecord(title="Efficient Scholarly Retrieval Agent", authors=["Carol Wang"], year=2024),
    ]
    result = PaperResolver(ResolutionConfig(fuzzy_review_threshold=0.85)).resolve(records)
    assert len(result.entities) == 2
    assert len(result.review_candidates) == 1
