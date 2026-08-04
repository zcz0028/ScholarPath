from scholarpath.evaluation.matching import (
    deduplicate_predictions,
    match_papers,
    pasa_title_key,
)
from scholarpath.paper.schema import PaperRecord


def test_strict_match_ignores_arxiv_version() -> None:
    predictions = [PaperRecord(title="A", arxiv_id="2401.01234v3")]
    gold = [PaperRecord(title="Different", arxiv_id="2401.01234")]
    result = match_papers(predictions, gold, mode="strict")
    assert result.tp == 1
    assert result.pairs[0].match_type == "strong_identifier"


def test_strict_title_fallback() -> None:
    predictions = [PaperRecord(title="GPT-4: A Study")]
    gold = [PaperRecord(title="GPT 4 A Study")]
    result = match_papers(predictions, gold, mode="strict")
    assert result.tp == 1
    assert result.pairs[0].match_type == "normalized_title"


def test_conflicting_arxiv_blocks_title_match() -> None:
    predictions = [PaperRecord(title="Same Paper", arxiv_id="2401.00001")]
    gold = [PaperRecord(title="Same Paper", arxiv_id="2401.00002")]
    result = match_papers(predictions, gold, mode="strict")
    assert result.tp == 0


def test_prediction_deduplication_retains_rank_order() -> None:
    predictions = [
        PaperRecord(title="Paper One", arxiv_id="2401.01234"),
        PaperRecord(title="Paper One", arxiv_id="2401.01234v2"),
        PaperRecord(title="Paper Two"),
    ]
    kept, removed = deduplicate_predictions(predictions)
    assert len(kept) == 2
    assert kept[0].arxiv_id == "2401.01234"
    assert len(removed) == 1


def test_pasa_title_key_removes_numbers_and_punctuation() -> None:
    assert pasa_title_key("GPT-4: A Study") == "gptastudy"
    predictions = [PaperRecord(title="GPT-4: A Study")]
    gold = [PaperRecord(title="GPT A Study")]
    result = match_papers(predictions, gold, mode="pasa_title")
    assert result.tp == 1
