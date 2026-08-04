from scholarpath.evaluation.metrics import EvaluationConfig, evaluate_records
from scholarpath.paper.schema import PaperRecord


def paper(title: str, arxiv_id: str | None = None) -> PaperRecord:
    return PaperRecord(title=title, arxiv_id=arxiv_id)


def test_macro_and_micro_metrics() -> None:
    gold = {
        "q1": {"question": "q1", "papers": [paper("A"), paper("B")]},
        "q2": {"question": "q2", "papers": [paper("C")]},
    }
    predictions = {
        "q1": {"papers": [paper("A"), paper("X")]},
        "q2": {"papers": [paper("C")]},
    }
    result = evaluate_records(gold, predictions, EvaluationConfig(recall_at=(1, 2)))
    summary = result.summary()

    assert summary["counts"]["tp"] == 2
    assert summary["counts"]["fp"] == 1
    assert summary["counts"]["fn"] == 1
    assert abs(summary["macro"]["precision"] - 0.75) < 1e-9
    assert abs(summary["macro"]["recall"] - 0.75) < 1e-9
    assert abs(summary["macro"]["f1"] - 0.75) < 1e-9
    assert abs(summary["micro"]["f1"] - (2 / 3)) < 1e-9


def test_missing_prediction_qid_scores_as_empty() -> None:
    gold = {
        "q1": {"question": "q1", "papers": [paper("A")]},
        "q2": {"question": "q2", "papers": [paper("B")]},
    }
    predictions = {"q1": {"papers": [paper("A")]}}
    result = evaluate_records(gold, predictions)
    assert result.missing_prediction_qids == ["q2"]
    q2 = next(item for item in result.per_query if item.qid == "q2")
    assert q2.tp == 0
    assert q2.fn == 1
    assert q2.f1 == 0.0


def test_recall_at_respects_rank() -> None:
    gold = {
        "q": {
            "question": "q",
            "papers": [paper("A"), paper("B")],
        }
    }
    predictions = {
        "q": {
            "papers": [paper("A"), paper("X"), paper("B")],
        }
    }
    result = evaluate_records(
        gold,
        predictions,
        EvaluationConfig(recall_at=(1, 2, 3)),
    )
    metrics = result.per_query[0].recall_at
    assert metrics["recall@1"] == 0.5
    assert metrics["recall@2"] == 0.5
    assert metrics["recall@3"] == 1.0


def test_duplicates_do_not_inflate_false_positives() -> None:
    gold = {"q": {"question": "q", "papers": [paper("A", "2401.1")]}}
    predictions = {
        "q": {
            "papers": [
                paper("A", "2401.00001"),
                paper("A", "2401.00001v2"),
            ]
        }
    }
    # Use valid five-digit arXiv IDs so normalization succeeds.
    gold["q"]["papers"] = [paper("A", "2401.00001")]
    result = evaluate_records(gold, predictions)
    query = result.per_query[0]
    assert query.prediction_count == 1
    assert query.duplicates_removed == 1
    assert query.tp == 1
    assert query.fp == 0
