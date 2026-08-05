from scholarpath.evaluation.pipeline_diagnostics import (
    StageCoverage,
    classify_failure,
    diagnose_pipeline,
)
from scholarpath.paper.schema import PaperRecord


def coverage(name: str, hits: int) -> StageCoverage:
    return StageCoverage(
        stage_name=name,
        candidate_count=100,
        matched_gold_count=hits,
        matched_gold_ids=["doi:10.1/example"] if hits else [],
        matched_gold_titles=["Example"] if hits else [],
        matched_ranks=[5] if hits else [],
        best_matched_rank=5 if hits else None,
        match_types=["strong_identifier"] if hits else [],
    )


def test_classifies_retrieval_miss():
    stages = {
        "b0_top100": coverage("b0_top100", 0),
        "b1_2_top100": coverage("b1_2_top100", 0),
        "b2_top100": coverage("b2_top100", 0),
        "b4_pool": coverage("b4_pool", 0),
        "b4_top100": coverage("b4_top100", 0),
    }
    assert classify_failure(stages, target_stage="b4_top100") == (
        "retrieval_miss",
        "b2_top100",
    )


def test_classifies_fusion_pool_miss():
    stages = {
        "b0_top100": coverage("b0_top100", 1),
        "b1_2_top100": coverage("b1_2_top100", 0),
        "b2_top100": coverage("b2_top100", 0),
        "b4_pool": coverage("b4_pool", 0),
        "b4_top100": coverage("b4_top100", 0),
    }
    assert classify_failure(stages, target_stage="b4_top100") == (
        "fusion_pool_miss",
        "b4_pool",
    )


def test_classifies_ranking_loss_top100():
    stages = {
        "b0_top100": coverage("b0_top100", 0),
        "b1_2_top100": coverage("b1_2_top100", 0),
        "b2_top100": coverage("b2_top100", 0),
        "b4_pool": coverage("b4_pool", 1),
        "b4_top100": coverage("b4_top100", 0),
    }
    assert classify_failure(stages, target_stage="b4_top100") == (
        "ranking_loss_top100",
        "b4_top100",
    )


def test_classifies_success():
    stages = {
        "b0_top100": coverage("b0_top100", 0),
        "b1_2_top100": coverage("b1_2_top100", 0),
        "b2_top100": coverage("b2_top100", 0),
        "b4_pool": coverage("b4_pool", 1),
        "b4_top100": coverage("b4_top100", 1),
    }
    assert classify_failure(stages, target_stage="b4_top100") == ("success", None)


def test_possible_matching_issue_has_priority_over_retrieval_miss():
    stages = {
        "b0_top100": coverage("b0_top100", 0),
        "b1_2_top100": coverage("b1_2_top100", 0),
        "b2_top100": coverage("b2_top100", 0),
        "b4_pool": coverage("b4_pool", 0),
        "b4_top100": coverage("b4_top100", 0),
    }
    assert classify_failure(
        stages,
        target_stage="b4_top100",
        possible_matching_issue_count=1,
    ) == ("possible_matching_issue", "b4_top100")


def test_diagnose_pipeline_uses_array_order_as_rank():
    gold_paper = PaperRecord(title="A Reliable Paper", doi="10.1000/example")
    wrong = PaperRecord(title="Wrong Paper", doi="10.1000/wrong")
    correct = PaperRecord(title="A Reliable Paper", doi="10.1000/example")
    gold = {
        "q1": {"qid": "q1", "question": "Find A Reliable Paper", "papers": [gold_paper]}
    }
    stages = {
        "b0_top100": {"q1": {"qid": "q1", "question": "", "papers": []}},
        "b1_2_top100": {"q1": {"qid": "q1", "question": "", "papers": []}},
        "b2_top100": {"q1": {"qid": "q1", "question": "", "papers": []}},
        "b4_pool": {"q1": {"qid": "q1", "question": "", "papers": [wrong, correct]}},
        "b4_top100": {"q1": {"qid": "q1", "question": "", "papers": [wrong, correct]}},
    }
    diagnoses, summary, _ = diagnose_pipeline(
        gold,
        stages,
        target_stage="b4_top100",
        expected_query_count=1,
    )
    assert diagnoses[0].stages["b4_top100"].matched_ranks == [2]
    assert diagnoses[0].stages["b4_top100"].best_matched_rank == 2
    assert summary.target_total_tp == 1
    assert summary.target_zero_recall_count == 0


def test_diagnose_pipeline_rejects_qid_mismatch():
    paper = PaperRecord(title="Paper")
    gold = {"q1": {"qid": "q1", "question": "", "papers": [paper]}}
    stages = {
        "b4_top100": {"q2": {"qid": "q2", "question": "", "papers": [paper]}}
    }
    try:
        diagnose_pipeline(
            gold,
            stages,
            target_stage="b4_top100",
            expected_query_count=1,
        )
    except ValueError as error:
        assert "qids do not match gold" in str(error)
    else:
        raise AssertionError("Expected qid mismatch to raise ValueError")
