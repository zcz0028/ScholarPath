from scholarpath.ranking.seed_gate import SeedGateConfig, evaluate_seed, select_seed_papers


def paper(i: int, **raw):
    return {"title": f"Paper {i}", "openalex_id": f"W{i}", "raw": {"day4_rescue_score": 0.7, **raw}}


def test_seed_gate_limits_seed_count():
    selected, _ = select_seed_papers(qid="q", papers=[paper(i) for i in range(1, 8)], config=SeedGateConfig(max_seeds_per_query=3))
    assert len(selected) == 3


def test_seed_gate_rejects_missing_openalex_id():
    value = {"title": "No id", "raw": {"day4_rescue_score": 0.8}}
    result = evaluate_seed(qid="q", paper_mapping=value, rank=1, config=SeedGateConfig())
    assert not result.accepted
    assert "missing_openalex_id" in result.rejection_reasons


def test_seed_gate_rejects_guard_reject():
    result = evaluate_seed(qid="q", paper_mapping=paper(1, b5_1_guard_decision="reject"), rank=1, config=SeedGateConfig())
    assert not result.accepted


def test_seed_gate_rejects_rank_outside_gate():
    result = evaluate_seed(qid="q", paper_mapping=paper(1), rank=16, config=SeedGateConfig(max_seed_rank=15))
    assert not result.accepted


def test_seed_gate_rejects_gold_fields():
    value = paper(1)
    value["gold_titles"] = ["x"]
    try:
        evaluate_seed(qid="q", paper_mapping=value, rank=1, config=SeedGateConfig())
    except ValueError:
        pass
    else:
        raise AssertionError("gold field must be rejected")


def test_multi_plan_support_increases_score():
    a = evaluate_seed(qid="q", paper_mapping=paper(1, day4_occurrence_count=1), rank=1, config=SeedGateConfig())
    b = evaluate_seed(qid="q", paper_mapping=paper(2, day4_occurrence_count=2), rank=1, config=SeedGateConfig())
    assert b.seed_score > a.seed_score
