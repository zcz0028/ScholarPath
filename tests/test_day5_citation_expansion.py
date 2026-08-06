from scholarpath.paper.schema import PaperRecord
from scholarpath.retrieval.citation_expansion import CitationBudget, CitationExpansionConfig, annotate_citation_candidate, deduplicate_citation_candidates


def test_only_one_hop_supported():
    try:
        CitationExpansionConfig(max_hops=2)
    except ValueError:
        pass
    else:
        raise AssertionError("two-hop expansion must be rejected")


def test_budget_enforced():
    budget = CitationBudget(max_api_calls=1)
    budget.record_api_call()
    assert not budget.can_call()
    try:
        budget.record_api_call()
    except RuntimeError:
        pass
    else:
        raise AssertionError("budget must be enforced")


def test_cache_hits_do_not_consume_api_budget():
    budget = CitationBudget(max_api_calls=1)
    budget.record_cache_hit()
    assert budget.actual_api_calls == 0
    assert budget.cache_hits == 1


def test_deduplicate_merges_citation_paths():
    seed1 = PaperRecord(title="Seed1", openalex_id="W1")
    seed2 = PaperRecord(title="Seed2", openalex_id="W2")
    a = annotate_citation_candidate(PaperRecord(title="Target", doi="10.1/x", openalex_id="W9"), seed=seed1, edge_type="reference", seed_rank=1, edge_rank=1, seed_score=0.8)
    b = annotate_citation_candidate(PaperRecord(title="Target", doi="10.1/x", openalex_id="W9"), seed=seed2, edge_type="cited_by", seed_rank=2, edge_rank=2, seed_score=0.7)
    result = deduplicate_citation_candidates([a, b], limit=10)
    assert len(result) == 1
    assert result[0].raw["day5_citation_support_count"] == 2


def test_candidate_limit_enforced():
    papers = [PaperRecord(title=f"P{i}", openalex_id=f"W{i}") for i in range(10)]
    assert len(deduplicate_citation_candidates(papers, limit=3)) == 3
