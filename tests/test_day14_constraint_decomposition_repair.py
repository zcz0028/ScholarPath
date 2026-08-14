from __future__ import annotations

from scholarpath.query.constraints import ConstraintDecomposer
from scholarpath.rerank.constraint_evidence import build_canonical_constraints


def texts(question: str) -> list[str]:
    return [
        item.text.casefold()
        for item in ConstraintDecomposer(max_constraints=8).decompose(question).constraints
    ]


def test_complex_llm_claim_preserves_complete_semantic_relations() -> None:
    question = (
        "Find papers supporting the claim knowledgeable LLMs have sufficient "
        "inductive capacity to analyze relationships between multiple papers "
        "and systematically write a survey."
    )
    items = texts(question)

    forbidden = {
        "supporting the claim knowledgeable",
        "llms have sufficient inductive",
        "relationships between multiple",
        "systematically write",
    }
    assert forbidden.isdisjoint(items)

    assert any("large language model" in item or item == "llm" for item in items)
    assert any("inductive capacity" in item for item in items)
    assert any(
        "relationships between multiple papers" in item
        for item in items
    )
    assert any(
        "systematically write a survey" in item
        for item in items
    )


def test_short_academic_terms_survive_boilerplate_cleanup() -> None:
    qat = texts("Could you list papers about QAT for LLMs?")
    assert any(item == "qat" for item in qat)
    assert any("llm" in item or "large language model" in item for item in qat)

    rlhf = texts("I want to know how RLHF changes large language model alignment.")
    assert any(item == "rlhf" for item in rlhf)


def test_prompt_boilerplate_is_not_promoted_to_constraints() -> None:
    items = texts(
        "Can you help me find papers which explain why graph neural networks "
        "work for molecular property prediction?"
    )
    joined = " | ".join(items)
    assert "can you" not in joined
    assert "help me" not in joined
    assert "find papers" not in joined
    assert "explaining why" not in joined


def test_dangling_relation_fragments_are_rejected() -> None:
    items = texts(
        "Papers on language models that analyze relationships between multiple "
        "documents and automatically generate a review."
    )
    assert "relationships between multiple" not in items
    assert all(not item.endswith(" between") for item in items)


def test_canonical_residual_recovery_does_not_reintroduce_prompt_fragments() -> None:
    question = (
        "Find papers supporting the claim knowledgeable LLMs have sufficient "
        "inductive capacity to analyze relationships between multiple papers "
        "and systematically write a survey."
    )
    decomposition = ConstraintDecomposer(max_constraints=8).decompose(question)
    canonical = build_canonical_constraints(
        question=question,
        constraints=decomposition.constraints,
    )
    canonical_texts = {item.canonical_text for item in canonical}

    assert "supporting claim knowledgeable" not in canonical_texts
    assert "relationships between multiple" not in canonical_texts
    assert "llm have sufficient inductive" not in canonical_texts


def test_prompt_residue_is_removed_precisely() -> None:
    samples = {
        "Is there any work that analyzes the scaling law of multi-module models?": {"is there"},
        "I am looking for research papers on multimodal foundation models.": {"am looking for research"},
        "How can LLM agents be evaluated for financial tasks? Note that I am referring to agents.": {"i am referring"},
    }
    for question, forbidden in samples.items():
        items = texts(question)
        assert forbidden.isdisjoint(items)


def test_legitimate_short_constraints_are_not_deleted() -> None:
    samples = [
        ("Could you list research about QAT for low-bit weights?", "qat"),
        ("Research using RLHF for hallucination reduction.", "rlhf"),
        ("Papers on mining factors in stock exchange analysis.", "mining factor"),
        ("Research on LLM-generated text detection.", "generated text"),
    ]
    for question, expected in samples:
        items = texts(question)
        assert any(expected in item for item in items)


def test_multimodal_day8_contract_remains_stable() -> None:
    question = (
        "recent papers on multimodal large language models "
        "for scientific document understanding"
    )
    from scholarpath.rerank.constraint_evidence import build_canonical_constraints
    decomposition = ConstraintDecomposer().decompose(question)
    canonical = build_canonical_constraints(question=question, constraints=decomposition.constraints)
    assert [item.canonical_text for item in canonical] == [
        "large language model",
        "multimodal",
        "scientific document understanding",
    ]
