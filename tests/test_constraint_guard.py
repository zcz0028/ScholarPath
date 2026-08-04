from scholarpath.guard.constraint_guard import (
    evaluate_paper_with_guard,
    extract_guard_constraints,
    make_guard_config,
)


def test_extract_exclude_survey_constraint() -> None:
    constraints = extract_guard_constraints(
        "Please find multimodal foundation model papers. Please exclude survey papers."
    )
    ids = {item.constraint_id for item in constraints}
    assert "exclude_survey_or_review" in ids


def test_survey_violation_is_rejected() -> None:
    constraints = extract_guard_constraints(
        "Find multimodal foundation model papers and exclude survey papers."
    )
    paper = {
        "title": "A Comprehensive Survey of Multimodal Foundation Models",
        "abstract": "This paper surveys multimodal foundation models.",
    }
    result = evaluate_paper_with_guard(
        paper,
        constraints,
        make_guard_config("balanced"),
    )
    assert result.decision == "reject"
    assert result.violation_tags


def test_hotpotqa_dataset_missing_downranks() -> None:
    constraints = extract_guard_constraints(
        "Papers that evaluate large language models on the HotPotQA dataset."
    )
    paper = {
        "title": "Adaptive Retrieval-Augmented Large Language Models",
        "abstract": "We evaluate on open-domain question answering datasets.",
    }
    result = evaluate_paper_with_guard(
        paper,
        constraints,
        make_guard_config("precision"),
    )
    assert result.guard_score < 1.0
    assert result.missing_constraints


def test_hotpotqa_dataset_satisfied() -> None:
    constraints = extract_guard_constraints(
        "Papers that evaluate large language models on the HotPotQA dataset."
    )
    paper = {
        "title": "Reasoning with Large Language Models on HotPotQA",
        "abstract": "Experiments are conducted on HotPotQA.",
    }
    result = evaluate_paper_with_guard(
        paper,
        constraints,
        make_guard_config("precision"),
    )
    assert result.guard_score > 0.5
    assert result.decision in {"pass", "soft_pass"}


def test_visual_audio_constraint_requires_both_modalities() -> None:
    constraints = extract_guard_constraints(
        "papers about multimodal foundation models that support both visual and audio inputs"
    )
    paper = {
        "title": "Visual Foundation Models for Images",
        "abstract": "We study image and video inputs.",
    }
    result = evaluate_paper_with_guard(
        paper,
        constraints,
        make_guard_config("precision"),
    )
    assert result.guard_score < 1.0
    assert result.missing_constraints


def test_smaller_bigger_dataset_relation_is_detected() -> None:
    constraints = extract_guard_constraints(
        "Give me papers which show that using a smaller dataset in large language model pre-training can result in better models than using bigger datasets."
    )
    ids = {item.constraint_id for item in constraints}
    assert "relation_smaller_dataset_better" in ids


def test_video_generation_constraint_is_detected_with_generate_videos() -> None:
    constraints = extract_guard_constraints(
        "List all papers that use autoregressive transformer to generate videos."
    )
    ids = {item.constraint_id for item in constraints}
    assert "method_autoregressive_transformer" in ids
    assert "task_video_generation" in ids


def test_image_video_description_constraint_is_detected() -> None:
    constraints = extract_guard_constraints(
        "Papers that apply RLHF to address the hallucination problem in image and video description."
    )
    ids = {item.constraint_id for item in constraints}
    assert "method_RLHF" in ids
    assert "task_hallucination" in ids
    assert "task_image_video_description" in ids


def test_rank_search_results_by_llm_constraint_is_detected() -> None:
    constraints = extract_guard_constraints(
        "Give me papers about how to rank search results by the use of LLM."
    )
    ids = {item.constraint_id for item in constraints}
    assert "method_ranking_with_llm" in ids


def test_missing_medium_constraint_is_soft_pass_not_full_pass() -> None:
    constraints = extract_guard_constraints(
        "multimodal foundation models that support both visual and audio inputs and exclude survey papers"
    )
    paper = {
        "title": "Mapping visual distance to audio frequency",
        "abstract": "We map visual distance to audio feedback. This is not a survey.",
    }
    result = evaluate_paper_with_guard(
        paper,
        constraints,
        make_guard_config("balanced"),
    )
    assert result.decision != "pass"
