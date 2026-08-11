from __future__ import annotations

import json

from scholarpath.query.constraints import QueryConstraint
from scholarpath.rerank.constraint_evidence import (
    build_canonical_constraints,
    build_constraint_evidence,
)
from scholarpath.rerank.recommendation_reason import build_recommendation_reason


def show(case: str, query: str, constraints: list[QueryConstraint], paper: dict) -> dict:
    canonical = build_canonical_constraints(
    question=query,
    constraints=constraints,
)
    evidence = build_constraint_evidence(canonical, paper)
    reason = build_recommendation_reason(constraints=canonical, evidence=evidence)
    return {
        "case": case,
        "query": query,
        "paper_title": paper.get("title"),
        "canonical_constraints": [x.to_dict() for x in canonical],
        "constraint_evidence": [x.to_dict() for x in evidence],
        "recommendation_reason": reason.to_dict(),
    }


def main() -> None:
    q1 = "recent papers on multimodal large language models for scientific document understanding"
    c1 = [
        QueryConstraint("c1_large_language_model", "large language model", "model_or_entity", ["large", "language", "model"], 1.1),
        QueryConstraint("c2_large_language_models", "large language models", "model_or_entity", ["large", "language", "models"], 1.1),
        QueryConstraint("c3_multimodal", "multimodal", "task_or_modality", ["multimodal"], 1.15),
    ]

    q2 = "graph neural networks for molecular property prediction"
    c2 = [
        QueryConstraint("c1_graph_neural_network", "graph neural network", "method_or_property", ["graph", "neural", "network"], 1.25),
        QueryConstraint("c2_graph_neural_networks_for", "graph neural networks for", "topic", ["graph", "neural", "networks", "for"], 1.0),
        QueryConstraint("c3_molecular_property_prediction", "molecular property prediction", "topic", ["molecular", "property", "prediction"], 1.0),
    ]

    cases = [
        show(
            "generic_llm_survey",
            q1,
            c1,
            {"title": "A Survey of Large Language Models", "abstract": "", "raw": {"concepts": []}},
        ),
        show(
            "multimodal_llm",
            q1,
            c1,
            {"title": "MM-LLMs: Recent Advances in MultiModal Large Language Models", "abstract": "", "raw": {"concepts": []}},
        ),
        show(
            "gnn_molecular_property",
            q2,
            c2,
            {"title": "A compact review of molecular property prediction with graph neural networks", "abstract": "", "raw": {"concepts": []}},
        ),
    ]
    print(json.dumps(cases, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
