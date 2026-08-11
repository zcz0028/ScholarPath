from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scholarpath.query.constraints import ConstraintDecomposer
from scholarpath.rerank.constraint_evidence import (
    build_canonical_constraints,
    build_constraint_evidence,
)


CASES = [
    {
        "name": "generic_llm_survey",
        "query": (
            "recent papers on multimodal large language models "
            "for scientific document understanding"
        ),
        "paper": {
            "title": "A Survey of Large Language Models",
            "abstract": "",
            "raw": {"concepts": []},
        },
    },
    {
        "name": "multimodal_llm",
        "query": (
            "recent papers on multimodal large language models "
            "for scientific document understanding"
        ),
        "paper": {
            "title": "MM-LLMs: Recent Advances in MultiModal Large Language Models",
            "abstract": "",
            "raw": {"concepts": []},
        },
    },
    {
        "name": "gnn_molecular_property",
        "query": "graph neural networks for molecular property prediction",
        "paper": {
            "title": (
                "A compact review of molecular property prediction "
                "with graph neural networks"
            ),
            "abstract": "",
            "raw": {"concepts": []},
        },
    },
]


def main() -> None:
    output = []
    decomposer = ConstraintDecomposer()

    for case in CASES:
        decomposition = decomposer.decompose(case["query"])
        canonical = build_canonical_constraints(
            question=case["query"],
            constraints=decomposition.constraints,
        )
        evidence = build_constraint_evidence(canonical, case["paper"])
        output.append(
            {
                "case": case["name"],
                "query": case["query"],
                "paper_title": case["paper"]["title"],
                "canonical_constraints": [item.to_dict() for item in canonical],
                "constraint_evidence": [item.to_dict() for item in evidence],
                "matched_count": sum(item.matched for item in evidence),
                "constraint_count": len(evidence),
            }
        )

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
