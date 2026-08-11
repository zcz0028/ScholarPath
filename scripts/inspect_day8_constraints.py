from __future__ import annotations

import argparse
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
    deduplicate_constraints,
    recover_residual_constraint_spans,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect Day8-1A V2 normalized, deduplicated and recovered constraints."
    )
    parser.add_argument("query", help="Research query to inspect.")
    args = parser.parse_args()

    decomposition = ConstraintDecomposer().decompose(args.query)
    deduplicated = deduplicate_constraints(decomposition.constraints)
    residual_spans = recover_residual_constraint_spans(
        question=args.query,
        canonical_constraints=deduplicated,
    )
    canonical = build_canonical_constraints(
        question=args.query,
        constraints=decomposition.constraints,
    )

    payload = {
        "query": args.query,
        "raw_constraints": [item.to_dict() for item in decomposition.constraints],
        "deduplicated_constraints": [item.to_dict() for item in deduplicated],
        "residual_spans": residual_spans,
        "canonical_constraints": [item.to_dict() for item in canonical],
        "raw_constraint_count": len(decomposition.constraints),
        "deduplicated_constraint_count": len(deduplicated),
        "canonical_constraint_count": len(canonical),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
