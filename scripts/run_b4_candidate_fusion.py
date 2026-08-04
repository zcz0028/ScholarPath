from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scholarpath.evaluation.io import load_gold, load_predictions
from scholarpath.evaluation.metrics import (
    EvaluationConfig,
    evaluate_records,
    write_evaluation_outputs,
)
from scholarpath.fusion.candidate_fusion import (
    CandidateSource,
    FusionConfig,
    fuse_candidate_records,
    load_source_predictions,
)
from scholarpath.query.constraints import ConstraintDecomposer
from scholarpath.rerank.semantic_rerank import annotate_and_semantic_rerank_record


DEFAULT_SOURCES = [
    "b0=outputs/b0_openalex",
    "b1_2=outputs/b1_2_raw_anchor_q3_p40",
    "b2=outputs/b2_openalex",
]

SOURCE_WEIGHTS = {
    "b0": 1.05,
    "b1": 1.00,
    "b1_2": 1.00,
    "b2": 0.86,
}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON object in {path}")
                records.append(value)
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_benchmark(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Line {line_no}: expected JSON object.")
            if value.get("qid") is None:
                raise ValueError(f"Line {line_no}: missing qid.")
            records.append(value)
    return records


def parse_source(value: str) -> CandidateSource:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--source must use name=directory format, for example b0=outputs/b0_openalex"
        )
    name, directory = value.split("=", 1)
    name = name.strip()
    directory = directory.strip()
    if not name or not directory:
        raise argparse.ArgumentTypeError("Empty source name or directory.")
    return CandidateSource(
        name=name,
        directory=directory,
        weight=SOURCE_WEIGHTS.get(name, 1.0),
    )


def run_evaluations(
    *,
    gold_path: str | Path,
    prediction_paths: dict[int, Path],
    output_dir: Path,
) -> dict[str, Any]:
    gold_records = load_gold(gold_path)
    summaries: dict[str, Any] = {}

    for k, prediction_path in sorted(prediction_paths.items()):
        predictions = load_predictions(prediction_path)
        for mode in ("strict", "pasa_title"):
            result = evaluate_records(
                gold_records=gold_records,
                prediction_records=predictions,
                config=EvaluationConfig(
                    mode=mode,
                    recall_at=(20, 50, 100),
                    deduplicate=True,
                ),
            )
            eval_dir = output_dir / "evaluation" / f"top{k}" / mode
            write_evaluation_outputs(result, eval_dir)
            summaries[f"top{k}_{mode}"] = result.summary()
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run B4 multi-source candidate fusion and B3 semantic reranking."
    )
    parser.add_argument("--benchmark", default="data/raw/RealScholarQuery/test.jsonl")
    parser.add_argument("--gold", default="data/processed/realscholarquery_gold.jsonl")
    parser.add_argument("--output-dir", default="outputs/b4_fusion_semantic")
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Candidate source in name=directory format. Can be repeated.",
    )
    parser.add_argument("--source-file", default="predictions_top100.jsonl")
    parser.add_argument("--max-fused-candidates", type=int, default=200)
    parser.add_argument(
        "--semantic-mode",
        choices=["conservative", "hybrid", "semantic_first"],
        default="semantic_first",
    )
    parser.add_argument("--max-constraints", type=int, default=6)
    parser.add_argument("--max-subqueries", type=int, default=6)
    parser.add_argument("--top-k", nargs="+", type=int, default=[20, 50, 100])
    return parser


def make_semantic_config_compatible(mode: str):
    # Imported here to keep compatibility if earlier modules are reloaded.
    from scholarpath.rerank.semantic_rerank import make_semantic_config
    return make_semantic_config(mode)


def main() -> int:
    args = build_parser().parse_args()

    top_ks = sorted({int(value) for value in args.top_k if int(value) > 0})
    if not top_ks:
        print("[ERROR] At least one positive --top-k is required.")
        return 2

    source_specs = args.source or DEFAULT_SOURCES
    sources = [parse_source(value) for value in source_specs]
    sources = [
        CandidateSource(
            name=source.name,
            directory=source.directory,
            file_name=args.source_file,
            weight=source.weight,
        )
        for source in sources
    ]

    benchmark_records = read_benchmark(args.benchmark)
    benchmark_by_qid = {str(item.get("qid")): item for item in benchmark_records}

    source_predictions = {
        source.name: load_source_predictions(source)
        for source in sources
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fusion_config = FusionConfig(max_fused_candidates=args.max_fused_candidates)
    semantic_config = make_semantic_config_compatible(args.semantic_mode)
    decomposer = ConstraintDecomposer(
        max_constraints=args.max_constraints,
        max_subqueries=args.max_subqueries,
    )

    fused_before_rerank: list[dict[str, Any]] = []
    reranked_records_top100: list[dict[str, Any]] = []
    fusion_stats: list[dict[str, Any]] = []
    decomposition_records: list[dict[str, Any]] = []

    for benchmark in benchmark_records:
        qid = str(benchmark.get("qid"))
        question = str(benchmark.get("question") or "")

        source_records = [
            (source, source_predictions[source.name].get(qid))
            for source in sources
        ]

        fused_record, stats = fuse_candidate_records(
            qid=qid,
            question=question,
            source_records=source_records,
            config=fusion_config,
        )
        fused_before_rerank.append(fused_record)
        fusion_stats.append(stats)

        decomposition = decomposer.decompose(question)
        decomposition_records.append({"qid": qid, **decomposition.to_dict()})

        reranked = annotate_and_semantic_rerank_record(
            fused_record,
            decomposition,
            semantic_config,
        )
        reranked_records_top100.append(reranked)

    prediction_paths: dict[int, Path] = {}
    for k in top_ks:
        records_k = [
            {
                "qid": record.get("qid"),
                "question": record.get("question"),
                "papers": (record.get("papers") or [])[:k],
            }
            for record in reranked_records_top100
        ]
        path = output_dir / f"predictions_top{k}.jsonl"
        write_jsonl(path, records_k)
        prediction_paths[k] = path

    write_jsonl(output_dir / "fused_candidates_before_rerank.jsonl", fused_before_rerank)
    write_jsonl(output_dir / "raw_predictions_top100.jsonl", reranked_records_top100)
    write_jsonl(output_dir / "fusion_stats.jsonl", fusion_stats)
    write_jsonl(output_dir / "query_decompositions.jsonl", decomposition_records)

    evaluation_summaries = run_evaluations(
        gold_path=args.gold,
        prediction_paths=prediction_paths,
        output_dir=output_dir,
    )

    total_unique = sum(int(item.get("unique_fused_candidates", 0)) for item in fusion_stats)
    total_kept = sum(int(item.get("kept_fused_candidates", 0)) for item in fusion_stats)
    total_multi_source = sum(int(item.get("multi_source_candidates", 0)) for item in fusion_stats)

    summary = {
        "baseline": "B4.0",
        "strategy": "multi_source_candidate_fusion_then_b3_semantic_rerank",
        "sources": [
            {
                "name": source.name,
                "directory": source.directory,
                "file_name": source.file_name,
                "weight": source.weight,
            }
            for source in sources
        ],
        "fusion": {
            "max_fused_candidates": args.max_fused_candidates,
            "total_unique_candidates_before_cap": total_unique,
            "total_kept_candidates": total_kept,
            "total_multi_source_candidates": total_multi_source,
            "mean_kept_per_query": total_kept / len(benchmark_records) if benchmark_records else 0.0,
            "semantic_mode": args.semantic_mode,
        },
        "queries": {
            "total": len(reranked_records_top100),
            "succeeded": len(reranked_records_top100),
            "failed": 0,
        },
        "retrieval": {
            "actual_api_calls": 0,
            "cache_hits": 0,
            "retries": 0,
            "raw_results": 0,
            "response_bytes": 0,
            "note": "B4.0 fuses existing candidate files and does not call OpenAlex.",
        },
        "tokens": {
            "input": 0,
            "output": 0,
            "total": 0,
            "note": "B4.0 uses deterministic fusion and existing B3 semantic reranker.",
        },
        "estimated_cost_usd": 0.0,
        "top_k_outputs": top_ks,
        "prediction_files": {
            str(k): str(path) for k, path in prediction_paths.items()
        },
        "evaluation": evaluation_summaries,
    }

    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    focus20 = evaluation_summaries.get("top20_strict", {})
    focus50 = evaluation_summaries.get("top50_strict", {})
    focus100 = evaluation_summaries.get("top100_strict", {})
    print(f"[OK] Queries: {len(reranked_records_top100)}")
    print(f"[OK] Sources: {', '.join(source.name for source in sources)}")
    print(f"[OK] Semantic mode: {args.semantic_mode}")
    print(
        "[OK] Top20 strict: "
        f"MacroF1={focus20.get('macro', {}).get('f1')} "
        f"TP={focus20.get('counts', {}).get('tp')} "
        f"FP={focus20.get('counts', {}).get('fp')} "
        f"FN={focus20.get('counts', {}).get('fn')}"
    )
    print(
        "[OK] Top50 strict: "
        f"MacroF1={focus50.get('macro', {}).get('f1')} "
        f"TP={focus50.get('counts', {}).get('tp')} "
        f"FP={focus50.get('counts', {}).get('fp')} "
        f"FN={focus50.get('counts', {}).get('fn')}"
    )
    print(
        "[OK] Top100 strict: "
        f"MacroF1={focus100.get('macro', {}).get('f1')} "
        f"TP={focus100.get('counts', {}).get('tp')} "
        f"FP={focus100.get('counts', {}).get('fp')} "
        f"FN={focus100.get('counts', {}).get('fn')}"
    )
    print("[OK] API calls: 0")
    print("[OK] Total tokens: 0")
    print(f"[OUTPUT] Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
