import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.eval_utils import (
    CROSS_LINGUAL_QUERIES,
    build_runtime,
    dataset_statistics,
    expand_methods,
    format_dataset_statistics,
    load_json,
    method_label,
    search_with_method,
    write_csv_report,
    write_json_report,
)


def evaluate_hard_negatives(
    tool_dataset: Dict[str, object],
    hard_negative_records: List[Dict[str, object]],
    top_k: int = 5,
    method: str = "toolscout",
) -> Dict[str, float]:
    runtime = build_runtime(
        tool_dataset,
        feedback_queries=hard_negative_records,
    )
    try:
        top1_hits = 0
        recall_hits = 0
        latencies_ms: List[float] = []

        for record in hard_negative_records:
            candidate_names = [record["correct_tool"]] + list(record["distractors"])
            candidate_tools = [
                runtime.tools_by_name[name]
                for name in candidate_names
                if name in runtime.tools_by_name
            ]
            if len(candidate_tools) != len(candidate_names):
                missing = set(candidate_names) - set(runtime.tools_by_name)
                raise ValueError(
                    "Unknown hard-negative tools: {0}".format(", ".join(sorted(missing)))
                )

            start = time.perf_counter()
            hits = search_with_method(
                record["query"],
                runtime=runtime,
                top_k=min(top_k, len(candidate_tools)),
                method=method,
                candidate_tools=candidate_tools,
            )
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

            names = [hit.tool.name for hit in hits]
            top1_hits += int(bool(names) and names[0] == record["correct_tool"])
            recall_hits += int(record["correct_tool"] in names)

        total = len(hard_negative_records)
        return {
            "precision_at_1": top1_hits / total,
            "recall_at_k": recall_hits / total,
            "avg_latency_ms": statistics.mean(latencies_ms),
        }
    finally:
        runtime.cleanup()


def render_results(results: List[Dict[str, float]], top_k: int) -> str:
    lines = [
        "| Method | Precision@1 | Recall@{0} | Avg Latency |".format(top_k),
        "| ------ | ----------: | ---------: | ----------: |",
    ]
    for row in results:
        lines.append(
            "| {method} | {precision_at_1:.3f} | {recall_at_k:.3f} | {avg_latency_ms:.3f} ms |".format(
                **row
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ToolScout under hard-negative distractors."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "tools_1000.json",
        help="Path to the tool dataset.",
    )
    parser.add_argument(
        "--hard-negatives",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "tool_hard_negatives.json",
        help="Path to the hard-negative benchmark dataset.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of candidate tools retrieved within each hard-negative set.",
    )
    parser.add_argument(
        "--method",
        choices=["random", "lexical", "bm25", "semantic", "semantic_rerank", "toolscout", "baseline", "all"],
        default="all",
        help="Method to evaluate.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write structured JSON results.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path to write the summary table as CSV.",
    )
    args = parser.parse_args()

    tool_dataset = load_json(args.dataset)
    hard_negatives = load_json(args.hard_negatives)
    stats = dataset_statistics(
        tool_dataset=tool_dataset,
        hard_negative_records=hard_negatives,
        extra_query_records=CROSS_LINGUAL_QUERIES,
    )

    results = []
    for method in expand_methods(args.method):
        metrics = evaluate_hard_negatives(
            tool_dataset=tool_dataset,
            hard_negative_records=hard_negatives,
            top_k=args.top_k,
            method=method,
        )
        metrics["method"] = method_label(method)
        results.append(metrics)

    print("ToolScout Hard Negative Evaluation")
    print(format_dataset_statistics(stats))
    print("")
    print(render_results(results, top_k=args.top_k))

    if args.output_json:
        write_json_report(
            args.output_json,
            {
                "evaluation": "hard_negative",
                "top_k": args.top_k,
                "dataset_statistics": stats,
                "results": results,
            },
        )
    if args.output_csv:
        write_csv_report(
            args.output_csv,
            rows=results,
            fieldnames=["method", "precision_at_1", "recall_at_k", "avg_latency_ms"],
        )


if __name__ == "__main__":
    main()
