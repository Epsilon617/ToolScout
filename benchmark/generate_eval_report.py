import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.e2e_eval import evaluate_e2e
from benchmark.eval_utils import (
    CROSS_LINGUAL_QUERIES,
    build_runtime,
    dataset_statistics,
    expand_methods,
    format_dataset_statistics,
    load_json,
    method_label,
    search_with_method,
)
from benchmark.hard_negative_eval import evaluate_hard_negatives
from benchmark.robustness_eval import evaluate_robustness


def evaluate_recall(
    tool_dataset: Dict[str, object],
    query_dataset: List[Dict[str, object]],
    top_k: int,
    method: str,
) -> Dict[str, float]:
    runtime = build_runtime(
        tool_dataset,
        feedback_queries=query_dataset,
    )
    try:
        recall_hits = 0
        latencies_ms: List[float] = []
        for record in query_dataset:
            start = time.perf_counter()
            hits = search_with_method(
                record["query"],
                runtime=runtime,
                top_k=top_k,
                method=method,
            )
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            names = [hit.tool.name for hit in hits]
            recall_hits += int(record["correct_tool"] in names)

        total = len(query_dataset)
        return {
            "recall_at_k": recall_hits / total,
            "avg_search_latency_ms": statistics.mean(latencies_ms),
        }
    finally:
        runtime.cleanup()


def execution_aware_gain(baseline_task_success: float, toolscout_task_success: float) -> float:
    if baseline_task_success <= 0:
        return 0.0
    return (toolscout_task_success - baseline_task_success) / baseline_task_success


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a multi-layer ToolScout evaluation report."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "tools_1000.json",
        help="Path to the tool dataset.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "tool_queries.json",
        help="Path to the query dataset.",
    )
    parser.add_argument(
        "--hard-negatives",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "tool_hard_negatives.json",
        help="Path to the hard-negative dataset.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of tools to retrieve for recall and pass@k metrics.",
    )
    args = parser.parse_args()

    tool_dataset = load_json(args.dataset)
    query_dataset = load_json(args.queries)
    hard_negative_dataset = load_json(args.hard_negatives)
    stats = dataset_statistics(
        tool_dataset=tool_dataset,
        query_records=query_dataset,
        hard_negative_records=hard_negative_dataset,
        extra_query_records=CROSS_LINGUAL_QUERIES,
    )

    comparison_rows = []
    summary_by_method: Dict[str, Dict[str, float]] = {}
    for method in expand_methods("all"):
        recall_metrics = evaluate_recall(
            tool_dataset=tool_dataset,
            query_dataset=query_dataset,
            top_k=args.top_k,
            method=method,
        )
        hard_negative_metrics = evaluate_hard_negatives(
            tool_dataset=tool_dataset,
            hard_negative_records=hard_negative_dataset,
            top_k=min(args.top_k, 4),
            method=method,
        )
        e2e_metrics = evaluate_e2e(
            tool_dataset=tool_dataset,
            query_dataset=query_dataset,
            top_k=args.top_k,
            method=method,
        )
        summary_by_method[method] = {
            "precision_at_1": hard_negative_metrics["precision_at_1"],
            "recall_at_k": recall_metrics["recall_at_k"],
            "pass_at_1": e2e_metrics["pass_at_1"],
            "avg_latency_ms": e2e_metrics["avg_latency_ms"],
            "task_success_rate": e2e_metrics["task_success_rate"],
        }
        comparison_rows.append(
            {
                "method": method_label(method),
                "precision_at_1": hard_negative_metrics["precision_at_1"],
                "recall_at_k": recall_metrics["recall_at_k"],
                "pass_at_1": e2e_metrics["pass_at_1"],
                "avg_latency_ms": e2e_metrics["avg_latency_ms"],
            }
        )

    semantic_metrics = summary_by_method["semantic"]
    toolscout_metrics = summary_by_method["toolscout"]
    gain = execution_aware_gain(
        baseline_task_success=semantic_metrics["task_success_rate"],
        toolscout_task_success=toolscout_metrics["task_success_rate"],
    )

    robustness = evaluate_robustness(
        tool_dataset=tool_dataset,
        query_dataset=query_dataset,
        top_k=args.top_k,
        mutation_rate=0.35,
        mutation_seed=7,
        sprawl_targets=[100, 1000, 10000],
        method="toolscout",
    )

    lines = [
        "ToolScout Evaluation Report",
        format_dataset_statistics(stats),
        "",
        "| Method | Precision@1 | Recall@{0} | Pass@1 | Avg Latency |".format(args.top_k),
        "| ------ | ----------: | ---------: | -----: | ----------: |",
    ]
    for row in comparison_rows:
        lines.append(
            "| {method} | {precision_at_1:.3f} | {recall_at_k:.3f} | {pass_at_1:.3f} | {avg_latency_ms:.3f} ms |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "| Metric | Baseline | ToolScout |",
            "| ------ | -------: | --------: |",
            "| Precision@1 | {0:.3f} | {1:.3f} |".format(
                semantic_metrics["precision_at_1"],
                toolscout_metrics["precision_at_1"],
            ),
            "| Pass@1 | {0:.3f} | {1:.3f} |".format(
                semantic_metrics["pass_at_1"],
                toolscout_metrics["pass_at_1"],
            ),
            "| Execution Gain | {0:+.1%} |  |".format(gain),
            "",
            "Robustness summary",
            "Schema Mutation Recall@{0}: {1:.3f}".format(
                args.top_k,
                robustness["schema_mutation"]["mutated"]["recall_at_k"],
            ),
            "Cross-Lingual Recall@{0}: {1:.3f}".format(
                args.top_k,
                robustness["cross_lingual"]["recall_at_k"],
            ),
            "Tool Sprawl 10k Latency: {0:.3f} ms".format(
                robustness["tool_sprawl"][-1]["avg_latency_ms"]
            ),
        ]
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
