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
    format_dataset_statistics,
    load_json,
    method_label,
    mutate_tool_records,
    replicate_tool_records,
    search_with_method,
    write_csv_report,
    write_json_report,
)


def evaluate_recall_and_latency(
    tool_records: List[Dict[str, object]],
    queries: List[Dict[str, object]],
    top_k: int,
    method: str,
) -> Dict[str, float]:
    runtime = build_runtime(
        {"tools": tool_records},
        feedback_queries=queries,
    )
    try:
        recall_hits = 0
        precision_hits = 0
        latencies_ms: List[float] = []

        for record in queries:
            start = time.perf_counter()
            hits = search_with_method(
                record["query"],
                runtime=runtime,
                top_k=top_k,
                method=method,
            )
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            names = [hit.tool.name for hit in hits]
            precision_hits += int(bool(names) and names[0] == record["correct_tool"])
            recall_hits += int(record["correct_tool"] in names)

        total = len(queries)
        return {
            "precision_at_1": precision_hits / total,
            "recall_at_k": recall_hits / total,
            "avg_latency_ms": statistics.mean(latencies_ms),
        }
    finally:
        runtime.cleanup()


def evaluate_robustness(
    tool_dataset: Dict[str, object],
    query_dataset: List[Dict[str, object]],
    top_k: int,
    mutation_rate: float,
    mutation_seed: int,
    sprawl_targets: List[int],
    method: str,
) -> Dict[str, object]:
    base_tools = list(tool_dataset["tools"])

    original_metrics = evaluate_recall_and_latency(
        base_tools,
        query_dataset,
        top_k=top_k,
        method=method,
    )
    mutated_tools = mutate_tool_records(
        base_tools,
        seed=mutation_seed,
        mutation_rate=mutation_rate,
    )
    mutated_metrics = evaluate_recall_and_latency(
        mutated_tools,
        query_dataset,
        top_k=top_k,
        method=method,
    )

    sprawl_results = []
    for target in sprawl_targets:
        expanded_tools = replicate_tool_records(base_tools, target_count=target)
        metrics = evaluate_recall_and_latency(
            expanded_tools,
            query_dataset,
            top_k=top_k,
            method=method,
        )
        metrics["tool_count"] = target
        sprawl_results.append(metrics)

    cross_lingual_metrics = evaluate_recall_and_latency(
        base_tools,
        CROSS_LINGUAL_QUERIES,
        top_k=top_k,
        method=method,
    )

    return {
        "schema_mutation": {
            "original": original_metrics,
            "mutated": mutated_metrics,
        },
        "tool_sprawl": sprawl_results,
        "cross_lingual": cross_lingual_metrics,
    }


def robustness_rows(metrics: Dict[str, object], top_k: int) -> List[Dict[str, object]]:
    schema = metrics["schema_mutation"]
    sprawl_rows = metrics["tool_sprawl"]
    rows = [
        {
            "test": "Schema Mutation",
            "metric": "Recall@{0}".format(top_k),
            "value": schema["mutated"]["recall_at_k"],
        },
        {
            "test": "Schema Mutation",
            "metric": "Precision@1",
            "value": schema["mutated"]["precision_at_1"],
        },
        {
            "test": "Schema Mutation",
            "metric": "Latency",
            "value": schema["mutated"]["avg_latency_ms"],
        },
        {
            "test": "Cross-Lingual Retrieval",
            "metric": "Recall@{0}".format(top_k),
            "value": metrics["cross_lingual"]["recall_at_k"],
        },
        {
            "test": "Cross-Lingual Retrieval",
            "metric": "Precision@1",
            "value": metrics["cross_lingual"]["precision_at_1"],
        },
        {
            "test": "Cross-Lingual Retrieval",
            "metric": "Latency",
            "value": metrics["cross_lingual"]["avg_latency_ms"],
        },
    ]
    for row in sprawl_rows:
        rows.append(
            {
                "test": "Tool Sprawl {0}".format(row["tool_count"]),
                "metric": "Recall@{0}".format(top_k),
                "value": row["recall_at_k"],
            }
        )
        rows.append(
            {
                "test": "Tool Sprawl {0}".format(row["tool_count"]),
                "metric": "Latency",
                "value": row["avg_latency_ms"],
            }
        )
    return rows


def render_summary_table(metrics: Dict[str, object], top_k: int) -> str:
    lines = [
        "| Test | Metric | Result |",
        "| ---- | ------ | ------ |",
    ]
    for row in robustness_rows(metrics, top_k=top_k):
        suffix = " ms" if row["metric"] == "Latency" else ""
        lines.append(
            "| {test} | {metric} | {value:.3f}{suffix} |".format(
                suffix=suffix,
                **row
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stress-test ToolScout under schema mutation, tool sprawl, and cross-lingual queries."
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
        "--top-k",
        type=int,
        default=5,
        help="Number of tools to retrieve.",
    )
    parser.add_argument(
        "--method",
        choices=["random", "lexical", "bm25", "semantic", "semantic_rerank", "toolscout", "baseline"],
        default="toolscout",
        help="Method to stress-test.",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.35,
        help="Fraction of tools whose schema fields are mutated.",
    )
    parser.add_argument(
        "--mutation-seed",
        type=int,
        default=7,
        help="Random seed for schema mutation.",
    )
    parser.add_argument(
        "--sprawl-targets",
        default="100,1000,10000",
        help="Comma-separated tool counts for tool sprawl evaluation.",
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
    query_dataset = load_json(args.queries)
    sprawl_targets = [int(value.strip()) for value in args.sprawl_targets.split(",") if value.strip()]
    metrics = evaluate_robustness(
        tool_dataset=tool_dataset,
        query_dataset=query_dataset,
        top_k=args.top_k,
        mutation_rate=args.mutation_rate,
        mutation_seed=args.mutation_seed,
        sprawl_targets=sprawl_targets,
        method=args.method,
    )
    stats = dataset_statistics(
        tool_dataset=tool_dataset,
        query_records=query_dataset,
        extra_query_records=CROSS_LINGUAL_QUERIES,
    )

    print("ToolScout Robustness Evaluation")
    print("method: {0}".format(method_label(args.method)))
    print(format_dataset_statistics(stats))
    print("")
    print(render_summary_table(metrics, top_k=args.top_k))

    if args.output_json:
        write_json_report(
            args.output_json,
            {
                "evaluation": "robustness",
                "method": args.method,
                "top_k": args.top_k,
                "dataset_statistics": stats,
                "metrics": metrics,
            },
        )
    if args.output_csv:
        write_csv_report(
            args.output_csv,
            rows=robustness_rows(metrics, top_k=args.top_k),
            fieldnames=["test", "metric", "value"],
        )


if __name__ == "__main__":
    main()
