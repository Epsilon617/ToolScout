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
    arguments_are_valid,
    build_runtime,
    dataset_statistics,
    expand_methods,
    format_dataset_statistics,
    load_json,
    method_label,
    mock_generate_arguments,
    search_with_method,
)
from toolscout import ToolExecutionSimulator


def evaluate_e2e(
    tool_dataset: Dict[str, object],
    query_dataset: List[Dict[str, object]],
    top_k: int = 5,
    method: str = "toolscout",
    seed: int = 17,
) -> Dict[str, float]:
    runtime = build_runtime(
        tool_dataset,
        feedback_queries=query_dataset,
    )
    try:
        simulator = ToolExecutionSimulator(seed=seed)
        pass_at_1 = 0
        pass_at_k = 0
        execution_successes = 0
        total_latencies_ms: List[float] = []
        search_latencies_ms: List[float] = []

        for record in query_dataset:
            start = time.perf_counter()
            hits = search_with_method(
                record["query"],
                runtime=runtime,
                top_k=top_k,
                method=method,
            )
            retrieval_latency = (time.perf_counter() - start) * 1000.0
            search_latencies_ms.append(retrieval_latency)

            if not hits:
                total_latencies_ms.append(retrieval_latency)
                continue

            execution_cache = {}
            top_hit = hits[0]
            top_arguments = mock_generate_arguments(record["query"], top_hit.tool)
            top_args_valid = arguments_are_valid(top_hit.tool, top_arguments)

            if top_args_valid:
                top_execution = simulator.simulate(top_hit.tool.name)
                execution_cache[top_hit.tool.name] = top_execution
                execution_success = top_execution.success
                total_latencies_ms.append(retrieval_latency + top_execution.latency_ms)
                execution_successes += int(execution_success)
                pass_at_1 += int(
                    execution_success and top_hit.tool.name == record["correct_tool"]
                )
            else:
                total_latencies_ms.append(retrieval_latency)

            solved_at_k = False
            for hit in hits:
                candidate_arguments = mock_generate_arguments(record["query"], hit.tool)
                if not arguments_are_valid(hit.tool, candidate_arguments):
                    continue
                execution = execution_cache.get(hit.tool.name)
                if execution is None:
                    execution = simulator.simulate(hit.tool.name)
                    execution_cache[hit.tool.name] = execution
                if execution.success and hit.tool.name == record["correct_tool"]:
                    solved_at_k = True
                    break
            pass_at_k += int(solved_at_k)

        total = len(query_dataset)
        return {
            "pass_at_1": pass_at_1 / total,
            "pass_at_k": pass_at_k / total,
            "execution_success_rate": execution_successes / total,
            "avg_latency_ms": statistics.mean(total_latencies_ms),
            "avg_search_latency_ms": statistics.mean(search_latencies_ms),
            "task_success_rate": pass_at_1 / total,
        }
    finally:
        runtime.cleanup()


def render_results(results: List[Dict[str, float]], top_k: int) -> str:
    lines = [
        "| Method | Pass@1 | Pass@{0} | Execution Success | Avg Latency | Avg Search Latency |".format(
            top_k
        ),
        "| ------ | -----: | ------: | ----------------: | ----------: | -----------------: |",
    ]
    for row in results:
        lines.append(
            "| {method} | {pass_at_1:.3f} | {pass_at_k:.3f} | {execution_success_rate:.1%} | {avg_latency_ms:.3f} ms | {avg_search_latency_ms:.3f} ms |".format(
                **row
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run end-to-end ToolScout evaluation with mock planning and simulated tool execution."
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
        help="Number of retrieved tools considered for pass@k.",
    )
    parser.add_argument(
        "--method",
        choices=["random", "semantic", "semantic_rerank", "toolscout", "baseline", "all"],
        default="all",
        help="Method to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for the execution simulator.",
    )
    args = parser.parse_args()

    tool_dataset = load_json(args.dataset)
    query_dataset = load_json(args.queries)
    stats = dataset_statistics(
        tool_dataset=tool_dataset,
        query_records=query_dataset,
        extra_query_records=CROSS_LINGUAL_QUERIES,
    )

    results = []
    for method in expand_methods(args.method):
        metrics = evaluate_e2e(
            tool_dataset=tool_dataset,
            query_dataset=query_dataset,
            top_k=args.top_k,
            method=method,
            seed=args.seed,
        )
        metrics["method"] = method_label(method)
        results.append(metrics)

    print("ToolScout End-to-End Evaluation")
    print(format_dataset_statistics(stats))
    print("")
    print(render_results(results, top_k=args.top_k))


if __name__ == "__main__":
    main()
