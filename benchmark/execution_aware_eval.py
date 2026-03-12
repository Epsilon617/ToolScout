import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from toolscout import (
    ExecutionFeedbackStore,
    ToolExecutionSimulator,
    ToolRegistry,
    ToolRetriever,
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_registry(dataset: Dict[str, object]) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in dataset["tools"]:
        registry.register_tool(
            name=tool["name"],
            description=tool["description"],
            args=tool.get("args", tool.get("arguments", [])),
            category=tool.get("category"),
            dependencies=tool.get("dependencies", []),
            tags=tool.get("tags", []),
            examples=tool.get("examples", []),
        )
    return registry


def seed_feedback_history(
    retriever: ToolRetriever,
    query_dataset: List[Dict[str, str]],
    store: ExecutionFeedbackStore,
    top_k: int,
    warmup_passes: int,
    seed: int,
) -> None:
    store.clear()
    simulator = ToolExecutionSimulator(seed=seed)
    pending_records = []

    for _ in range(warmup_passes):
        for record in query_dataset:
            query = record["query"]
            hits = retriever.search(query, top_k=top_k)
            query_embedding = retriever.encoder.encode_query(query)
            for hit in hits:
                simulation = simulator.simulate(hit.tool.name)
                pending_records.append(
                    {
                        "query_embedding": query_embedding,
                        "tool": hit.tool.name,
                        "success": simulation.success,
                        "latency_ms": simulation.latency_ms,
                        "error": simulation.error,
                    }
                )

    if pending_records:
        store.record_executions(pending_records)


def evaluate_mode(
    retriever: ToolRetriever,
    query_dataset: List[Dict[str, str]],
    top_k: int,
    execution_aware: bool,
    feedback_store: ExecutionFeedbackStore,
    seed: int,
) -> Dict[str, float]:
    simulator = ToolExecutionSimulator(seed=seed)
    recall_hits = 0
    tool_success_hits = 0
    task_success_hits = 0
    total_latency_ms: List[float] = []
    search_latency_ms: List[float] = []

    for record in query_dataset:
        start = time.perf_counter()
        hits = retriever.search(
            record["query"],
            top_k=top_k,
            execution_aware=execution_aware,
            feedback_store=feedback_store,
        )
        retrieval_latency = (time.perf_counter() - start) * 1000.0
        search_latency_ms.append(retrieval_latency)

        names = [hit.tool.name for hit in hits]
        if record["correct_tool"] in names:
            recall_hits += 1

        if not hits:
            total_latency_ms.append(retrieval_latency)
            continue

        selected_tool = hits[0].tool.name
        simulation = simulator.simulate(selected_tool)
        total_latency_ms.append(retrieval_latency + simulation.latency_ms)
        tool_success_hits += int(simulation.success)
        task_success_hits += int(
            simulation.success and selected_tool == record["correct_tool"]
        )

    total = len(query_dataset)
    return {
        "tool_success_rate": tool_success_hits / total,
        "avg_latency_ms": statistics.mean(total_latency_ms),
        "avg_search_latency_ms": statistics.mean(search_latency_ms),
        "recall_at_k": recall_hits / total,
        "task_success_rate": task_success_hits / total,
    }


def render_results_table(results: List[Dict[str, object]], top_k: int) -> str:
    lines = [
        "| mode | tool success rate | avg latency | avg search latency | recall@{0} | task success rate |".format(
            top_k
        ),
        "|------|------------------:|------------:|-------------------:|----------:|------------------:|",
    ]
    for row in results:
        lines.append(
            "| {mode} | {tool_success_rate:.1%} | {avg_latency_ms:.3f} ms | {avg_search_latency_ms:.3f} ms | {recall_at_k:.3f} | {task_success_rate:.1%} |".format(
                **row
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare semantic retrieval against execution-aware reranking."
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
        help="Path to the query evaluation dataset.",
    )
    parser.add_argument(
        "--feedback-store",
        type=Path,
        default=PROJECT_ROOT / "data" / "execution_eval_feedback.json",
        help="Path to the benchmark feedback store.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved tools to evaluate.",
    )
    parser.add_argument(
        "--warmup-passes",
        type=int,
        default=10,
        help="Number of replay passes used to build execution history.",
    )
    parser.add_argument(
        "--warmup-seed",
        type=int,
        default=7,
        help="Random seed for warmup feedback generation.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=17,
        help="Random seed used for the evaluation simulator.",
    )
    args = parser.parse_args()

    tool_dataset = load_json(args.dataset)
    query_dataset = load_json(args.queries)
    registry = build_registry(tool_dataset)
    retriever = ToolRetriever(registry=registry)
    retriever.fit()

    feedback_store = ExecutionFeedbackStore(args.feedback_store)
    seed_feedback_history(
        retriever=retriever,
        query_dataset=query_dataset,
        store=feedback_store,
        top_k=args.top_k,
        warmup_passes=args.warmup_passes,
        seed=args.warmup_seed,
    )

    results = [
        {
            "mode": "baseline",
            **evaluate_mode(
                retriever=retriever,
                query_dataset=query_dataset,
                top_k=args.top_k,
                execution_aware=False,
                feedback_store=feedback_store,
                seed=args.eval_seed,
            ),
        },
        {
            "mode": "execution-aware",
            **evaluate_mode(
                retriever=retriever,
                query_dataset=query_dataset,
                top_k=args.top_k,
                execution_aware=True,
                feedback_store=feedback_store,
                seed=args.eval_seed,
            ),
        },
    ]

    print("ToolScout Execution-Aware Evaluation")
    print("backend: {0}".format(retriever.backend_summary))
    print("queries: {0}".format(len(query_dataset)))
    print("warmup records: {0}".format(len(feedback_store.records())))
    print("feedback store: {0}".format(feedback_store.path))
    print("")
    print(render_results_table(results, top_k=args.top_k))


if __name__ == "__main__":
    main()
