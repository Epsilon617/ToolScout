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

from toolscout import ToolRegistry, ToolRetriever


def load_dataset(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_registry(tools: List[Dict[str, object]]) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
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


def estimate_tokens(text: str) -> int:
    return max(1, int(round(len(text) / 4.0)))


def prompt_for_tools(tools: List[object]) -> str:
    return "\n\n".join(tool.to_prompt_text() for tool in tools)


def compute_recall_at_k(
    retriever: ToolRetriever, queries: List[Dict[str, object]], top_k: int
) -> float:
    hits = 0
    for query_record in queries:
        retrieved = retriever.retrieve_names(query_record["query"], top_k=top_k)
        if any(name in retrieved for name in query_record["expected_tools"]):
            hits += 1
    return hits / len(queries)


def compute_token_reduction(
    retriever: ToolRetriever,
    registry: ToolRegistry,
    queries: List[Dict[str, object]],
    top_k: int,
) -> float:
    full_prompt = prompt_for_tools(registry.list_tools())
    full_tokens = estimate_tokens(full_prompt)
    reduced = []
    for query_record in queries:
        hits = retriever.search(query_record["query"], top_k=top_k)
        reduced.append(estimate_tokens(prompt_for_tools([hit.tool for hit in hits])))
    return 1.0 - (statistics.mean(reduced) / full_tokens)


def compute_latency_ms(
    retriever: ToolRetriever, queries: List[Dict[str, object]], top_k: int, runs: int
) -> float:
    latencies: List[float] = []
    total_runs = max(runs, len(queries))
    for index in range(total_runs):
        query = queries[index % len(queries)]["query"]
        start = time.perf_counter()
        retriever.search(query, top_k=top_k)
        latencies.append((time.perf_counter() - start) * 1000.0)
    return statistics.mean(latencies)


def render_table(results: List[Dict[str, object]], top_k: int) -> str:
    lines = [
        "ToolScout Scaling Test",
        "",
        "| tools | latency | recall@{0} | token reduction |".format(top_k),
        "|------:|--------:|----------:|----------------:|",
    ]
    for row in results:
        lines.append(
            "| {tools} | {latency_ms:.3f} ms | {recall:.3f} | {token_reduction:.1%} |".format(
                **row
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure ToolScout scaling behavior."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "tools_1000.json",
        help="Path to the large synthetic benchmark dataset.",
    )
    parser.add_argument(
        "--tool-counts",
        default="10,100,1000",
        help="Comma-separated tool counts to evaluate.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=50,
        help="Number of repeated retrieval runs used for the latency average.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of tools retrieved per query.",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    tool_counts = [int(value.strip()) for value in args.tool_counts.split(",") if value.strip()]
    results: List[Dict[str, object]] = []
    backend_summary = None

    for tool_count in tool_counts:
        subset = dataset["tools"][:tool_count]
        available_names = {tool["name"] for tool in subset}
        queries = [
            record
            for record in dataset["queries"]
            if any(name in available_names for name in record["expected_tools"])
        ]
        if not queries:
            raise ValueError("No benchmark queries apply to tool subset size {0}.".format(tool_count))

        registry = build_registry(subset)
        retriever = ToolRetriever(registry=registry)
        retriever.fit()
        backend_summary = retriever.backend_summary

        results.append(
            {
                "tools": tool_count,
                "latency_ms": compute_latency_ms(
                    retriever, queries=queries, top_k=args.top_k, runs=args.runs
                ),
                "recall": compute_recall_at_k(
                    retriever, queries=queries, top_k=args.top_k
                ),
                "token_reduction": compute_token_reduction(
                    retriever,
                    registry=registry,
                    queries=queries,
                    top_k=args.top_k,
                ),
            }
        )

    print("backend: {0}".format(backend_summary))
    print(render_table(results, top_k=args.top_k))


if __name__ == "__main__":
    main()
