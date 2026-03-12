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
    SkillRegistry,
    SkillRetriever,
    ToolGraph,
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


def estimate_tokens(text: str) -> int:
    return max(1, int(round(len(text) / 4.0)))


def prompt_for_tools(tools: List[object]) -> str:
    return "\n\n".join(tool.to_prompt_text() for tool in tools)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ToolScout on ToolBench-style single-tool selection."
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
        "--skills-dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "skills.json",
        help="Path to the skills dataset used for graph-aware routing.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved tools to evaluate.",
    )
    parser.add_argument(
        "--graph-aware",
        action="store_true",
        help="Evaluate skill routing plus tool graph dependency expansion.",
    )
    args = parser.parse_args()

    tool_dataset = load_json(args.dataset)
    query_dataset = load_json(args.queries)
    registry = build_registry(tool_dataset)
    retriever = ToolRetriever(registry=registry)
    retriever.fit()

    skill_retriever = None
    tool_graph = None
    mode = "standard"
    if args.graph_aware:
        skills = SkillRegistry.from_json(args.skills_dataset)
        skill_retriever = SkillRetriever(registry=skills)
        skill_retriever.fit()
        tool_graph = ToolGraph.from_registry(registry)
        mode = "graph-aware"

    recalls = 0
    latencies_ms: List[float] = []
    reduced_prompt_tokens: List[int] = []
    full_prompt_tokens = estimate_tokens(prompt_for_tools(registry.list_tools()))

    for record in query_dataset:
        start = time.perf_counter()
        if args.graph_aware:
            hits = skill_retriever.route(
                record["query"],
                tool_registry=registry,
                tool_retriever=retriever,
                tool_graph=tool_graph,
                top_k_tools=args.top_k,
                graph_aware=True,
            ).tools
        else:
            hits = retriever.search(record["query"], top_k=args.top_k)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

        names = [hit.tool.name for hit in hits]
        if record["correct_tool"] in names:
            recalls += 1

        reduced_prompt_tokens.append(
            estimate_tokens(prompt_for_tools([hit.tool for hit in hits]))
        )

    recall_at_k = recalls / len(query_dataset)
    avg_latency_ms = statistics.mean(latencies_ms)
    avg_reduction = 1.0 - (
        statistics.mean(reduced_prompt_tokens) / full_prompt_tokens
    )

    print("ToolScout Tool Selection Evaluation")
    print("mode: {0}".format(mode))
    print("backend: {0}".format(retriever.backend_summary))
    print("queries: {0}".format(len(query_dataset)))
    print("recall@{0}: {1:.3f}".format(args.top_k, recall_at_k))
    print("avg latency ms: {0:.3f}".format(avg_latency_ms))
    print("avg token reduction: {0:.1%}".format(avg_reduction))


if __name__ == "__main__":
    main()
