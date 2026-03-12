import argparse
import json
import math
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


def expand_tools(base_tools: List[Dict[str, object]], target_count: int) -> List[Dict[str, object]]:
    expanded: List[Dict[str, object]] = []
    replica = 0
    while len(expanded) < target_count:
        for tool in base_tools:
            entry = dict(tool)
            entry["name"] = "{0}_{1:04d}".format(tool["name"], replica)
            entry["description"] = "{0} Segment {1}.".format(
                tool["description"], replica
            )
            entry["examples"] = list(tool.get("examples", [])) + [
                "tenant {0} {1}".format(replica, tool["name"])
            ]
            expanded.append(entry)
            if len(expanded) >= target_count:
                break
        replica += 1
    return expanded


def build_registry(tools: List[Dict[str, object]]) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register_tool(
            name=tool["name"],
            description=tool["description"],
            args=tool.get("args", []),
            tags=tool.get("tags", []),
            examples=tool.get("examples", []),
        )
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure ToolScout scaling behavior."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "synthetic_tools.json",
        help="Path to the synthetic benchmark dataset.",
    )
    parser.add_argument(
        "--target-tools",
        type=int,
        default=1000,
        help="Total number of tools to index.",
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
    scaled_tools = expand_tools(dataset["tools"], args.target_tools)
    registry = build_registry(scaled_tools)
    retriever = ToolRetriever(registry=registry)

    build_start = time.perf_counter()
    retriever.fit()
    build_seconds = time.perf_counter() - build_start

    queries = [record["query"] for record in dataset["queries"]]
    search_latencies: List[float] = []
    for index in range(args.runs):
        query = queries[index % len(queries)]
        start = time.perf_counter()
        retriever.search(query, top_k=args.top_k)
        search_latencies.append(time.perf_counter() - start)

    print("ToolScout Scaling Test")
    print("backend: {0}".format(retriever.backend_summary))
    print("indexed tools: {0}".format(len(scaled_tools)))
    print("index build seconds: {0:.4f}".format(build_seconds))
    print(
        "avg search milliseconds: {0:.3f}".format(
            statistics.mean(search_latencies) * 1000.0
        )
    )
    print(
        "p95 search milliseconds: {0:.3f}".format(
            sorted(search_latencies)[math.ceil(0.95 * len(search_latencies)) - 1]
            * 1000.0
        )
    )


if __name__ == "__main__":
    main()

