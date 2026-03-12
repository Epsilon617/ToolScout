import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from toolscout import ToolRegistry, ToolRetriever


def load_dataset(path: Path) -> Dict[str, object]:
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


def reciprocal_rank(retrieved_names: List[str], expected: List[str]) -> float:
    ranks = [
        retrieved_names.index(tool_name) + 1
        for tool_name in expected
        if tool_name in retrieved_names
    ]
    if not ranks:
        return 0.0
    return 1.0 / min(ranks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ToolScout retrieval accuracy."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "synthetic_tools.json",
        help="Path to the synthetic benchmark dataset.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum k used for reporting recall@k and MRR.",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    registry = build_registry(dataset)
    retriever = ToolRetriever(registry=registry)
    retriever.fit()

    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_k = 0
    reciprocal_ranks: List[float] = []

    for example in dataset["queries"]:
        results = retriever.search(example["query"], top_k=args.top_k)
        names = [result.tool.name for result in results]
        expected = example["expected_tools"]
        reciprocal_ranks.append(reciprocal_rank(names, expected))

        if any(tool_name in names[:1] for tool_name in expected):
            recall_at_1 += 1
        if any(tool_name in names[: min(3, len(names))] for tool_name in expected):
            recall_at_3 += 1
        if any(tool_name in names for tool_name in expected):
            recall_at_k += 1

    total = len(dataset["queries"])
    print("ToolScout Retrieval Accuracy")
    print("backend: {0}".format(retriever.backend_summary))
    print("examples: {0}".format(total))
    print("recall@1: {0:.3f}".format(recall_at_1 / total))
    print("recall@3: {0:.3f}".format(recall_at_3 / total))
    print("recall@{0}: {1:.3f}".format(args.top_k, recall_at_k / total))
    print("mrr: {0:.3f}".format(statistics.mean(reciprocal_ranks)))


if __name__ == "__main__":
    main()
