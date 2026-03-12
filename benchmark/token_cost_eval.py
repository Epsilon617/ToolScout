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
        description="Estimate prompt token savings from ToolScout retrieval."
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
        help="Number of retrieved tools included in the reduced prompt.",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    registry = build_registry(dataset)
    retriever = ToolRetriever(registry=registry)
    retriever.fit()

    full_prompt = prompt_for_tools(registry.list_tools())
    full_tokens = estimate_tokens(full_prompt)
    reduced_tokens: List[int] = []

    for example in dataset["queries"]:
        hits = retriever.search(example["query"], top_k=args.top_k)
        reduced_prompt = prompt_for_tools([hit.tool for hit in hits])
        reduced_tokens.append(estimate_tokens(reduced_prompt))

    mean_reduced = statistics.mean(reduced_tokens)
    reduction_ratio = 1.0 - (mean_reduced / full_tokens)

    print("ToolScout Token Cost Evaluation")
    print("backend: {0}".format(retriever.backend_summary))
    print("queries: {0}".format(len(dataset["queries"])))
    print("full catalog prompt tokens: {0}".format(full_tokens))
    print("avg retrieved prompt tokens: {0:.1f}".format(mean_reduced))
    print("avg reduction: {0:.1%}".format(reduction_ratio))
    print("min retrieved prompt tokens: {0}".format(min(reduced_tokens)))
    print("max retrieved prompt tokens: {0}".format(max(reduced_tokens)))


if __name__ == "__main__":
    main()
