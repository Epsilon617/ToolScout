import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from toolscout import ToolRegistry, ToolRetriever


DEFAULT_QUERIES = [
    "latest Nvidia news",
    "weather in Tokyo",
    "calculate square root",
    "translate hello to Spanish",
    "find restaurants near me",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a large-scale ToolScout retrieval demo."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "tools_1000.json",
        help="Path to the generated 1000-tool dataset.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of tools to display for each query.",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(
            "Dataset not found at {0}. Run `python datasets/generate_tools.py` first.".format(
                args.dataset
            )
        )

    registry = ToolRegistry.from_json(args.dataset)
    retriever = ToolRetriever(registry=registry)
    retriever.fit()

    print("ToolScout Retrieval Demo")
    print("dataset: {0}".format(args.dataset.name))
    print("backend: {0}".format(retriever.backend_summary))

    for query in DEFAULT_QUERIES:
        hits = retriever.search(query, top_k=args.top_k)
        print("")
        print("Query: {0}".format(query))
        print("")
        print("Top tools:")
        for rank, hit in enumerate(hits, start=1):
            print("{0} {1}".format(rank, hit.tool.name))


if __name__ == "__main__":
    main()
