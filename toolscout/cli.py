from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from .registry.tool_registry import ToolRegistry
from .retriever.tool_retriever import ToolRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "tools_1000.json"


def build_retriever(dataset_path: Path) -> ToolRetriever:
    if not dataset_path.exists():
        raise SystemExit(
            "Dataset not found at {0}. Run `python datasets/generate_tools.py` first.".format(
                dataset_path
            )
        )

    registry = ToolRegistry.from_json(dataset_path)
    retriever = ToolRetriever(registry=registry)
    retriever.fit()
    return retriever


def render_top_tools(names: Iterable[str]) -> str:
    lines: List[str] = ["Top tools:"]
    for rank, name in enumerate(names, start=1):
        lines.append("{0} {1}".format(rank, name))
    return "\n".join(lines)


def command_search(args: argparse.Namespace) -> int:
    retriever = build_retriever(args.dataset)
    hits = retriever.search(args.query, top_k=args.top_k)
    print(render_top_tools(hit.tool.name for hit in hits))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ToolScout command line interface.")
    subparsers = parser.add_subparsers(dest="command")

    search_parser = subparsers.add_parser(
        "search", help="Retrieve the top-k relevant tools for a query."
    )
    search_parser.add_argument("query", help="The user query to search against the tool catalog.")
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved tools to return.",
    )
    search_parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to a JSON tool dataset. Defaults to datasets/tools_1000.json.",
    )
    search_parser.set_defaults(func=command_search)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
