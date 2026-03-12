from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

from .execution_feedback import DEFAULT_FEEDBACK_STORE, ExecutionFeedbackStore
from .mcp_adapter import load_mcp_registry, load_mcp_tools
from .registry.tool_registry import ToolRegistry
from .retriever.tool_retriever import ToolRetriever
from .skill_registry import SkillRegistry
from .skill_retriever import SkillRetriever
from .tool_graph import ToolGraph
from .tool_simulator import ToolExecutionSimulator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "tools_1000.json"
DEFAULT_SKILLS_DATASET = PROJECT_ROOT / "datasets" / "skills.json"
DEFAULT_QUERY_DATASET = PROJECT_ROOT / "datasets" / "tool_queries.json"


def ensure_dataset_exists(dataset_path: Path, generation_hint: str) -> None:
    if not dataset_path.exists():
        raise SystemExit(
            "Dataset not found at {0}. {1}".format(dataset_path, generation_hint)
        )


def load_json_records(path: Path) -> object:
    ensure_dataset_exists(path, "Provide a valid JSON file path.")
    return json.loads(path.read_text(encoding="utf-8"))


def build_tool_registry(dataset_path: Path) -> ToolRegistry:
    ensure_dataset_exists(
        dataset_path,
        "Run `python datasets/generate_tools.py` first.",
    )
    return ToolRegistry.from_json(dataset_path)


def build_skill_registry(skills_path: Path) -> SkillRegistry:
    ensure_dataset_exists(
        skills_path,
        "Expected a skills dataset such as datasets/skills.json.",
    )
    return SkillRegistry.from_json(skills_path)


def build_retriever(dataset_path: Path) -> ToolRetriever:
    registry = build_tool_registry(dataset_path)
    retriever = ToolRetriever(registry=registry)
    retriever.fit()
    return retriever


def render_top_tools(names: Iterable[str]) -> str:
    lines: List[str] = ["Top tools:"]
    for rank, name in enumerate(names, start=1):
        lines.append("{0} {1}".format(rank, name))
    return "\n".join(lines)


def render_skill_route(skill_name: str, tool_names: Iterable[str]) -> str:
    lines: List[str] = [
        "Skill: {0}".format(skill_name),
        "",
        "Tools:",
    ]
    for rank, name in enumerate(tool_names, start=1):
        lines.append("{0} {1}".format(rank, name))
    return "\n".join(lines)


def render_feedback_stats(
    store: ExecutionFeedbackStore,
    tool_name: Optional[str] = None,
    limit: int = 10,
) -> str:
    stats = store.tool_stats(tool_name=tool_name)
    lines: List[str] = [
        "Feedback stats",
        "store: {0}".format(store.path),
        "records: {0}".format(len(store.records())),
        "tools: {0}".format(len(stats)),
    ]

    if not stats:
        lines.append("")
        lines.append("No execution feedback has been recorded yet.")
        return "\n".join(lines)

    lines.append("")
    for rank, item in enumerate(stats[:limit], start=1):
        lines.append(
            "{0} {1} | runs {2} | success {3:.1%} | avg latency {4:.1f} ms".format(
                rank,
                item["tool"],
                item["count"],
                item["success_rate"],
                item["average_latency_ms"],
            )
        )
    return "\n".join(lines)


def command_search(args: argparse.Namespace) -> int:
    registry = build_tool_registry(args.dataset)
    retriever = ToolRetriever(registry=registry)
    retriever.fit()
    feedback_store = (
        ExecutionFeedbackStore(args.feedback_store)
        if args.execution_aware
        else None
    )

    if args.graph_aware:
        skill_registry = build_skill_registry(args.skills_dataset)
        skill_retriever = SkillRetriever(registry=skill_registry)
        skill_retriever.fit()
        hits = skill_retriever.route(
            args.query,
            tool_registry=registry,
            tool_retriever=retriever,
            tool_graph=ToolGraph.from_registry(registry),
            top_k_tools=args.top_k,
            graph_aware=True,
            execution_aware=args.execution_aware,
            feedback_store=feedback_store,
        ).tools
    else:
        hits = retriever.search(
            args.query,
            top_k=args.top_k,
            execution_aware=args.execution_aware,
            feedback_store=feedback_store,
        )

    print(render_top_tools(hit.tool.name for hit in hits))
    return 0


def command_skill(args: argparse.Namespace) -> int:
    tool_registry = build_tool_registry(args.dataset)
    skill_registry = build_skill_registry(args.skills_dataset)
    tool_retriever = ToolRetriever(registry=tool_registry)
    tool_retriever.fit()
    skill_retriever = SkillRetriever(registry=skill_registry)
    skill_retriever.fit()
    feedback_store = (
        ExecutionFeedbackStore(args.feedback_store)
        if args.execution_aware
        else None
    )

    routed = skill_retriever.route(
        args.query,
        tool_registry=tool_registry,
        tool_retriever=tool_retriever,
        tool_graph=ToolGraph.from_registry(tool_registry),
        top_k_tools=args.top_k,
        graph_aware=True,
        execution_aware=args.execution_aware,
        feedback_store=feedback_store,
    )
    print(render_skill_route(routed.skill.name, (hit.tool.name for hit in routed.tools)))
    return 0


def command_feedback_stats(args: argparse.Namespace) -> int:
    store = ExecutionFeedbackStore(args.feedback_store)
    print(render_feedback_stats(store, tool_name=args.tool, limit=args.top))
    return 0


def command_simulate_runs(args: argparse.Namespace) -> int:
    tool_registry = build_tool_registry(args.dataset)
    tool_retriever = ToolRetriever(registry=tool_registry)
    tool_retriever.fit()

    skill_retriever = None
    tool_graph = None
    if args.graph_aware:
        skill_registry = build_skill_registry(args.skills_dataset)
        skill_retriever = SkillRetriever(registry=skill_registry)
        skill_retriever.fit()
        tool_graph = ToolGraph.from_registry(tool_registry)

    query_records = load_json_records(args.queries)
    if not isinstance(query_records, list):
        raise SystemExit("Expected a JSON array of query records.")
    tracks_recall = any("correct_tool" in record for record in query_records)

    store = ExecutionFeedbackStore(args.feedback_store)
    if args.clear:
        store.clear()
    simulator = ToolExecutionSimulator(seed=args.seed)

    records_written = 0
    successes = 0
    latencies_ms: List[float] = []
    recall_hits = 0
    pending_records = []

    for _ in range(args.passes):
        for record in query_records:
            query = record["query"]
            if args.graph_aware:
                hits = skill_retriever.route(
                    query,
                    tool_registry=tool_registry,
                    tool_retriever=tool_retriever,
                    tool_graph=tool_graph,
                    top_k_tools=args.top_k,
                    graph_aware=True,
                    execution_aware=args.execution_aware,
                    feedback_store=store if args.execution_aware else None,
                ).tools
            else:
                hits = tool_retriever.search(
                    query,
                    top_k=args.top_k,
                    execution_aware=args.execution_aware,
                    feedback_store=store if args.execution_aware else None,
                )

            if not hits:
                continue

            selected_tool = hits[0].tool.name
            simulation = simulator.simulate(selected_tool)
            pending_records.append(
                {
                    "query_embedding": tool_retriever.encoder.encode_query(query),
                    "tool": selected_tool,
                    "success": simulation.success,
                    "latency_ms": simulation.latency_ms,
                    "error": simulation.error,
                }
            )
            records_written += 1
            successes += int(simulation.success)
            latencies_ms.append(simulation.latency_ms)

            if "correct_tool" in record:
                retrieved_names = [hit.tool.name for hit in hits]
                if record["correct_tool"] in retrieved_names:
                    recall_hits += 1

    if pending_records:
        store.record_executions(pending_records)

    lines = [
        "Simulation runs complete",
        "store: {0}".format(store.path),
        "records written: {0}".format(records_written),
    ]
    if records_written:
        lines.append("tool success rate: {0:.1%}".format(successes / records_written))
        lines.append(
            "avg latency ms: {0:.3f}".format(sum(latencies_ms) / len(latencies_ms))
        )
        if tracks_recall:
            total_queries = len(query_records) * args.passes
            lines.append(
                "recall@{0}: {1:.3f}".format(args.top_k, recall_hits / total_queries)
            )

    print("\n".join(lines))
    return 0


def command_load_mcp(args: argparse.Namespace) -> int:
    ensure_dataset_exists(args.path, "Provide a JSON file with MCP tool definitions.")
    tools = load_mcp_tools(args.path)
    lines = [
        "Loaded {0} MCP tools from {1}".format(len(tools), args.path),
    ]

    if args.query:
        registry = load_mcp_registry(args.path)
        retriever = ToolRetriever(registry=registry)
        retriever.fit()
        hits = retriever.search(args.query, top_k=args.top_k)
        lines.append("")
        lines.append("Query: {0}".format(args.query))
        lines.append("")
        lines.append(render_top_tools(hit.tool.name for hit in hits))
    else:
        preview = tools[: args.preview]
        if preview:
            lines.append("")
            lines.append("Preview:")
            for rank, tool in enumerate(preview, start=1):
                arguments = ", ".join(tool.get("arguments", [])) or "none"
                lines.append(
                    "{0} {1} ({2})".format(rank, tool["name"], arguments)
                )

    print("\n".join(lines))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ToolScout command line interface.")
    subparsers = parser.add_subparsers(dest="command")

    search_parser = subparsers.add_parser(
        "search", help="Retrieve the top-k relevant tools for a query."
    )
    search_parser.add_argument(
        "query", help="The user query to search against the tool catalog."
    )
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
    search_parser.add_argument(
        "--graph-aware",
        action="store_true",
        help="Route through the skill layer and expand dependencies from the tool graph.",
    )
    search_parser.add_argument(
        "--execution-aware",
        action="store_true",
        help="Rerank candidates using execution success rates and latency feedback.",
    )
    search_parser.add_argument(
        "--skills-dataset",
        type=Path,
        default=DEFAULT_SKILLS_DATASET,
        help="Path to a JSON skills dataset. Used with --graph-aware.",
    )
    search_parser.add_argument(
        "--feedback-store",
        type=Path,
        default=DEFAULT_FEEDBACK_STORE,
        help="Path to the execution feedback store. Used with --execution-aware.",
    )
    search_parser.set_defaults(func=command_search)

    skill_parser = subparsers.add_parser(
        "skill", help="Retrieve the most relevant skill and its routed tools."
    )
    skill_parser.add_argument("query", help="The user query to route through skills.")
    skill_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of tools to return for the selected skill.",
    )
    skill_parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to a JSON tool dataset. Defaults to datasets/tools_1000.json.",
    )
    skill_parser.add_argument(
        "--skills-dataset",
        type=Path,
        default=DEFAULT_SKILLS_DATASET,
        help="Path to a JSON skills dataset. Defaults to datasets/skills.json.",
    )
    skill_parser.add_argument(
        "--execution-aware",
        action="store_true",
        help="Rerank routed tools using execution success rates and latency feedback.",
    )
    skill_parser.add_argument(
        "--feedback-store",
        type=Path,
        default=DEFAULT_FEEDBACK_STORE,
        help="Path to the execution feedback store. Used with --execution-aware.",
    )
    skill_parser.set_defaults(func=command_skill)

    feedback_parser = subparsers.add_parser(
        "feedback-stats", help="Inspect recorded execution feedback."
    )
    feedback_parser.add_argument(
        "--feedback-store",
        type=Path,
        default=DEFAULT_FEEDBACK_STORE,
        help="Path to the execution feedback store.",
    )
    feedback_parser.add_argument(
        "--tool",
        help="Optional tool name to inspect.",
    )
    feedback_parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Maximum number of tools to show.",
    )
    feedback_parser.set_defaults(func=command_feedback_stats)

    simulate_parser = subparsers.add_parser(
        "simulate-runs",
        help="Simulate tool executions and record feedback for a query set.",
    )
    simulate_parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERY_DATASET,
        help="Path to a JSON array of query records.",
    )
    simulate_parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to a JSON tool dataset. Defaults to datasets/tools_1000.json.",
    )
    simulate_parser.add_argument(
        "--skills-dataset",
        type=Path,
        default=DEFAULT_SKILLS_DATASET,
        help="Path to a JSON skills dataset. Used with --graph-aware.",
    )
    simulate_parser.add_argument(
        "--feedback-store",
        type=Path,
        default=DEFAULT_FEEDBACK_STORE,
        help="Path to the execution feedback store.",
    )
    simulate_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved tools per query.",
    )
    simulate_parser.add_argument(
        "--passes",
        type=int,
        default=1,
        help="Number of times to replay the query dataset.",
    )
    simulate_parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for the execution simulator.",
    )
    simulate_parser.add_argument(
        "--graph-aware",
        action="store_true",
        help="Route queries through skills and the tool dependency graph.",
    )
    simulate_parser.add_argument(
        "--execution-aware",
        action="store_true",
        help="Use execution-aware reranking while simulating runs.",
    )
    simulate_parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the feedback store before writing new records.",
    )
    simulate_parser.set_defaults(func=command_simulate_runs)

    load_mcp_parser = subparsers.add_parser(
        "load-mcp", help="Load MCP-style tool definitions into ToolScout."
    )
    load_mcp_parser.add_argument(
        "path",
        type=Path,
        help="Path to a JSON file containing MCP-style tool definitions.",
    )
    load_mcp_parser.add_argument(
        "--query",
        help="Optional query to run against the loaded MCP tool set.",
    )
    load_mcp_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of tools to return when used with --query.",
    )
    load_mcp_parser.add_argument(
        "--preview",
        type=int,
        default=10,
        help="Number of loaded tools to preview when --query is omitted.",
    )
    load_mcp_parser.set_defaults(func=command_load_mcp)

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
