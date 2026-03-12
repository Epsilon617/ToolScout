from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from toolscout import (
    ExecutionFeedbackStore,
    RetrievalResult,
    SkillRegistry,
    SkillRetriever,
    ToolDefinition,
    ToolExecutionSimulator,
    ToolGraph,
    ToolRegistry,
    ToolRetriever,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SKILLS_DATASET = PROJECT_ROOT / "datasets" / "skills.json"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")

METHOD_ORDER = ["random", "semantic", "semantic_rerank", "toolscout"]
METHOD_LABELS = {
    "random": "Random Retrieval",
    "semantic": "Semantic Retrieval",
    "semantic_rerank": "Semantic + Rerank",
    "toolscout": "ToolScout",
}
METHOD_ALIASES = {
    "baseline": "semantic",
    "random_retrieval": "random",
    "semantic_retrieval": "semantic",
    "semantic_plus_rerank": "semantic_rerank",
}

COMPANY_TICKERS = {
    "nvidia": "NVDA",
    "英伟达": "NVDA",
    "microsoft": "MSFT",
    "apple": "AAPL",
    "tesla": "TSLA",
}

LANGUAGE_NAMES = {
    "spanish": "Spanish",
    "西班牙语": "Spanish",
    "japanese": "Japanese",
    "日语": "Japanese",
    "french": "French",
    "法语": "French",
    "german": "German",
    "德语": "German",
    "chinese": "Chinese",
    "中文": "Chinese",
}

CROSS_LINGUAL_QUERIES = [
    {"query": "帮我查英伟达股价", "correct_tool": "finance_api"},
    {"query": "最近的英伟达新闻", "correct_tool": "news_search"},
    {"query": "东京天气怎么样", "correct_tool": "weather_api"},
    {"query": "把 hello 翻译成西班牙语", "correct_tool": "translation_service"},
    {"query": "附近有什么餐馆", "correct_tool": "maps_places_api"},
    {"query": "搜索向量数据库资料", "correct_tool": "web_search"},
]


@dataclass
class EvaluationRuntime:
    registry: ToolRegistry
    retriever: ToolRetriever
    tools_by_name: Dict[str, ToolDefinition]
    tool_graph: ToolGraph
    skill_registry: Optional[SkillRegistry]
    skill_retriever: Optional[SkillRetriever]
    feedback_store: ExecutionFeedbackStore
    _tempdir: Optional[tempfile.TemporaryDirectory] = None

    def cleanup(self) -> None:
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_registry(dataset: Mapping[str, object]) -> ToolRegistry:
    registry = ToolRegistry()
    records = dataset["tools"] if isinstance(dataset, Mapping) else dataset
    for tool in records:
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


def build_retriever(dataset: Mapping[str, object]) -> ToolRetriever:
    registry = build_registry(dataset)
    retriever = ToolRetriever(registry=registry)
    retriever.fit()
    return retriever


def normalize_method(method: str) -> str:
    normalized = METHOD_ALIASES.get(method, method)
    if normalized not in METHOD_LABELS:
        raise ValueError("Unknown evaluation method '{0}'.".format(method))
    return normalized


def expand_methods(selection: str) -> List[str]:
    if selection == "all":
        return list(METHOD_ORDER)
    return [normalize_method(selection)]


def method_label(method: str) -> str:
    return METHOD_LABELS[normalize_method(method)]


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def tool_text(tool: ToolDefinition) -> str:
    return " | ".join(
        [
            tool.name,
            tool.description,
            " ".join(tool.args),
            " ".join(tool.tags),
            " ".join(tool.examples),
            tool.category or "",
        ]
    )


def estimate_tokens(text: str) -> int:
    return max(1, int(round(len(text) / 4.0)))


def prompt_for_tools(tools: Sequence[ToolDefinition]) -> str:
    return "\n\n".join(tool.to_prompt_text() for tool in tools)


def detect_languages(records: Sequence[Mapping[str, object]]) -> List[str]:
    found = set()
    for record in records:
        query = str(record.get("query", ""))
        if re.search(r"[\u4e00-\u9fff]", query):
            found.add("Chinese")
        if re.search(r"[A-Za-z]", query):
            found.add("English")
    ordered = [language for language in ["English", "Chinese"] if language in found]
    leftovers = sorted(found.difference(ordered))
    return ordered + leftovers


def dataset_statistics(
    tool_dataset: Mapping[str, object],
    query_records: Optional[Sequence[Mapping[str, object]]] = None,
    hard_negative_records: Optional[Sequence[Mapping[str, object]]] = None,
    extra_query_records: Optional[Sequence[Mapping[str, object]]] = None,
) -> Dict[str, object]:
    combined_queries: List[Mapping[str, object]] = []
    if query_records is not None:
        combined_queries.extend(query_records)
    if hard_negative_records is not None:
        combined_queries.extend(hard_negative_records)
    if extra_query_records is not None:
        combined_queries.extend(extra_query_records)

    languages = detect_languages(combined_queries)
    return {
        "queries": len(query_records) if query_records is not None else len(hard_negative_records or []),
        "tools": len(tool_dataset["tools"]),
        "hard_negatives": len(hard_negative_records or []),
        "languages": " + ".join(languages) if languages else "Unknown",
    }


def format_dataset_statistics(stats: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "Dataset statistics:",
            "queries: {0}".format(stats["queries"]),
            "tools: {0}".format(stats["tools"]),
            "hard negatives: {0}".format(stats["hard_negatives"]),
            "languages: {0}".format(stats["languages"]),
        ]
    )


def lexical_overlap_score(query: str, tool: ToolDefinition) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    overlap = query_tokens & set(tokenize(tool_text(tool)))
    return len(overlap) / len(query_tokens)


def metadata_richness_score(tool: ToolDefinition) -> float:
    richness = len(tool.args) + len(tool.examples) + len(tool.tags)
    if tool.category:
        richness += 1
    if tool.dependencies:
        richness += 1
    return min(richness / 8.0, 1.0)


def deterministic_random(query: str, seed: int = 7) -> random.Random:
    digest = hashlib.sha256("{0}:{1}".format(seed, query).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def random_search(
    query: str,
    tools: Sequence[ToolDefinition],
    top_k: int,
    seed: int = 7,
) -> List[RetrievalResult]:
    ordered = list(tools)
    rng = deterministic_random(query, seed=seed)
    rng.shuffle(ordered)
    selected = ordered[: min(top_k, len(ordered))]
    return [
        RetrievalResult(
            tool=tool,
            score=0.0,
            rank=rank,
            source="random",
        )
        for rank, tool in enumerate(selected, start=1)
    ]


def baseline_search(
    query: str, tools: Sequence[ToolDefinition], top_k: int
) -> List[RetrievalResult]:
    return semantic_search_results(query, tools=tools, top_k=top_k)


def semantic_search_results(
    query: str,
    tools: Sequence[ToolDefinition],
    top_k: int,
) -> List[RetrievalResult]:
    lowered_query = query.lower()
    query_tokens = set(tokenize(query))
    matches: List[RetrievalResult] = []

    for tool in tools:
        text = tool_text(tool)
        lowered_text = text.lower()
        tool_tokens = set(tokenize(text))
        overlap = len(query_tokens & tool_tokens)
        substring_hits = sum(
            1 for token in query_tokens if token and token in lowered_text
        )
        name_hits = sum(
            1 for token in query_tokens if token and token in tool.name.lower()
        )
        category_bonus = (
            1.0 if tool.category and tool.category in lowered_query else 0.0
        )
        score = (
            overlap * 2.0
            + substring_hits * 0.25
            + name_hits * 0.5
            + category_bonus
        )
        matches.append(
            RetrievalResult(
                tool=tool,
                score=score,
                rank=0,
                source="semantic",
            )
        )

    matches.sort(key=lambda item: (item.score, item.tool.name), reverse=True)
    return [
        RetrievalResult(
            tool=result.tool,
            score=result.score,
            rank=rank,
            source=result.source,
        )
        for rank, result in enumerate(matches[:top_k], start=1)
    ]


def rerank_without_feedback(
    query: str,
    results: Sequence[RetrievalResult],
) -> List[RetrievalResult]:
    reranked: List[RetrievalResult] = []
    for result in results:
        lexical = lexical_overlap_score(query, result.tool)
        metadata = metadata_richness_score(result.tool)
        combined_score = (result.score * 0.8) + (lexical * 0.15) + (metadata * 0.05)
        reranked.append(
            RetrievalResult(
                tool=result.tool,
                score=combined_score,
                rank=result.rank,
                source="semantic_rerank",
            )
        )

    reranked.sort(key=lambda item: (item.score, item.tool.name), reverse=True)
    return [
        RetrievalResult(
            tool=result.tool,
            score=result.score,
            rank=rank,
            source=result.source,
        )
        for rank, result in enumerate(reranked, start=1)
    ]


def semantic_rerank_search(
    query: str,
    runtime: EvaluationRuntime,
    top_k: int,
    candidate_tools: Optional[Sequence[ToolDefinition]] = None,
) -> List[RetrievalResult]:
    if candidate_tools is None:
        candidate_limit = min(max(top_k * 3, top_k), len(runtime.retriever._tools))
        base_results = runtime.retriever.search(query, top_k=candidate_limit)
    else:
        candidate_limit = min(max(top_k * 3, top_k), len(candidate_tools))
        base_results = runtime.retriever.search_candidates(
            query,
            candidate_tools=candidate_tools,
            top_k=candidate_limit,
        )
    return rerank_without_feedback(query, base_results)[:top_k]


def build_runtime(
    tool_dataset: Mapping[str, object],
    feedback_queries: Optional[Sequence[Mapping[str, object]]] = None,
    skills_path: Path = DEFAULT_SKILLS_DATASET,
    feedback_top_k: int = 5,
    warmup_passes: int = 8,
    warmup_seed: int = 7,
) -> EvaluationRuntime:
    registry = build_registry(tool_dataset)
    retriever = ToolRetriever(registry=registry)
    retriever.fit()

    skill_registry = None
    skill_retriever = None
    if skills_path.exists():
        skill_registry = SkillRegistry.from_json(skills_path)
        skill_retriever = SkillRetriever(registry=skill_registry)
        skill_retriever.fit()

    tempdir = tempfile.TemporaryDirectory()
    feedback_path = Path(tempdir.name) / "tool_feedback.json"
    feedback_store = ExecutionFeedbackStore(feedback_path)

    runtime = EvaluationRuntime(
        registry=registry,
        retriever=retriever,
        tools_by_name={tool.name: tool for tool in registry.list_tools()},
        tool_graph=ToolGraph.from_registry(registry),
        skill_registry=skill_registry,
        skill_retriever=skill_retriever,
        feedback_store=feedback_store,
        _tempdir=tempdir,
    )
    if feedback_queries:
        seed_execution_feedback(
            runtime,
            query_records=feedback_queries,
            top_k=feedback_top_k,
            warmup_passes=warmup_passes,
            seed=warmup_seed,
        )
    return runtime


def seed_execution_feedback(
    runtime: EvaluationRuntime,
    query_records: Sequence[Mapping[str, object]],
    top_k: int = 5,
    warmup_passes: int = 8,
    seed: int = 7,
) -> None:
    runtime.feedback_store.clear()
    simulator = ToolExecutionSimulator(seed=seed)
    pending = []

    for _ in range(warmup_passes):
        for record in query_records:
            query = str(record["query"])
            hits = runtime.retriever.search(query, top_k=top_k)
            query_embedding = runtime.retriever.encoder.encode_query(query)
            for hit in hits:
                simulation = simulator.simulate(hit.tool.name)
                pending.append(
                    {
                        "query_embedding": query_embedding,
                        "tool": hit.tool.name,
                        "success": simulation.success,
                        "latency_ms": simulation.latency_ms,
                        "error": simulation.error,
                    }
                )

    if pending:
        runtime.feedback_store.record_executions(pending)


def toolscout_search(
    query: str,
    runtime: EvaluationRuntime,
    top_k: int,
    candidate_tools: Optional[Sequence[ToolDefinition]] = None,
) -> List[RetrievalResult]:
    if candidate_tools is None:
        if runtime.skill_retriever is not None:
            routed = runtime.skill_retriever.route(
                query,
                tool_registry=runtime.registry,
                tool_retriever=runtime.retriever,
                tool_graph=runtime.tool_graph,
                top_k_tools=top_k,
                graph_aware=True,
                execution_aware=True,
                feedback_store=runtime.feedback_store,
            )
            return routed.tools[:top_k]
        return runtime.retriever.search(
            query,
            top_k=top_k,
            execution_aware=True,
            feedback_store=runtime.feedback_store,
        )

    candidate_map = {tool.name: tool for tool in candidate_tools}
    search_space = list(candidate_tools)

    if runtime.skill_retriever is not None:
        skill_hits = runtime.skill_retriever.search(query, top_k=1)
        if skill_hits:
            expanded = runtime.tool_graph.expand_dependencies(skill_hits[0].skill.tools)
            prioritized_names = [name for name in expanded if name in candidate_map]
            if prioritized_names:
                prioritized_tools = [candidate_map[name] for name in prioritized_names]
                hits = runtime.retriever.search_candidates(
                    query,
                    candidate_tools=prioritized_tools,
                    top_k=min(max(top_k * 2, top_k), len(prioritized_tools)),
                    execution_aware=True,
                    feedback_store=runtime.feedback_store,
                )
                if len(hits) >= top_k:
                    return hits[:top_k]
                seen = {hit.tool.name for hit in hits}
                remainder = [
                    tool for tool in candidate_tools if tool.name not in seen
                ]
                if remainder:
                    fallback_hits = runtime.retriever.search_candidates(
                        query,
                        candidate_tools=remainder,
                        top_k=min(top_k, len(remainder)),
                        execution_aware=True,
                        feedback_store=runtime.feedback_store,
                    )
                    combined = list(hits)
                    combined.extend(
                        hit for hit in fallback_hits if hit.tool.name not in seen
                    )
                    return combined[:top_k]
                return hits[:top_k]

    return runtime.retriever.search_candidates(
        query,
        candidate_tools=search_space,
        top_k=min(max(top_k * 3, top_k), len(search_space)),
        execution_aware=True,
        feedback_store=runtime.feedback_store,
    )[:top_k]


def search_with_method(
    query: str,
    runtime: EvaluationRuntime,
    top_k: int,
    method: str = "toolscout",
    candidate_tools: Optional[Sequence[ToolDefinition]] = None,
    random_seed: int = 7,
) -> List[RetrievalResult]:
    selected_method = normalize_method(method)
    tools = list(candidate_tools) if candidate_tools is not None else runtime.retriever._tools

    if selected_method == "random":
        return random_search(query, tools=tools, top_k=top_k, seed=random_seed)
    if selected_method == "semantic":
        if candidate_tools is not None:
            return runtime.retriever.search_candidates(
                query,
                candidate_tools=candidate_tools,
                top_k=min(top_k, len(candidate_tools)),
            )
        return runtime.retriever.search(query, top_k=top_k)
    if selected_method == "semantic_rerank":
        return semantic_rerank_search(
            query,
            runtime=runtime,
            top_k=top_k,
            candidate_tools=candidate_tools,
        )
    return toolscout_search(
        query,
        runtime=runtime,
        top_k=top_k,
        candidate_tools=candidate_tools,
    )


def search_with_mode(
    query: str,
    retriever: ToolRetriever,
    top_k: int,
    mode: str = "toolscout",
    candidate_tools: Optional[Sequence[ToolDefinition]] = None,
) -> List[RetrievalResult]:
    runtime = build_runtime(
        {"tools": [tool.to_dict() for tool in retriever._tools]},
        feedback_queries=[],
    )
    try:
        return search_with_method(
            query,
            runtime=runtime,
            top_k=top_k,
            method=mode,
            candidate_tools=candidate_tools,
        )
    finally:
        runtime.cleanup()


def extract_after_markers(query: str, markers: Sequence[str]) -> str:
    lowered = query.lower()
    for marker in markers:
        marker_lower = marker.lower()
        if marker_lower in lowered:
            start = lowered.index(marker_lower) + len(marker_lower)
            return query[start:].strip(" ?.,")
    return ""


def guess_ticker(query: str) -> str:
    uppercase_tokens = re.findall(r"\b[A-Z]{2,5}\b", query)
    if uppercase_tokens:
        return uppercase_tokens[0]
    lowered = query.lower()
    for company, ticker in COMPANY_TICKERS.items():
        if company in lowered:
            return ticker
    return "NVDA"


def guess_language(query: str) -> str:
    lowered = query.lower()
    for token, language in LANGUAGE_NAMES.items():
        if token in lowered:
            return language
    return "Spanish"


def extract_translation_text(query: str) -> str:
    lowered = query.lower()
    if "translate " in lowered and " to " in lowered:
        segment = query[lowered.index("translate ") + len("translate ") :]
        return segment[: segment.lower().index(" to ")].strip(" ?.,")
    if "把 " in query and " 翻译成" in query:
        segment = query.split("把 ", 1)[1]
        return segment.split(" 翻译成", 1)[0].strip(" ?.,")
    return "hello"


def extract_math_expression(query: str) -> str:
    direct = re.findall(r"[-+/*().0-9xX= ]+", query)
    candidates = [item.strip() for item in direct if item.strip()]
    if candidates:
        return max(candidates, key=len)
    lowered = query.lower()
    if "square root of " in lowered:
        start = lowered.index("square root of ") + len("square root of ")
        return query[start:].strip(" ?.,")
    return "1 + 1"


def mock_generate_arguments(query: str, tool: ToolDefinition) -> Dict[str, str]:
    lowered = query.lower()
    arguments: Dict[str, str] = {}

    for arg in tool.args:
        lowered_arg = arg.lower()
        if lowered_arg in {"ticker", "symbol"}:
            arguments[arg] = guess_ticker(query)
            continue
        if lowered_arg in {"city", "location"}:
            value = extract_after_markers(
                query,
                [" in ", " for ", " near ", "在", "查", "看看"],
            )
            arguments[arg] = value or "Tokyo"
            continue
        if lowered_arg == "date":
            if "tomorrow" in lowered or "明天" in query:
                arguments[arg] = "tomorrow"
            elif "today" in lowered or "今天" in query:
                arguments[arg] = "today"
            else:
                arguments[arg] = (
                    extract_after_markers(query, [" on ", " for "]) or "today"
                )
            continue
        if lowered_arg == "target_language":
            arguments[arg] = guess_language(query)
            continue
        if lowered_arg == "text":
            arguments[arg] = extract_translation_text(query)
            continue
        if lowered_arg == "expression":
            arguments[arg] = extract_math_expression(query)
            continue
        if lowered_arg in {"query", "topic"}:
            value = extract_after_markers(
                query, [" about ", " on ", " for ", "查", "搜索"]
            )
            arguments[arg] = value or query.strip(" ?.")
            continue
        arguments[arg] = query.strip(" ?.")

    return arguments


def arguments_are_valid(tool: ToolDefinition, arguments: Mapping[str, str]) -> bool:
    for arg in tool.args:
        value = arguments.get(arg)
        if value is None:
            return False
        if not str(value).strip():
            return False
    return True


def replicate_tool_records(
    tools: Sequence[Mapping[str, object]], target_count: int
) -> List[Dict[str, object]]:
    base_records = [copy.deepcopy(tool) for tool in tools]
    if target_count <= len(base_records):
        return base_records[:target_count]

    expanded = list(base_records)
    index = 0
    while len(expanded) < target_count:
        original = base_records[index % len(base_records)]
        clone = copy.deepcopy(original)
        clone["name"] = "{0}_clone_{1:05d}".format(original["name"], index + 1)
        clone["description"] = "{0} Replica variant {1}.".format(
            original["description"], index + 1
        )
        clone["examples"] = [
            "{0} variant {1}".format(example, index + 1)
            for example in original.get("examples", [])
        ]
        expanded.append(clone)
        index += 1
    return expanded


def mutate_tool_records(
    tools: Sequence[Mapping[str, object]],
    seed: int = 7,
    mutation_rate: float = 0.35,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    mutated = [copy.deepcopy(tool) for tool in tools]

    for index, tool in enumerate(mutated):
        if rng.random() < mutation_rate:
            tool["description"] = (
                "Structured interface for configurable multi-parameter processing."
            )
        if rng.random() < mutation_rate:
            source_args = tool.get("args", tool.get("arguments", []))
            renamed_args = [
                "field_{0}_{1}".format(index, arg_index)
                for arg_index, _ in enumerate(source_args, start=1)
            ]
            tool["arguments"] = renamed_args
            if "args" in tool:
                tool["args"] = list(renamed_args)
    return mutated
