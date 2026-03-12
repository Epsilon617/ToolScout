from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from ..encoder.tool_encoder import ToolEncoder
from ..execution_feedback import ExecutionFeedbackStore
from ..index.tool_index import ToolIndex
from ..registry.tool_registry import ToolDefinition, ToolRegistry
from ..tool_graph import ToolGraph


@dataclass(frozen=True)
class RetrievalResult:
    tool: ToolDefinition
    score: float
    rank: int
    source: str = "retrieved"


class ToolRetriever:
    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        encoder: Optional[ToolEncoder] = None,
        index: Optional[ToolIndex] = None,
    ) -> None:
        self.registry = registry
        self.encoder = encoder or ToolEncoder()
        self.index = index or ToolIndex()
        self._tools: List[ToolDefinition] = []

    @property
    def backend_summary(self) -> str:
        return "{0} encoder + {1} index".format(
            self.encoder.resolved_backend, self.index.backend
        )

    def fit(
        self, tools: Optional[Sequence[ToolDefinition]] = None
    ) -> "ToolRetriever":
        selected_tools = list(tools) if tools is not None else self._registry_tools()
        if not selected_tools:
            raise ValueError("No tools available to index.")

        embeddings = self.encoder.encode_tools(selected_tools)
        self.index.build(embeddings)
        self._tools = selected_tools
        return self

    def search(
        self,
        query: str,
        top_k: int = 5,
        execution_aware: bool = False,
        feedback_store: Optional[ExecutionFeedbackStore] = None,
    ) -> List[RetrievalResult]:
        if not self._tools:
            self.fit()

        query_embedding = self.encoder.encode_query(query)
        return self._search_from_index(
            query_embedding=query_embedding,
            tools=self._tools,
            index=self.index,
            top_k=top_k,
            execution_aware=execution_aware,
            feedback_store=feedback_store,
        )

    def search_candidates(
        self,
        query: str,
        candidate_tools: Sequence[ToolDefinition],
        top_k: int = 5,
        execution_aware: bool = False,
        feedback_store: Optional[ExecutionFeedbackStore] = None,
    ) -> List[RetrievalResult]:
        selected_tools = list(candidate_tools)
        if not selected_tools:
            return []

        candidate_index = ToolIndex(use_faiss=self.index.use_faiss)
        candidate_embeddings = self.encoder.encode_tools(selected_tools)
        candidate_index.build(candidate_embeddings)

        query_embedding = self.encoder.encode_query(query)
        return self._search_from_index(
            query_embedding=query_embedding,
            tools=selected_tools,
            index=candidate_index,
            top_k=top_k,
            execution_aware=execution_aware,
            feedback_store=feedback_store,
        )

    def search_graph_aware(
        self,
        query: str,
        top_k: int = 5,
        tool_graph: Optional[ToolGraph] = None,
        seed_tools: Optional[Sequence[ToolDefinition]] = None,
        max_dependency_depth: Optional[int] = None,
        execution_aware: bool = False,
        feedback_store: Optional[ExecutionFeedbackStore] = None,
    ) -> List[RetrievalResult]:
        base_results = (
            self.search_candidates(
                query,
                candidate_tools=seed_tools,
                top_k=top_k,
                execution_aware=execution_aware,
                feedback_store=feedback_store,
            )
            if seed_tools is not None
            else self.search(
                query,
                top_k=top_k,
                execution_aware=execution_aware,
                feedback_store=feedback_store,
            )
        )
        if tool_graph is None:
            return base_results

        expanded_names = tool_graph.expand_dependencies(
            [result.tool.name for result in base_results],
            max_depth=max_dependency_depth,
        )
        ordered_results = list(base_results)
        seen_names = {result.tool.name for result in ordered_results}

        for tool_name in expanded_names:
            if tool_name in seen_names:
                continue
            dependency_tool = self._lookup_tool(tool_name)
            if dependency_tool is None:
                continue
            ordered_results.append(
                RetrievalResult(
                    tool=dependency_tool,
                    score=0.0,
                    rank=len(ordered_results) + 1,
                    source="dependency",
                )
            )
            seen_names.add(tool_name)

        return ordered_results

    def retrieve_names(
        self,
        query: str,
        top_k: int = 5,
        execution_aware: bool = False,
        feedback_store: Optional[ExecutionFeedbackStore] = None,
    ) -> List[str]:
        return [
            result.tool.name
            for result in self.search(
                query,
                top_k=top_k,
                execution_aware=execution_aware,
                feedback_store=feedback_store,
            )
        ]

    def _registry_tools(self) -> List[ToolDefinition]:
        if self.registry is None:
            raise ValueError("No registry was provided to ToolRetriever.")
        return self.registry.list_tools()

    def _lookup_tool(self, tool_name: str) -> Optional[ToolDefinition]:
        if self.registry is not None and self.registry.has_tool(tool_name):
            return self.registry.get_tool(tool_name)
        for tool in self._tools:
            if tool.name == tool_name:
                return tool
        return None

    def _search_from_index(
        self,
        query_embedding,
        tools: Sequence[ToolDefinition],
        index: ToolIndex,
        top_k: int,
        execution_aware: bool = False,
        feedback_store: Optional[ExecutionFeedbackStore] = None,
    ) -> List[RetrievalResult]:
        candidate_limit = top_k
        if execution_aware:
            candidate_limit = min(max(top_k * 3, top_k), len(tools))

        matches = index.search(query_embedding, top_k=candidate_limit)
        base_results = [
            RetrievalResult(
                tool=tools[match.index],
                score=match.score,
                rank=rank,
                source="retrieved",
            )
            for rank, match in enumerate(matches, start=1)
        ]
        if not execution_aware:
            return base_results[:top_k]

        store = feedback_store or ExecutionFeedbackStore()
        reranked = self._rerank_with_execution_feedback(base_results, store)
        return reranked[:top_k]

    def _rerank_with_execution_feedback(
        self,
        results: Sequence[RetrievalResult],
        feedback_store: ExecutionFeedbackStore,
    ) -> List[RetrievalResult]:
        if not results:
            return []

        tool_stats = feedback_store.tool_stats_map(
            [result.tool.name for result in results]
        )
        average_latencies = {
            result.tool.name: tool_stats.get(result.tool.name, {}).get(
                "average_latency_ms"
            )
            for result in results
        }
        known_latencies = [
            latency
            for latency in average_latencies.values()
            if latency is not None
        ]
        fallback_latency = (
            statistics.mean(known_latencies)
            if known_latencies
            else feedback_store.global_average_latency() or 150.0
        )

        inverse_latencies = {}
        for result in results:
            latency = average_latencies[result.tool.name]
            if latency is None:
                latency = fallback_latency
            inverse_latencies[result.tool.name] = 1.0 / max(latency, 1.0)

        min_inverse = min(inverse_latencies.values())
        max_inverse = max(inverse_latencies.values())

        reranked: List[RetrievalResult] = []
        for result in results:
            success_rate = tool_stats.get(result.tool.name, {}).get(
                "success_rate", 0.5
            )
            inverse_latency = inverse_latencies[result.tool.name]
            if max_inverse == min_inverse:
                latency_score = 1.0
            else:
                latency_score = (inverse_latency - min_inverse) / (
                    max_inverse - min_inverse
                )
            combined_score = (
                (result.score * 0.7)
                + (success_rate * 0.2)
                + (latency_score * 0.1)
            )
            reranked.append(
                RetrievalResult(
                    tool=result.tool,
                    score=combined_score,
                    rank=result.rank,
                    source=result.source,
                )
            )

        reranked.sort(key=lambda item: item.score, reverse=True)
        return [
            RetrievalResult(
                tool=result.tool,
                score=result.score,
                rank=rank,
                source=result.source,
            )
            for rank, result in enumerate(reranked, start=1)
        ]
