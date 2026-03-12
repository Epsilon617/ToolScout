from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from ..encoder.tool_encoder import ToolEncoder
from ..index.tool_index import ToolIndex
from ..registry.tool_registry import ToolDefinition, ToolRegistry


@dataclass(frozen=True)
class RetrievalResult:
    tool: ToolDefinition
    score: float
    rank: int


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

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        if not self._tools:
            self.fit()

        query_embedding = self.encoder.encode_query(query)
        matches = self.index.search(query_embedding, top_k=top_k)
        return [
            RetrievalResult(
                tool=self._tools[match.index],
                score=match.score,
                rank=rank,
            )
            for rank, match in enumerate(matches, start=1)
        ]

    def retrieve_names(self, query: str, top_k: int = 5) -> List[str]:
        return [result.tool.name for result in self.search(query, top_k=top_k)]

    def _registry_tools(self) -> List[ToolDefinition]:
        if self.registry is None:
            raise ValueError("No registry was provided to ToolRetriever.")
        return self.registry.list_tools()

