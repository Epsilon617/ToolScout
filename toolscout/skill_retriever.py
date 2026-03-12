from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .encoder.tool_encoder import ToolEncoder
from .execution_feedback import ExecutionFeedbackStore
from .index.tool_index import ToolIndex
from .registry.tool_registry import ToolRegistry
from .retriever.tool_retriever import RetrievalResult, ToolRetriever
from .skill_registry import SkillDefinition, SkillRegistry
from .tool_graph import ToolGraph


@dataclass(frozen=True)
class SkillRetrievalResult:
    skill: SkillDefinition
    score: float
    rank: int


@dataclass(frozen=True)
class SkillRoutingResult:
    skill: SkillDefinition
    tools: List[RetrievalResult]


class SkillRetriever:
    def __init__(
        self,
        registry: Optional[SkillRegistry] = None,
        encoder: Optional[ToolEncoder] = None,
        index: Optional[ToolIndex] = None,
    ) -> None:
        self.registry = registry
        self.encoder = encoder or ToolEncoder()
        self.index = index or ToolIndex()
        self._skills: List[SkillDefinition] = []

    def fit(
        self, skills: Optional[Sequence[SkillDefinition]] = None
    ) -> "SkillRetriever":
        selected_skills = list(skills) if skills is not None else self._registry_skills()
        if not selected_skills:
            raise ValueError("No skills available to index.")

        embeddings = self.encoder.encode_texts(
            [skill.to_embedding_text() for skill in selected_skills]
        )
        self.index.build(embeddings)
        self._skills = selected_skills
        return self

    def search(self, query: str, top_k: int = 3) -> List[SkillRetrievalResult]:
        if not self._skills:
            self.fit()

        query_embedding = self.encoder.encode_query(query)
        matches = self.index.search(query_embedding, top_k=top_k)
        return [
            SkillRetrievalResult(
                skill=self._skills[match.index],
                score=match.score,
                rank=rank,
            )
            for rank, match in enumerate(matches, start=1)
        ]

    def route(
        self,
        query: str,
        tool_registry: ToolRegistry,
        tool_retriever: Optional[ToolRetriever] = None,
        tool_graph: Optional[ToolGraph] = None,
        top_k_skills: int = 1,
        top_k_tools: int = 5,
        graph_aware: bool = True,
        execution_aware: bool = False,
        feedback_store: Optional[ExecutionFeedbackStore] = None,
    ) -> SkillRoutingResult:
        skill_hits = self.search(query, top_k=top_k_skills)
        if not skill_hits:
            raise ValueError("No skills were retrieved for the query.")

        selected_skill = skill_hits[0].skill
        retriever = tool_retriever or ToolRetriever(registry=tool_registry)
        candidate_tools = [
            tool_registry.get_tool(tool_name)
            for tool_name in selected_skill.tools
            if tool_registry.has_tool(tool_name)
        ]
        if not candidate_tools:
            raise ValueError(
                "Skill '{0}' does not reference any registered tools.".format(
                    selected_skill.name
                )
            )

        if graph_aware:
            routed_tools = retriever.search_graph_aware(
                query,
                top_k=top_k_tools,
                tool_graph=tool_graph or ToolGraph.from_registry(tool_registry),
                seed_tools=candidate_tools,
                execution_aware=execution_aware,
                feedback_store=feedback_store,
            )
        else:
            routed_tools = retriever.search_candidates(
                query,
                candidate_tools=candidate_tools,
                top_k=top_k_tools,
                execution_aware=execution_aware,
                feedback_store=feedback_store,
            )

        return SkillRoutingResult(skill=selected_skill, tools=routed_tools)

    def _registry_skills(self) -> List[SkillDefinition]:
        if self.registry is None:
            raise ValueError("No registry was provided to SkillRetriever.")
        return self.registry.list_skills()
