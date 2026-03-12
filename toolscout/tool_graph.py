from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Set

from .registry.tool_registry import ToolDefinition, ToolRegistry


class ToolGraph:
    def __init__(self) -> None:
        self._children: Dict[str, List[str]] = {}
        self._parents: Dict[str, List[str]] = {}

    def add_tool(
        self, tool_name: str, dependencies: Optional[Iterable[str]] = None
    ) -> None:
        self._children.setdefault(tool_name, [])
        self._parents.setdefault(tool_name, [])

        for dependency in dependencies or []:
            self.add_dependency(tool_name, dependency)

    def add_dependency(self, parent_tool: str, dependency_tool: str) -> None:
        self._children.setdefault(parent_tool, [])
        self._parents.setdefault(parent_tool, [])
        self._children.setdefault(dependency_tool, [])
        self._parents.setdefault(dependency_tool, [])

        if dependency_tool not in self._children[parent_tool]:
            self._children[parent_tool].append(dependency_tool)
        if parent_tool not in self._parents[dependency_tool]:
            self._parents[dependency_tool].append(parent_tool)

    def children(self, tool_name: str) -> List[str]:
        return list(self._children.get(tool_name, []))

    def parents(self, tool_name: str) -> List[str]:
        return list(self._parents.get(tool_name, []))

    def dependencies(self, tool_name: str) -> List[str]:
        return self.children(tool_name)

    def dependents(self, tool_name: str) -> List[str]:
        return self.parents(tool_name)

    def traverse(
        self,
        start_tools: Sequence[str],
        direction: str = "downstream",
        max_depth: Optional[int] = None,
    ) -> List[str]:
        if direction not in {"downstream", "upstream", "both"}:
            raise ValueError(
                "direction must be one of 'downstream', 'upstream', or 'both'."
            )

        queue = deque((tool_name, 0) for tool_name in start_tools)
        visited: Set[str] = set()
        ordered: List[str] = []

        while queue:
            tool_name, depth = queue.popleft()
            if tool_name in visited:
                continue

            visited.add(tool_name)
            ordered.append(tool_name)

            if max_depth is not None and depth >= max_depth:
                continue

            neighbors: List[str] = []
            if direction in {"downstream", "both"}:
                neighbors.extend(self.children(tool_name))
            if direction in {"upstream", "both"}:
                neighbors.extend(self.parents(tool_name))

            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))

        return ordered

    def expand_dependencies(
        self, tool_names: Sequence[str], max_depth: Optional[int] = None
    ) -> List[str]:
        seeds = list(dict.fromkeys(tool_names))
        expanded = self.traverse(
            seeds, direction="downstream", max_depth=max_depth
        )
        return expanded

    def cluster_for(self, tool_name: str) -> List[str]:
        return self.traverse([tool_name], direction="both")

    def clusters(self) -> List[List[str]]:
        unseen = set(self._children) | set(self._parents)
        clusters: List[List[str]] = []

        while unseen:
            seed = next(iter(unseen))
            cluster = self.cluster_for(seed)
            clusters.append(cluster)
            unseen.difference_update(cluster)

        clusters.sort(key=lambda cluster: (-len(cluster), cluster[0] if cluster else ""))
        return clusters

    @classmethod
    def from_tools(cls, tools: Sequence[ToolDefinition]) -> "ToolGraph":
        graph = cls()
        for tool in tools:
            graph.add_tool(tool.name, dependencies=tool.dependencies)
        return graph

    @classmethod
    def from_registry(cls, registry: ToolRegistry) -> "ToolGraph":
        return cls.from_tools(registry.list_tools())
