from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

ToolHandler = Callable[..., Any]
PathLike = Union[str, Path]


@dataclass
class ToolDefinition:
    name: str
    description: str
    args: List[str]
    handler: Optional[ToolHandler] = None
    tags: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    category: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)

    def to_embedding_text(self) -> str:
        arguments = ", ".join(self.args) if self.args else "none"
        examples = "; ".join(self.examples) if self.examples else "none"
        parts = [
            "name: {0}".format(self.name),
            "description: {0}".format(self.description),
            "args: {0}".format(arguments),
            "examples: {0}".format(examples),
        ]
        if self.category:
            parts.append("category: {0}".format(self.category))
        if self.dependencies:
            parts.append("dependencies: {0}".format(", ".join(self.dependencies)))
        return " | ".join(parts)

    def to_prompt_text(self) -> str:
        arguments = ", ".join(self.args) if self.args else "none"
        lines = [
            "Tool: {0}".format(self.name),
            "Description: {0}".format(self.description),
            "Arguments: {0}".format(arguments),
        ]
        if self.category:
            lines.append("Category: {0}".format(self.category))
        if self.dependencies:
            lines.append("Dependencies: {0}".format(", ".join(self.dependencies)))
        return "\n".join(lines)

    def to_dict(self, include_handler: bool = False) -> Dict[str, Any]:
        payload = {
            "name": self.name,
            "description": self.description,
            "args": list(self.args),
            "arguments": list(self.args),
            "category": self.category,
            "dependencies": list(self.dependencies),
            "tags": list(self.tags),
            "examples": list(self.examples),
            "metadata": dict(self.metadata),
        }
        if include_handler:
            payload["handler"] = self.handler
        return payload


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        args: Iterable[str],
        handler: Optional[ToolHandler] = None,
        tags: Optional[Iterable[str]] = None,
        examples: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        replace: bool = False,
        category: Optional[str] = None,
        dependencies: Optional[Iterable[str]] = None,
    ) -> ToolDefinition:
        name = name.strip()
        description = description.strip()
        arg_list = [arg.strip() for arg in args]

        if not name:
            raise ValueError("Tool name cannot be empty.")
        if not description:
            raise ValueError("Tool description cannot be empty.")
        if not replace and name in self._tools:
            raise ValueError("Tool '{0}' is already registered.".format(name))

        tool = ToolDefinition(
            name=name,
            description=description,
            args=arg_list,
            handler=handler,
            category=category.strip() if isinstance(category, str) and category.strip() else None,
            dependencies=[dependency.strip() for dependency in dependencies or []],
            tags=list(tags or []),
            examples=list(examples or []),
            metadata=dict(metadata or {}),
        )
        self._tools[name] = tool
        return tool

    def get_tool(self, name: str) -> ToolDefinition:
        if name not in self._tools:
            raise KeyError("Unknown tool '{0}'.".format(name))
        return self._tools[name]

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> List[ToolDefinition]:
        return list(self._tools.values())

    def remove_tool(self, name: str) -> ToolDefinition:
        if name not in self._tools:
            raise KeyError("Unknown tool '{0}'.".format(name))
        return self._tools.pop(name)

    def clear(self) -> None:
        self._tools.clear()

    def format_tools_for_prompt(
        self, tools: Optional[Iterable[ToolDefinition]] = None
    ) -> str:
        selected = list(tools) if tools is not None else self.list_tools()
        return "\n\n".join(tool.to_prompt_text() for tool in selected)

    def extend_from_json(self, path: PathLike) -> List[ToolDefinition]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        records = payload["tools"] if isinstance(payload, dict) else payload
        registered: List[ToolDefinition] = []

        for record in records:
            registered.append(
                self.register_tool(
                    name=record["name"],
                    description=record["description"],
                    args=record.get("args", record.get("arguments", [])),
                    category=record.get("category"),
                    dependencies=record.get("dependencies", []),
                    tags=record.get("tags", []),
                    examples=record.get("examples", []),
                    metadata=record.get("metadata", {}),
                )
            )

        return registered

    @classmethod
    def from_json(cls, path: PathLike) -> "ToolRegistry":
        registry = cls()
        registry.extend_from_json(path)
        return registry
