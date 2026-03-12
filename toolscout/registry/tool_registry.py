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

    def to_embedding_text(self) -> str:
        arguments = ", ".join(self.args) if self.args else "none"
        tags = ", ".join(self.tags) if self.tags else "none"
        examples = "; ".join(self.examples) if self.examples else "none"
        return "\n".join(
            [
                "tool",
                "name: {0}".format(self.name),
                "description: {0}".format(self.description),
                "arguments: {0}".format(arguments),
                "tags: {0}".format(tags),
                "examples: {0}".format(examples),
            ]
        )

    def to_prompt_text(self) -> str:
        arguments = ", ".join(self.args) if self.args else "none"
        return "\n".join(
            [
                "Tool: {0}".format(self.name),
                "Description: {0}".format(self.description),
                "Arguments: {0}".format(arguments),
            ]
        )

    def to_dict(self, include_handler: bool = False) -> Dict[str, Any]:
        payload = {
            "name": self.name,
            "description": self.description,
            "args": list(self.args),
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
                    args=record.get("args", []),
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

