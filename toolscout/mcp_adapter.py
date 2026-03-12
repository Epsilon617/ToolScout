from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from .registry.tool_registry import ToolRegistry

PathLike = Union[str, Path]


def mcp_tool_to_toolscout(mcp_tool: Dict[str, Any]) -> Dict[str, Any]:
    input_schema = mcp_tool.get("input_schema", {})
    properties = input_schema.get("properties", {})
    arguments = list(properties.keys())
    return {
        "name": mcp_tool["name"],
        "description": mcp_tool.get("description", ""),
        "arguments": arguments,
        "category": mcp_tool.get("category"),
        "examples": mcp_tool.get("examples", []),
        "dependencies": mcp_tool.get("dependencies", []),
    }


def load_mcp_tools(path: PathLike) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload["tools"] if isinstance(payload, dict) else payload
    return [mcp_tool_to_toolscout(record) for record in records]


def load_mcp_registry(path: PathLike) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in load_mcp_tools(path):
        registry.register_tool(
            name=tool["name"],
            description=tool["description"],
            args=tool.get("arguments", []),
            category=tool.get("category"),
            dependencies=tool.get("dependencies", []),
            examples=tool.get("examples", []),
        )
    return registry
