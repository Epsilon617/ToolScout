from __future__ import annotations

from typing import Any, Dict, Mapping

from ..registry.tool_registry import ToolRegistry


class ToolExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def execute(
        self, tool_name: str, arguments: Mapping[str, Any]
    ) -> Any:
        tool = self.registry.get_tool(tool_name)
        arguments_dict: Dict[str, Any] = dict(arguments)

        missing = [arg for arg in tool.args if arg not in arguments_dict]
        if missing:
            raise ValueError(
                "Missing required arguments for '{0}': {1}".format(
                    tool_name, ", ".join(missing)
                )
            )

        unexpected = [
            arg for arg in arguments_dict.keys() if arg not in tool.args
        ]
        if unexpected:
            raise ValueError(
                "Unexpected arguments for '{0}': {1}".format(
                    tool_name, ", ".join(unexpected)
                )
            )

        if tool.handler is None:
            raise ValueError(
                "Tool '{0}' has no execution handler.".format(tool_name)
            )

        ordered_args = {name: arguments_dict[name] for name in tool.args}
        return tool.handler(**ordered_args)

