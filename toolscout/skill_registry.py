from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

PathLike = Union[str, Path]


@dataclass
class SkillDefinition:
    name: str
    description: str
    tools: List[str]
    examples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_embedding_text(self) -> str:
        tool_names = ", ".join(self.tools) if self.tools else "none"
        examples = "; ".join(self.examples) if self.examples else "none"
        return " | ".join(
            [
                "skill: {0}".format(self.name),
                "description: {0}".format(self.description),
                "tools: {0}".format(tool_names),
                "examples: {0}".format(examples),
            ]
        )


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: Dict[str, SkillDefinition] = {}

    def register_skill(
        self,
        name: str,
        description: str,
        tools: Iterable[str],
        examples: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        replace: bool = False,
    ) -> SkillDefinition:
        skill_name = name.strip()
        skill_description = description.strip()
        tool_names = [tool.strip() for tool in tools]

        if not skill_name:
            raise ValueError("Skill name cannot be empty.")
        if not skill_description:
            raise ValueError("Skill description cannot be empty.")
        if not replace and skill_name in self._skills:
            raise ValueError("Skill '{0}' is already registered.".format(skill_name))

        skill = SkillDefinition(
            name=skill_name,
            description=skill_description,
            tools=tool_names,
            examples=list(examples or []),
            metadata=dict(metadata or {}),
        )
        self._skills[skill_name] = skill
        return skill

    def get_skill(self, name: str) -> SkillDefinition:
        if name not in self._skills:
            raise KeyError("Unknown skill '{0}'.".format(name))
        return self._skills[name]

    def list_skills(self) -> List[SkillDefinition]:
        return list(self._skills.values())

    def extend_from_json(self, path: PathLike) -> List[SkillDefinition]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        records = payload["skills"] if isinstance(payload, dict) else payload
        loaded: List[SkillDefinition] = []
        for record in records:
            loaded.append(
                self.register_skill(
                    name=record["name"],
                    description=record["description"],
                    tools=record.get("tools", []),
                    examples=record.get("examples", []),
                    metadata=record.get("metadata", {}),
                )
            )
        return loaded

    @classmethod
    def from_json(cls, path: PathLike) -> "SkillRegistry":
        registry = cls()
        registry.extend_from_json(path)
        return registry
