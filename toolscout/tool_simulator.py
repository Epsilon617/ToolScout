from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class SimulationResult:
    tool_name: str
    success: bool
    latency_ms: float
    error: Optional[str]


DEFAULT_PROFILES: Dict[str, Dict[str, object]] = {
    "weather": {"success_rate": 0.95, "latency_range": (80.0, 140.0)},
    "finance": {"success_rate": 0.80, "latency_range": (120.0, 240.0)},
    "news": {"success_rate": 0.88, "latency_range": (140.0, 260.0)},
    "maps": {"success_rate": 0.90, "latency_range": (90.0, 180.0)},
    "math": {"success_rate": 0.98, "latency_range": (20.0, 60.0)},
    "code": {"success_rate": 0.85, "latency_range": (100.0, 220.0)},
    "translation": {"success_rate": 0.92, "latency_range": (70.0, 150.0)},
    "search": {"success_rate": 0.87, "latency_range": (110.0, 210.0)},
    "social": {"success_rate": 0.84, "latency_range": (130.0, 230.0)},
    "calendar": {"success_rate": 0.96, "latency_range": (40.0, 90.0)},
    "default": {"success_rate": 0.89, "latency_range": (90.0, 180.0)},
}

TOOL_PROFILES: Dict[str, Dict[str, object]] = {
    "weather_api": {"success_rate": 0.97, "latency_range": (70.0, 110.0)},
    "finance_api": {"success_rate": 0.82, "latency_range": (110.0, 190.0)},
    "news_api": {"success_rate": 0.84, "latency_range": (170.0, 280.0)},
    "news_search": {"success_rate": 0.94, "latency_range": (120.0, 190.0)},
    "news_summary": {"success_rate": 0.91, "latency_range": (130.0, 200.0)},
    "maps_places_api": {"success_rate": 0.95, "latency_range": (80.0, 120.0)},
    "math_solver": {"success_rate": 0.99, "latency_range": (20.0, 40.0)},
    "code_search": {"success_rate": 0.90, "latency_range": (90.0, 160.0)},
    "translation_service": {"success_rate": 0.96, "latency_range": (60.0, 110.0)},
    "web_search": {"success_rate": 0.90, "latency_range": (100.0, 160.0)},
    "social_search": {"success_rate": 0.86, "latency_range": (120.0, 190.0)},
    "calendar_lookup": {"success_rate": 0.98, "latency_range": (35.0, 65.0)},
}


class ToolExecutionSimulator:
    def __init__(self, seed: int = 7) -> None:
        self._random = random.Random(seed)

    def simulate(self, tool_name: str) -> SimulationResult:
        profile = TOOL_PROFILES.get(tool_name)
        if profile is None:
            category = self._infer_category(tool_name)
            profile = DEFAULT_PROFILES.get(category, DEFAULT_PROFILES["default"])
        success_rate = float(profile["success_rate"])
        latency_min, latency_max = profile["latency_range"]  # type: ignore[misc]
        success = self._random.random() < success_rate
        latency_ms = round(self._random.uniform(latency_min, latency_max), 3)
        error = None if success else "{0} simulated failure".format(tool_name)
        return SimulationResult(
            tool_name=tool_name,
            success=success,
            latency_ms=latency_ms,
            error=error,
        )

    def _infer_category(self, tool_name: str) -> str:
        lowered = tool_name.lower()
        if lowered.startswith("weather"):
            return "weather"
        if lowered.startswith("finance"):
            return "finance"
        if lowered.startswith("news") or lowered.startswith("headline"):
            return "news"
        if lowered.startswith("maps") or lowered.startswith("places"):
            return "maps"
        if lowered.startswith("math"):
            return "math"
        if lowered.startswith("code"):
            return "code"
        if lowered.startswith("translation") or lowered.startswith("lingua"):
            return "translation"
        if lowered.startswith("web") or lowered.endswith("_search"):
            return "search"
        if lowered.startswith("social"):
            return "social"
        if lowered.startswith("calendar"):
            return "calendar"
        return "default"
