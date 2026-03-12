from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

PathLike = Union[str, Path]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEEDBACK_STORE = PROJECT_ROOT / "data" / "tool_feedback_store.json"


def _ensure_store_exists(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps({"records": []}, indent=2), encoding="utf-8")


def _serialize_embedding(query_embedding: Sequence[float]) -> List[float]:
    if hasattr(query_embedding, "tolist"):
        values = query_embedding.tolist()
    else:
        values = list(query_embedding)
    return [float(value) for value in values]


class ExecutionFeedbackStore:
    def __init__(self, path: PathLike = DEFAULT_FEEDBACK_STORE) -> None:
        self.path = Path(path)
        self._payload_cache: Optional[Dict[str, Any]] = None
        _ensure_store_exists(self.path)

    def _load_payload(self) -> Dict[str, Any]:
        if self._payload_cache is not None:
            return self._payload_cache
        _ensure_store_exists(self.path)
        raw = self.path.read_text(encoding="utf-8").strip()
        if not raw:
            self._payload_cache = {"records": []}
            return self._payload_cache
        try:
            self._payload_cache = json.loads(raw)
        except json.JSONDecodeError:
            self._payload_cache = {"records": []}
        return self._payload_cache

    def _write_payload(self, payload: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self.path.parent,
            delete=False,
        ) as handle:
            handle.write(json.dumps(payload, indent=2))
            temp_path = Path(handle.name)
        temp_path.replace(self.path)
        self._payload_cache = payload

    def records(self) -> List[Dict[str, Any]]:
        return list(self._load_payload().get("records", []))

    def clear(self) -> None:
        self._write_payload({"records": []})

    def record_execution(
        self,
        query_embedding: Sequence[float],
        tool_name: str,
        success: bool,
        latency_ms: float,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = self._load_payload()
        record = {
            "query_embedding": _serialize_embedding(query_embedding),
            "tool": tool_name,
            "success": bool(success),
            "latency_ms": float(latency_ms),
            "error": error,
        }
        payload.setdefault("records", []).append(record)
        self._write_payload(payload)
        return record

    def record_executions(
        self, records: Iterable[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        payload = self._load_payload()
        serialized_records = []
        for record in records:
            serialized = {
                "query_embedding": _serialize_embedding(record["query_embedding"]),
                "tool": record["tool"],
                "success": bool(record["success"]),
                "latency_ms": float(record["latency_ms"]),
                "error": record.get("error"),
            }
            serialized_records.append(serialized)
        payload.setdefault("records", []).extend(serialized_records)
        self._write_payload(payload)
        return serialized_records

    def records_for_tool(self, tool_name: str) -> List[Dict[str, Any]]:
        return [
            record for record in self.records() if record.get("tool") == tool_name
        ]

    def compute_tool_success_rate(self, tool_name: str) -> float:
        records = self.records_for_tool(tool_name)
        if not records:
            return 0.5
        successes = sum(1 for record in records if record.get("success"))
        return successes / len(records)

    def compute_average_latency(self, tool_name: str) -> Optional[float]:
        records = self.records_for_tool(tool_name)
        if not records:
            return None
        return sum(record["latency_ms"] for record in records) / len(records)

    def global_average_latency(self) -> Optional[float]:
        records = self.records()
        if not records:
            return None
        return sum(record["latency_ms"] for record in records) / len(records)

    def tool_stats_map(
        self, tool_names: Optional[Iterable[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        selected_names = set(tool_names) if tool_names is not None else None
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for record in self.records():
            tool_name = record["tool"]
            if selected_names is not None and tool_name not in selected_names:
                continue
            grouped.setdefault(tool_name, []).append(record)

        stats: Dict[str, Dict[str, float]] = {}
        for name, records in grouped.items():
            success_rate = sum(
                1 for record in records if record.get("success")
            ) / len(records)
            average_latency = sum(
                record["latency_ms"] for record in records
            ) / len(records)
            stats[name] = {
                "count": float(len(records)),
                "success_rate": success_rate,
                "average_latency_ms": average_latency,
            }
        return stats

    def tool_stats(self, tool_name: Optional[str] = None) -> List[Dict[str, Any]]:
        grouped = self.tool_stats_map([tool_name] if tool_name is not None else None)
        stats: List[Dict[str, Any]] = []
        for name, values in grouped.items():
            stats.append(
                {
                    "tool": name,
                    "count": int(values["count"]),
                    "success_rate": values["success_rate"],
                    "average_latency_ms": values["average_latency_ms"],
                }
            )

        stats.sort(key=lambda item: (-item["count"], item["tool"]))
        return stats


def record_execution(
    query_embedding: Sequence[float],
    tool_name: str,
    success: bool,
    latency_ms: float,
    error: Optional[str] = None,
    store_path: PathLike = DEFAULT_FEEDBACK_STORE,
) -> Dict[str, Any]:
    store = ExecutionFeedbackStore(store_path)
    return store.record_execution(
        query_embedding=query_embedding,
        tool_name=tool_name,
        success=success,
        latency_ms=latency_ms,
        error=error,
    )


def compute_tool_success_rate(
    tool_name: str, store_path: PathLike = DEFAULT_FEEDBACK_STORE
) -> float:
    store = ExecutionFeedbackStore(store_path)
    return store.compute_tool_success_rate(tool_name)
