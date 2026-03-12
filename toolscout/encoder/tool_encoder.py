from __future__ import annotations

import re
import warnings
import zlib
from typing import Iterable, List, Optional, Sequence

import numpy as np

from ..registry.tool_registry import ToolDefinition

_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


class ToolEncoder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        backend: str = "auto",
        fallback_dimensions: int = 384,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.backend = backend
        self.fallback_dimensions = fallback_dimensions
        self.device = device
        self._model = None
        self._resolved_backend: Optional[str] = None

    @property
    def resolved_backend(self) -> str:
        if self._resolved_backend is None:
            self._resolve_backend()
        return self._resolved_backend or "keyword"

    def _resolve_backend(self) -> None:
        if self.backend == "keyword":
            self._resolved_backend = "keyword"
            return

        if self.backend not in ("auto", "sentence-transformers"):
            raise ValueError(
                "Unsupported encoder backend '{0}'. Use 'auto', "
                "'sentence-transformers', or 'keyword'.".format(self.backend)
            )

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            if self.backend == "sentence-transformers":
                raise ImportError(
                    "sentence-transformers is required for the selected backend."
                ) from exc
            warnings.warn(
                "sentence-transformers is not installed. Falling back to the "
                "deterministic keyword encoder.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._resolved_backend = "keyword"
            return

        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._resolved_backend = "sentence-transformers"

    def render_tool(self, tool: ToolDefinition) -> str:
        return tool.to_embedding_text()

    def encode_tools(self, tools: Sequence[ToolDefinition]) -> np.ndarray:
        return self.encode_texts([self.render_tool(tool) for tool in tools])

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode_texts([query])[0]

    def encode_queries(self, queries: Iterable[str]) -> np.ndarray:
        return self.encode_texts(list(queries))

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.fallback_dimensions), dtype=np.float32)

        if self.resolved_backend == "sentence-transformers":
            vectors = self._model.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(vectors, dtype=np.float32)

        return self._encode_with_keyword_backend(texts)

    def _encode_with_keyword_backend(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros(
            (len(texts), self.fallback_dimensions), dtype=np.float32
        )

        for row_index, text in enumerate(texts):
            tokens = self._tokenize(text)
            for token in tokens:
                vectors[row_index, self._bucket(token)] += 1.0
            for left, right in zip(tokens, tokens[1:]):
                vectors[row_index, self._bucket(left + "__" + right)] += 0.5

        return self._normalize(vectors)

    def _tokenize(self, text: str) -> List[str]:
        return _TOKEN_PATTERN.findall(text.lower())

    def _bucket(self, token: str) -> int:
        return zlib.crc32(token.encode("utf-8")) % self.fallback_dimensions

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vectors / norms

