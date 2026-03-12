from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None


@dataclass(frozen=True)
class IndexMatch:
    index: int
    score: float


class ToolIndex:
    def __init__(self, use_faiss: bool = True) -> None:
        self.use_faiss = use_faiss
        self._backend: Optional[str] = None
        self._faiss_index = None
        self._embeddings: Optional[np.ndarray] = None
        self._dimension: Optional[int] = None
        self._size = 0

    @property
    def backend(self) -> str:
        if self._backend is None:
            return "uninitialized"
        return self._backend

    @property
    def ntotal(self) -> int:
        return self._size

    def build(self, embeddings: np.ndarray) -> None:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("Embeddings must be a 2D matrix.")
        if matrix.shape[0] == 0:
            raise ValueError("Cannot build an index with zero embeddings.")

        matrix = self._normalize(matrix)
        self._dimension = int(matrix.shape[1])
        self._size = int(matrix.shape[0])

        if self.use_faiss and faiss is not None:
            index = faiss.IndexFlatIP(self._dimension)
            index.add(matrix)
            self._faiss_index = index
            self._embeddings = None
            self._backend = "faiss"
            return

        self._embeddings = matrix
        self._faiss_index = None
        self._backend = "numpy"

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[IndexMatch]:
        if self._size == 0 or self._dimension is None:
            raise RuntimeError("Index is empty. Call build() before search().")
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self._dimension:
            raise ValueError(
                "Query embedding dimension {0} does not match index dimension "
                "{1}.".format(query.shape[1], self._dimension)
            )

        query = self._normalize(query)
        limit = min(top_k, self._size)

        if self._backend == "faiss":
            scores, indices = self._faiss_index.search(query, limit)
            return [
                IndexMatch(index=int(idx), score=float(score))
                for idx, score in zip(indices[0], scores[0])
                if idx >= 0
            ]

        scores = np.dot(self._embeddings, query[0])
        ranking = np.argsort(-scores)[:limit]
        return [
            IndexMatch(index=int(index), score=float(scores[index]))
            for index in ranking
        ]

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vectors / norms

