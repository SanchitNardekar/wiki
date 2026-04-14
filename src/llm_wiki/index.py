"""Embedding index — cosine similarity search over wiki pages."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from llm_wiki.models.embeddings import EmbeddingIndexData, PageEmbedding


class EmbeddingIndex:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._data = self._load()

    def _load(self) -> EmbeddingIndexData:
        if self._path.exists() and self._path.stat().st_size > 0:
            return EmbeddingIndexData.model_validate_json(self._path.read_text())
        return EmbeddingIndexData()

    def save(self) -> None:
        self._path.write_text(self._data.model_dump_json(indent=2))

    def upsert(self, embedding: PageEmbedding) -> None:
        self._data.embeddings[embedding.slug] = embedding

    def top_k(self, query_vector: list[float], k: int = 5) -> list[tuple[str, float]]:
        if not self._data.embeddings:
            return []
        q = np.array(query_vector, dtype=np.float32)
        scores = [
            (slug, float(np.dot(q, emb.as_array())))
            for slug, emb in self._data.embeddings.items()
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def slugs(self) -> list[str]:
        return list(self._data.embeddings.keys())

    def contains(self, slug: str) -> bool:
        return slug in self._data.embeddings

    def remove(self, slug: str) -> None:
        self._data.embeddings.pop(slug, None)

    @property
    def size(self) -> int:
        return self._data.slug_count
