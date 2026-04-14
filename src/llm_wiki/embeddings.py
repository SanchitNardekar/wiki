"""OpenAI embedding wrapper with normalisation and truncation."""

from __future__ import annotations

import numpy as np
from openai import OpenAI

from llm_wiki.models.embeddings import EMBEDDING_MODEL, PageEmbedding

_MAX_INPUT_CHARS = 8191 * 4  # ~8191 tokens at ~4 chars/token


class EmbeddingService:
    def __init__(self, client: OpenAI, model: str = EMBEDDING_MODEL) -> None:
        self._client = client
        self._model = model

    def embed(self, text: str) -> list[float]:
        truncated = text[:_MAX_INPUT_CHARS]
        response = self._client.embeddings.create(model=self._model, input=truncated)
        raw: list[float] = response.data[0].embedding
        arr = np.array(raw, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    def embed_page(self, slug: str, text: str) -> PageEmbedding:
        return PageEmbedding(slug=slug, vector=self.embed(text), model=self._model)
