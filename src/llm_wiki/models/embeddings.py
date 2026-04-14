"""Pydantic models for the embedding index."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field, field_validator

EMBEDDING_DIM: int = 1536
EMBEDDING_MODEL: str = "text-embedding-3-small"


class PageEmbedding(BaseModel):
    slug: str
    vector: list[float] = Field(description=f"Normalised {EMBEDDING_DIM}-dim vector")
    model: str = Field(default=EMBEDDING_MODEL)

    @field_validator("vector")
    @classmethod
    def vector_has_correct_dims(cls, v: list[float]) -> list[float]:
        if len(v) != EMBEDDING_DIM:
            raise ValueError(f"Expected {EMBEDDING_DIM}-dimensional vector, got {len(v)}")
        return v

    def as_array(self) -> np.ndarray:
        return np.array(self.vector, dtype=np.float32)


class EmbeddingIndexData(BaseModel):
    model: str = Field(default=EMBEDDING_MODEL)
    embeddings: dict[str, PageEmbedding] = Field(default_factory=dict)

    @property
    def slug_count(self) -> int:
        return len(self.embeddings)
