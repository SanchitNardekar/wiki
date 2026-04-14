"""Tests for PageEmbedding and EmbeddingIndexData."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from llm_wiki.models.embeddings import EMBEDDING_DIM, EmbeddingIndexData, PageEmbedding


def make_vector(seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return (vec / np.linalg.norm(vec)).tolist()


class TestPageEmbedding:
    def test_valid_embedding_created(self) -> None:
        emb = PageEmbedding(slug="transformers", vector=make_vector())
        assert emb.slug == "transformers"
        assert len(emb.vector) == EMBEDDING_DIM

    def test_wrong_dimension_rejected(self) -> None:
        with pytest.raises(ValidationError, match=str(EMBEDDING_DIM)):
            PageEmbedding(slug="t", vector=[0.1, 0.2, 0.3])

    def test_as_array_returns_float32(self) -> None:
        emb = PageEmbedding(slug="t", vector=make_vector())
        arr = emb.as_array()
        assert arr.dtype == np.float32
        assert arr.shape == (EMBEDDING_DIM,)

    def test_embedding_round_trips_json(self) -> None:
        emb = PageEmbedding(slug="t", vector=make_vector())
        restored = PageEmbedding.model_validate_json(emb.model_dump_json())
        assert restored.slug == emb.slug
        assert len(restored.vector) == EMBEDDING_DIM


class TestEmbeddingIndexData:
    def test_empty_index_constructed(self) -> None:
        idx = EmbeddingIndexData()
        assert idx.slug_count == 0
        assert idx.embeddings == {}

    def test_index_with_embeddings(self) -> None:
        emb = PageEmbedding(slug="a", vector=make_vector())
        idx = EmbeddingIndexData(embeddings={"a": emb})
        assert idx.slug_count == 1

    def test_index_round_trips_json(self) -> None:
        emb = PageEmbedding(slug="a", vector=make_vector())
        idx = EmbeddingIndexData(embeddings={"a": emb})
        restored = EmbeddingIndexData.model_validate_json(idx.model_dump_json())
        assert restored.slug_count == 1
        assert "a" in restored.embeddings

    def test_embedding_dim_constant_is_1536(self) -> None:
        assert EMBEDDING_DIM == 1536
