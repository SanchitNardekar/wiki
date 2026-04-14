"""Tests for EmbeddingIndex."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from llm_wiki.index import EmbeddingIndex
from llm_wiki.models.embeddings import EMBEDDING_DIM, PageEmbedding


def _unit_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


class TestEmbeddingIndex:
    def test_empty_index_initialises_from_empty_file(self, tmp_path: Path) -> None:
        idx = EmbeddingIndex(tmp_path / "embeddings.json")
        assert idx.slugs() == []

    def test_upsert_and_retrieve_slug(self, tmp_path: Path) -> None:
        idx = EmbeddingIndex(tmp_path / "embeddings.json")
        idx.upsert(PageEmbedding(slug="a", vector=_unit_vec(0)))
        assert idx.contains("a")

    def test_save_and_reload_persists_data(self, tmp_path: Path) -> None:
        path = tmp_path / "embeddings.json"
        idx = EmbeddingIndex(path)
        idx.upsert(PageEmbedding(slug="a", vector=_unit_vec(0)))
        idx.save()
        assert EmbeddingIndex(path).contains("a")

    def test_top_k_returns_sorted_by_similarity(self, tmp_path: Path) -> None:
        idx = EmbeddingIndex(tmp_path / "embeddings.json")
        query_vec = _unit_vec(0)
        far_arr = np.random.default_rng(999).standard_normal(EMBEDDING_DIM).astype(np.float32)
        far_vec = (far_arr / np.linalg.norm(far_arr)).tolist()

        idx.upsert(PageEmbedding(slug="close", vector=_unit_vec(0)))
        idx.upsert(PageEmbedding(slug="far", vector=far_vec))

        results = idx.top_k(query_vec, k=2)
        assert results[0][0] == "close"
        assert results[0][1] > results[1][1]

    def test_top_k_respects_k_limit(self, tmp_path: Path) -> None:
        idx = EmbeddingIndex(tmp_path / "embeddings.json")
        for i in range(10):
            idx.upsert(PageEmbedding(slug=f"p{i}", vector=_unit_vec(i)))
        assert len(idx.top_k(_unit_vec(0), k=3)) == 3

    def test_top_k_on_empty_index_returns_empty_list(self, tmp_path: Path) -> None:
        idx = EmbeddingIndex(tmp_path / "embeddings.json")
        assert idx.top_k(_unit_vec(0)) == []

    def test_contains_returns_false_for_missing_slug(self, tmp_path: Path) -> None:
        idx = EmbeddingIndex(tmp_path / "embeddings.json")
        assert not idx.contains("ghost")
