"""Tests for EmbeddingService."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from llm_wiki.embeddings import EmbeddingService
from llm_wiki.models.embeddings import EMBEDDING_DIM


def _make_raw_vector(seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(EMBEDDING_DIM).tolist()


class TestEmbeddingService:
    def _make_service(self, raw_vector: list[float] | None = None) -> EmbeddingService:
        mock_client = MagicMock()
        vec = raw_vector or _make_raw_vector()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=vec)]
        mock_client.embeddings.create.return_value = mock_response
        return EmbeddingService(client=mock_client)

    def test_embed_returns_normalised_vector(self) -> None:
        svc = self._make_service()
        result = svc.embed("Some text to embed")
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_embed_returns_correct_dimension(self) -> None:
        svc = self._make_service()
        assert len(svc.embed("text")) == EMBEDDING_DIM

    def test_embed_page_returns_page_embedding(self) -> None:
        svc = self._make_service()
        emb = svc.embed_page(slug="transformers", text="Transformer architecture")
        assert emb.slug == "transformers"
        assert len(emb.vector) == EMBEDDING_DIM

    def test_embed_truncates_very_long_text(self) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=_make_raw_vector())]
        mock_client.embeddings.create.return_value = mock_response
        svc = EmbeddingService(client=mock_client)

        svc.embed("x" * 100_000)

        call_kwargs = mock_client.embeddings.create.call_args
        passed_input = call_kwargs.kwargs.get("input") or call_kwargs.args[1]
        assert len(passed_input) <= 32_768
