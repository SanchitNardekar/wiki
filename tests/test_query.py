"""Tests for the 4-step query pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from llm_wiki.models.embeddings import EMBEDDING_DIM, PageEmbedding
from llm_wiki.models.operations import QueryResult


def _unit_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _mock_openai_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


class TestRunQuery:
    def test_query_returns_answer(self, tmp_wiki_dir: Path) -> None:
        from llm_wiki.config import WikiSettings
        from llm_wiki.embeddings import EmbeddingService
        from llm_wiki.index import EmbeddingIndex
        from llm_wiki.query import run_query
        from llm_wiki.wiki import WikiRepository

        os.environ.setdefault("OPENAI_API_KEY", "x")

        repo = WikiRepository(tmp_wiki_dir / "wiki")
        repo.write(slug="topic_a", title="Topic A", body="## Overview\n\nContent.", sources=[], tags=[])

        index = EmbeddingIndex(tmp_wiki_dir / "wiki" / ".meta" / "embeddings.json")
        index.upsert(PageEmbedding(slug="topic_a", vector=_unit_vec(0)))
        index.save()

        mock_embed_svc = MagicMock(spec=EmbeddingService)
        mock_embed_svc.embed.return_value = _unit_vec(0)

        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = _mock_openai_response("Topic A is about...")

        result = run_query(
            question="What is Topic A?",
            repo=repo, index=index, embedding_service=mock_embed_svc,
            openai_client=mock_openai, settings=WikiSettings(),
        )
        assert isinstance(result, QueryResult)
        assert len(result.answer) > 0

    def test_query_with_save_creates_new_page(self, tmp_wiki_dir: Path) -> None:
        from llm_wiki.config import WikiSettings
        from llm_wiki.embeddings import EmbeddingService
        from llm_wiki.index import EmbeddingIndex
        from llm_wiki.query import run_query
        from llm_wiki.wiki import WikiRepository

        os.environ.setdefault("OPENAI_API_KEY", "x")

        repo = WikiRepository(tmp_wiki_dir / "wiki")
        repo.write(slug="topic_a", title="Topic A", body="Content.", sources=[], tags=[])

        index = EmbeddingIndex(tmp_wiki_dir / "wiki" / ".meta" / "embeddings.json")
        index.upsert(PageEmbedding(slug="topic_a", vector=_unit_vec(0)))
        index.save()

        mock_embed_svc = MagicMock(spec=EmbeddingService)
        mock_embed_svc.embed.return_value = _unit_vec(0)
        mock_embed_svc.embed_page.return_value = PageEmbedding(slug="saved_answer", vector=_unit_vec(1))

        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = _mock_openai_response("The answer is...")

        result = run_query(
            question="What is the answer?",
            repo=repo, index=index, embedding_service=mock_embed_svc,
            openai_client=mock_openai, settings=WikiSettings(), save=True,
        )
        assert result.saved_slug is not None


    def test_query_uses_max_completion_tokens_not_max_tokens(self, tmp_wiki_dir: Path) -> None:
        """GPT-5.2 requires max_completion_tokens, not max_tokens."""
        from llm_wiki.config import WikiSettings
        from llm_wiki.embeddings import EmbeddingService
        from llm_wiki.index import EmbeddingIndex
        from llm_wiki.query import run_query
        from llm_wiki.wiki import WikiRepository

        os.environ.setdefault("OPENAI_API_KEY", "x")

        repo = WikiRepository(tmp_wiki_dir / "wiki")
        repo.write(slug="topic_a", title="Topic A", body="Content.", sources=[], tags=[])

        index = EmbeddingIndex(tmp_wiki_dir / "wiki" / ".meta" / "embeddings.json")
        index.upsert(PageEmbedding(slug="topic_a", vector=_unit_vec(0)))
        index.save()

        mock_embed_svc = MagicMock(spec=EmbeddingService)
        mock_embed_svc.embed.return_value = _unit_vec(0)

        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = _mock_openai_response("Answer.")

        run_query(
            question="test?", repo=repo, index=index, embedding_service=mock_embed_svc,
            openai_client=mock_openai, settings=WikiSettings(),
        )

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert "max_completion_tokens" in call_kwargs, "Should use max_completion_tokens for GPT-5.2"
        assert "max_tokens" not in call_kwargs, "max_tokens is not supported by GPT-5.2"
