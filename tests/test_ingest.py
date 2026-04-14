"""Tests for the 5-step ingest pipeline and resolve_source()."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from llm_wiki.ingest import resolve_source, route, run_ingest
from llm_wiki.models.embeddings import PageEmbedding
from llm_wiki.models.operations import RouteResult
from llm_wiki.models.source import ResolvedSource, SourceType


def _mock_openai_response(text: str) -> MagicMock:
    """Build a mock matching OpenAI chat completions response shape."""
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


class TestResolveSource:
    def test_local_file_resolved(self, tmp_path: Path) -> None:
        source_file = tmp_path / "notes.md"
        source_file.write_text("# Notes\n\nContent here.")
        result = resolve_source(str(source_file))
        assert result.source_type == SourceType.LOCAL_FILE
        assert result.name == "notes.md"
        assert "Content here." in result.text

    def test_youtube_url_detected(self) -> None:
        with patch("llm_wiki.ingest.YouTubeTranscriptApi") as mock_api:
            mock_api.get_transcript.return_value = [
                {"text": "Hello world", "start": 0.0, "duration": 2.0}
            ]
            result = resolve_source("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert result.source_type == SourceType.YOUTUBE
        assert "Hello world" in result.text

    def test_http_url_resolved(self) -> None:
        with patch("llm_wiki.ingest.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.text = "<html><body><p>Article content.</p></body></html>"
            mock_httpx.get.return_value = mock_response
            result = resolve_source("https://arxiv.org/abs/1706.03762")
        assert result.source_type == SourceType.HTTP
        assert "Article content." in result.text

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            resolve_source("/nonexistent/path/file.md")

    def test_long_text_truncated_to_max_chars(self, tmp_path: Path) -> None:
        source_file = tmp_path / "huge.txt"
        source_file.write_text("x" * 200_000)
        result = resolve_source(str(source_file), max_chars=50_000)
        assert len(result.text) <= 50_000


class TestRoute:
    def test_route_returns_valid_slugs(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response('["topic_a", "topic_b"]')
        source = ResolvedSource(raw="f.md", source_type=SourceType.LOCAL_FILE, name="f.md", text="text")
        result = route(client=client, source=source, schema_context="context", model="gpt-5.2")
        assert isinstance(result, RouteResult)
        assert "topic_a" in result.relevant_slugs

    def test_route_handles_malformed_llm_response(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response("not valid json")
        source = ResolvedSource(raw="f.md", source_type=SourceType.LOCAL_FILE, name="f.md", text="text")
        result = route(client=client, source=source, schema_context="ctx", model="m")
        assert result.relevant_slugs == []


class TestRunIngest:
    def test_full_ingest_pipeline_writes_pages(self, tmp_wiki_dir: Path) -> None:
        from llm_wiki.config import WikiSettings
        from llm_wiki.embeddings import EmbeddingService
        from llm_wiki.index import EmbeddingIndex
        from llm_wiki.wiki import WikiRepository

        source_file = tmp_wiki_dir / "sources" / "test.md"
        source_file.write_text("# Test\nContent about topic A.")

        repo = WikiRepository(tmp_wiki_dir / "wiki")
        index = EmbeddingIndex(tmp_wiki_dir / "wiki" / ".meta" / "embeddings.json")

        rng = np.random.default_rng(0)
        vec = rng.standard_normal(1536).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        mock_embed_svc = MagicMock(spec=EmbeddingService)
        mock_embed_svc.embed.return_value = [0.0] * 1536
        mock_embed_svc.embed_page.return_value = PageEmbedding(slug="topic_a", vector=vec.tolist())

        mock_openai = MagicMock()
        mock_openai.chat.completions.create.side_effect = [
            _mock_openai_response('["topic_a"]'),
            _mock_openai_response("## Topic A\n\nSynthesised content."),
            _mock_openai_response("# Index\n\n| topic_a | Topic A |"),
        ]

        os.environ.setdefault("OPENAI_API_KEY", "x")
        settings = WikiSettings()

        result = run_ingest(
            source_path=str(source_file), repo=repo, index=index,
            embedding_service=mock_embed_svc, openai_client=mock_openai, settings=settings,
        )
        assert result.success
        assert "topic_a" in result.slugs_touched


    def test_ingest_uses_max_completion_tokens_not_max_tokens(self, tmp_wiki_dir: Path) -> None:
        """GPT-5.2 requires max_completion_tokens, not max_tokens."""
        from llm_wiki.config import WikiSettings
        from llm_wiki.embeddings import EmbeddingService
        from llm_wiki.index import EmbeddingIndex
        from llm_wiki.wiki import WikiRepository

        source_file = tmp_wiki_dir / "sources" / "test.md"
        source_file.write_text("Content about topic A.")

        repo = WikiRepository(tmp_wiki_dir / "wiki")
        index = EmbeddingIndex(tmp_wiki_dir / "wiki" / ".meta" / "embeddings.json")

        rng = np.random.default_rng(0)
        vec = rng.standard_normal(1536).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        mock_embed_svc = MagicMock(spec=EmbeddingService)
        mock_embed_svc.embed.return_value = [0.0] * 1536
        mock_embed_svc.embed_page.return_value = PageEmbedding(slug="topic_a", vector=vec.tolist())

        mock_openai = MagicMock()
        mock_openai.chat.completions.create.side_effect = [
            _mock_openai_response('["topic_a"]'),
            _mock_openai_response("## Synthesised."),
            _mock_openai_response("# Index"),
        ]

        os.environ.setdefault("OPENAI_API_KEY", "x")
        run_ingest(
            source_path=str(source_file), repo=repo, index=index,
            embedding_service=mock_embed_svc, openai_client=mock_openai, settings=WikiSettings(),
        )

        for call in mock_openai.chat.completions.create.call_args_list:
            kwargs = call.kwargs
            assert "max_completion_tokens" in kwargs, "Should use max_completion_tokens for GPT-5.2"
            assert "max_tokens" not in kwargs, "max_tokens is not supported by GPT-5.2"
