"""Tests for SourceType, ResolvedSource, and IngestRequest."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_wiki.models.source import IngestRequest, ResolvedSource, SourceType


class TestResolvedSource:
    def test_local_file_source_constructed(self) -> None:
        src = ResolvedSource(
            raw="sources/paper.txt",
            source_type=SourceType.LOCAL_FILE,
            name="paper.txt",
            text="Some content here.",
        )
        assert src.source_type == SourceType.LOCAL_FILE
        assert src.name == "paper.txt"

    def test_youtube_source_constructed(self) -> None:
        src = ResolvedSource(
            raw="https://youtube.com/watch?v=abc123",
            source_type=SourceType.YOUTUBE,
            name="youtube:abc123",
            text="Transcript content here.",
        )
        assert src.source_type == SourceType.YOUTUBE

    def test_http_source_constructed(self) -> None:
        src = ResolvedSource(
            raw="https://arxiv.org/abs/1706.03762",
            source_type=SourceType.HTTP,
            name="arxiv.org",
            text="Abstract and content.",
        )
        assert src.source_type == SourceType.HTTP

    def test_token_estimate_rough_approximation(self) -> None:
        src = ResolvedSource(
            raw="f.txt",
            source_type=SourceType.LOCAL_FILE,
            name="f.txt",
            text="a" * 4000,
        )
        assert src.token_estimate == 1000

    def test_empty_text_is_valid(self) -> None:
        src = ResolvedSource(
            raw="empty.txt", source_type=SourceType.LOCAL_FILE, name="empty.txt", text=""
        )
        assert src.text == ""

    def test_source_type_enum_values(self) -> None:
        assert SourceType.LOCAL_FILE == "local_file"
        assert SourceType.YOUTUBE == "youtube"
        assert SourceType.HTTP == "http"


class TestIngestRequest:
    def test_minimal_request_constructed(self) -> None:
        req = IngestRequest(source_path="sources/paper.md")
        assert req.source_path == "sources/paper.md"
        assert req.force is False

    def test_force_flag_respected(self) -> None:
        req = IngestRequest(source_path="sources/x.md", force=True)
        assert req.force is True

    def test_empty_source_path_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IngestRequest(source_path="")
