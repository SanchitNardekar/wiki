"""Tests for WikiSettings and WikiPaths."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from llm_wiki.config import WikiPaths, WikiSettings


class TestWikiSettings:
    def test_settings_load_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        settings = WikiSettings()
        assert settings.openai_api_key == "sk-openai-test"

    def test_default_models_are_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "x")
        settings = WikiSettings()
        assert settings.openai_model == "gpt-5.2"
        assert settings.embedding_model == "text-embedding-3-small"
        assert settings.query_top_k == 5

    def test_missing_api_key_raises(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with pytest.raises(Exception):
            WikiSettings(_env_file=None)  # type: ignore[call-arg]


class TestWikiPaths:
    def test_paths_resolve_from_root(self, tmp_path: Path) -> None:
        paths = WikiPaths(root=tmp_path)
        assert paths.wiki_dir == tmp_path / "wiki"
        assert paths.sources_dir == tmp_path / "sources"
        assert paths.meta_dir == tmp_path / "wiki" / ".meta"
        assert paths.schema_path == tmp_path / "wiki" / ".meta" / "schema.json"
        assert paths.embeddings_path == tmp_path / "wiki" / ".meta" / "embeddings.json"
        assert paths.index_path == tmp_path / "wiki" / "index.md"
        assert paths.log_path == tmp_path / "wiki" / "log.md"
