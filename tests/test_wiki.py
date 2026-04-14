"""Tests for WikiPage and WikiRepository."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from llm_wiki.models.frontmatter import WikiSchema
from llm_wiki.wiki import WikiRepository


class TestWikiRepository:
    def test_read_returns_none_for_missing_page(self, tmp_wiki_dir: Path) -> None:
        assert WikiRepository(tmp_wiki_dir / "wiki").read("nonexistent") is None

    def test_write_creates_readable_page(self, tmp_wiki_dir: Path) -> None:
        repo = WikiRepository(tmp_wiki_dir / "wiki")
        repo.write(slug="topic_a", title="Topic A", body="## Overview\n\nContent here.", sources=["source.md"], tags=["test"])
        page = repo.read("topic_a")
        assert page is not None
        assert page.front_matter.slug == "topic_a"
        assert "Content here." in page.body

    def test_write_persists_front_matter_correctly(self, tmp_wiki_dir: Path) -> None:
        repo = WikiRepository(tmp_wiki_dir / "wiki")
        repo.write(slug="topic_a", title="Topic A", body="Body.", sources=["paper.txt", "notes.md"], tags=["ml", "test"])
        page = repo.read("topic_a")
        assert page is not None
        assert "paper.txt" in page.front_matter.sources
        assert "ml" in page.front_matter.tags

    def test_write_updates_date_to_today(self, tmp_wiki_dir: Path) -> None:
        repo = WikiRepository(tmp_wiki_dir / "wiki")
        repo.write(slug="topic_a", title="T", body="B", sources=[], tags=[])
        page = repo.read("topic_a")
        assert page is not None
        assert page.front_matter.updated == date.today()

    def test_list_slugs_excludes_index_and_log(self, tmp_wiki_dir: Path) -> None:
        slugs = WikiRepository(tmp_wiki_dir / "wiki").list_slugs()
        assert "index" not in slugs
        assert "log" not in slugs

    def test_list_slugs_returns_written_pages(self, tmp_wiki_dir: Path) -> None:
        repo = WikiRepository(tmp_wiki_dir / "wiki")
        repo.write(slug="topic_a", title="T", body="B", sources=[], tags=[])
        repo.write(slug="topic_b", title="T", body="B", sources=[], tags=[])
        slugs = repo.list_slugs()
        assert "topic_a" in slugs
        assert "topic_b" in slugs

    def test_load_schema_returns_wiki_schema(self, tmp_wiki_dir: Path) -> None:
        schema = WikiRepository(tmp_wiki_dir / "wiki").load_schema()
        assert isinstance(schema, WikiSchema)
        assert len(schema.pages) == 3

    def test_append_log_writes_entry(self, tmp_wiki_dir: Path) -> None:
        repo = WikiRepository(tmp_wiki_dir / "wiki")
        repo.append_log("- 2026-04-14 — Test entry")
        assert "Test entry" in (tmp_wiki_dir / "wiki" / "log.md").read_text()

    def test_full_text_property_includes_title(self, tmp_wiki_dir: Path) -> None:
        repo = WikiRepository(tmp_wiki_dir / "wiki")
        repo.write(slug="topic_a", title="My Title", body="Body.", sources=[], tags=[])
        page = repo.read("topic_a")
        assert page is not None
        assert "My Title" in page.full_text
