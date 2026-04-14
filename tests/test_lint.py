"""Tests for the lint health checks."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from llm_wiki.models.embeddings import EMBEDDING_DIM, PageEmbedding
from llm_wiki.models.operations import LintIssueKind


def _unit_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


class TestRunLint:
    def test_clean_wiki_passes_lint(self, tmp_wiki_dir: Path) -> None:
        from llm_wiki.index import EmbeddingIndex
        from llm_wiki.lint import run_lint
        from llm_wiki.wiki import WikiRepository

        repo = WikiRepository(tmp_wiki_dir / "wiki")
        index = EmbeddingIndex(tmp_wiki_dir / "wiki" / ".meta" / "embeddings.json")

        for slug in ["topic_a", "topic_b", "topic_c"]:
            repo.write(slug=slug, title=slug.replace("_", " ").title(), body=f"Content about {slug}.", sources=["source.md"], tags=["test"])
            index.upsert(PageEmbedding(slug=slug, vector=_unit_vec(hash(slug) % 100)))
        index.save()

        assert run_lint(repo=repo, index=index, deep=False).passed

    def test_orphaned_page_detected(self, tmp_wiki_dir: Path) -> None:
        from llm_wiki.index import EmbeddingIndex
        from llm_wiki.lint import run_lint
        from llm_wiki.wiki import WikiRepository

        repo = WikiRepository(tmp_wiki_dir / "wiki")
        index = EmbeddingIndex(tmp_wiki_dir / "wiki" / ".meta" / "embeddings.json")
        repo.write(slug="ghost_page", title="Ghost", body="Not in schema.", sources=[], tags=[])

        result = run_lint(repo=repo, index=index, deep=False)
        assert not result.passed
        assert any(i.slug == "ghost_page" for i in result.issues_of_kind(LintIssueKind.ORPHANED_PAGE))

    def test_missing_page_detected(self, tmp_wiki_dir: Path) -> None:
        from llm_wiki.index import EmbeddingIndex
        from llm_wiki.lint import run_lint
        from llm_wiki.wiki import WikiRepository

        repo = WikiRepository(tmp_wiki_dir / "wiki")
        index = EmbeddingIndex(tmp_wiki_dir / "wiki" / ".meta" / "embeddings.json")

        result = run_lint(repo=repo, index=index, deep=False)
        assert len(result.issues_of_kind(LintIssueKind.MISSING_PAGE)) == 3

    def test_stale_embedding_detected(self, tmp_wiki_dir: Path) -> None:
        from llm_wiki.index import EmbeddingIndex
        from llm_wiki.lint import run_lint
        from llm_wiki.wiki import WikiRepository

        repo = WikiRepository(tmp_wiki_dir / "wiki")
        index = EmbeddingIndex(tmp_wiki_dir / "wiki" / ".meta" / "embeddings.json")
        repo.write(slug="topic_a", title="Topic A", body="Content.", sources=["s.md"], tags=[])

        result = run_lint(repo=repo, index=index, deep=False)
        assert any(i.slug == "topic_a" for i in result.issues_of_kind(LintIssueKind.STALE_EMBEDDING))

    def test_broken_xref_detected(self, tmp_wiki_dir: Path) -> None:
        from llm_wiki.index import EmbeddingIndex
        from llm_wiki.lint import run_lint
        from llm_wiki.wiki import WikiRepository

        repo = WikiRepository(tmp_wiki_dir / "wiki")
        index = EmbeddingIndex(tmp_wiki_dir / "wiki" / ".meta" / "embeddings.json")
        repo.write(slug="topic_a", title="Topic A", body="Links to [[does_not_exist]].", sources=["s.md"], tags=[])

        result = run_lint(repo=repo, index=index, deep=False)
        assert any("does_not_exist" in i.detail for i in result.issues_of_kind(LintIssueKind.BROKEN_XREF))

    def test_large_page_with_headings_flagged(self, tmp_wiki_dir: Path) -> None:
        from llm_wiki.index import EmbeddingIndex
        from llm_wiki.lint import run_lint
        from llm_wiki.wiki import WikiRepository

        repo = WikiRepository(tmp_wiki_dir / "wiki")
        index = EmbeddingIndex(tmp_wiki_dir / "wiki" / ".meta" / "embeddings.json")

        big_body = "## Collaborative Filtering\n\n" + ("x " * 15_000) + "\n\n## Content Based\n\n" + ("y " * 15_000)
        repo.write(slug="topic_a", title="Topic A", body=big_body, sources=["s.md"], tags=[])

        result = run_lint(repo=repo, index=index, max_tokens=6_000)
        large = result.issues_of_kind(LintIssueKind.PAGE_TOO_LARGE)
        assert len(large) == 1
        assert "collaborative_filtering" in large[0].detail
        assert "content_based" in large[0].detail

    def test_small_page_not_flagged_for_size(self, tmp_wiki_dir: Path) -> None:
        from llm_wiki.index import EmbeddingIndex
        from llm_wiki.lint import run_lint
        from llm_wiki.wiki import WikiRepository

        repo = WikiRepository(tmp_wiki_dir / "wiki")
        index = EmbeddingIndex(tmp_wiki_dir / "wiki" / ".meta" / "embeddings.json")
        repo.write(slug="topic_a", title="Topic A", body="Short content.", sources=["s.md"], tags=[])

        result = run_lint(repo=repo, index=index, max_tokens=6_000)
        assert len(result.issues_of_kind(LintIssueKind.PAGE_TOO_LARGE)) == 0
