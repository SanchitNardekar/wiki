"""Shared pytest fixtures for the full test suite."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from llm_wiki.models.embeddings import EMBEDDING_DIM, EmbeddingIndexData, PageEmbedding
from llm_wiki.models.frontmatter import PageSpec, WikiFrontMatter, WikiSchema


@pytest.fixture
def sample_schema() -> WikiSchema:
    return WikiSchema(
        name="Test Wiki",
        pages=[
            PageSpec(slug="topic_a", title="Topic A", description="About topic A"),
            PageSpec(slug="topic_b", title="Topic B", description="About topic B"),
            PageSpec(slug="topic_c", title="Topic C", description="About topic C"),
        ],
    )


@pytest.fixture
def tmp_wiki_dir(tmp_path: Path, sample_schema: WikiSchema) -> Path:
    """A fully initialised wiki directory in a temp folder.

    Returns the *root* path (tmp_path), not wiki/ itself.
    """
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    meta_dir = wiki_dir / ".meta"
    meta_dir.mkdir()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()

    (meta_dir / "schema.json").write_text(sample_schema.model_dump_json(indent=2))
    (meta_dir / "embeddings.json").write_text(EmbeddingIndexData().model_dump_json(indent=2))
    (wiki_dir / "index.md").write_text("# Index\n\nNo pages yet.\n")
    (wiki_dir / "log.md").write_text("# Log\n\n")

    return tmp_path


@pytest.fixture
def sample_embedding() -> PageEmbedding:
    rng = np.random.default_rng(42)
    vec: np.ndarray = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return PageEmbedding(slug="topic_a", vector=vec.tolist())


@pytest.fixture
def sample_front_matter() -> WikiFrontMatter:
    return WikiFrontMatter(
        slug="topic_a",
        title="Topic A",
        tags=["test", "example"],
        sources=["source1.md"],
        updated=date(2026, 4, 14),
    )
