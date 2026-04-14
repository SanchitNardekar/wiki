# LLM Wiki — Claude Code Implementation Plan

**Pattern:** Karpathy's LLM Wiki (compiling knowledge that compounds)
**Stack:** Python 3.13 · uv · ruff · ty · Pydantic v2 · Typer · Anthropic API · OpenAI API
**Methodology:** TDD throughout — every test file is written *before* the module it covers.

---

## Architecture Overview

The system has three cleanly separated layers, each with strict ownership:

```
┌─────────────┬────────────────────────┬───────────────────────────────┐
│    Layer    │       Location         │             Owner             │
├─────────────┼────────────────────────┼───────────────────────────────┤
│ Raw Sources │ sources/               │ Human (add files here)        │
├─────────────┼────────────────────────┼───────────────────────────────┤
│ Wiki        │ wiki/*.md + .meta/     │ LLM (auto-maintained)         │
├─────────────┼────────────────────────┼───────────────────────────────┤
│ Schema      │ wiki/.meta/schema.json │ Human (defines page universe) │
└─────────────┴────────────────────────┴───────────────────────────────┘
```

The four core operations map to the knowledge lifecycle:

- **init** — Bootstrap directory structure from schema
- **ingest** — Route → Synthesize → Embed → Update Index → Log (5 steps)
- **query** — Embed → Search → Assemble → Stream (4 steps), optional `--save`
- **lint** — Structural checks; `--deep` triggers LLM contradiction analysis

---

## Final Project Structure

```
llm-wiki/
├── pyproject.toml
├── .python-version
├── .env.example
├── .gitignore
├── CLAUDE.md                          # Agent runtime rules
├── main.py                            # Entry point shortcut
│
├── src/
│   └── llm_wiki/
│       ├── __init__.py
│       ├── cli.py                     # Typer CLI: 6 commands
│       ├── config.py                  # Pydantic Settings + path constants
│       ├── prompts.py                 # ALL prompt text + QUERY_TEMPLATES dict
│       ├── embeddings.py              # OpenAI text-embedding-3-small wrapper
│       ├── index.py                   # EmbeddingIndex: cosine similarity search
│       ├── wiki.py                    # WikiPage + WikiRepository CRUD
│       ├── ingest.py                  # 5-step ingest pipeline
│       ├── query.py                   # 4-step query pipeline
│       ├── lint.py                    # Health checks
│       └── models/
│           ├── __init__.py
│           ├── frontmatter.py         # WikiFrontMatter, PageSpec, WikiSchema
│           ├── source.py              # SourceType, ResolvedSource, IngestRequest
│           ├── operations.py          # RouteResult, IngestResult, QueryResult, LintResult
│           └── embeddings.py          # PageEmbedding, EmbeddingIndexData
│
├── tests/
│   ├── conftest.py
│   ├── test_models/
│   │   ├── __init__.py
│   │   ├── test_frontmatter.py
│   │   ├── test_source.py
│   │   ├── test_operations.py
│   │   └── test_embeddings.py
│   ├── test_config.py
│   ├── test_prompts.py
│   ├── test_embeddings_module.py
│   ├── test_index.py
│   ├── test_wiki.py
│   ├── test_ingest.py
│   ├── test_query.py
│   └── test_lint.py
│
├── sources/                           # Raw sources (immutable, human-managed)
│   └── .gitkeep
│
└── wiki/                              # LLM-maintained knowledge base
    ├── index.md
    ├── log.md
    └── .meta/
        ├── schema.json
        └── embeddings.json
```

---

## Phase 0 — Project Scaffold

### Step 0.1 — Create the directory skeleton

```bash
mkdir -p src/llm_wiki/models
mkdir -p tests/test_models
mkdir -p sources
mkdir -p wiki/.meta
touch src/llm_wiki/__init__.py
touch src/llm_wiki/models/__init__.py
touch tests/__init__.py
touch tests/test_models/__init__.py
touch sources/.gitkeep
```

### Step 0.2 — `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "llm-wiki"
version = "0.1.0"
description = "Karpathy's LLM Wiki pattern — compiling knowledge that compounds"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "anthropic>=0.50.0",
    "openai>=1.50.0",
    "numpy>=2.0.0",
    "python-frontmatter>=1.1.0",
    "pydantic>=2.7.0",
    "pydantic-settings>=2.3.0",
    "typer>=0.12.0",
    "rich>=14.0.0",
    "httpx>=0.27.0",
    "youtube-transcript-api>=0.6.0",
    "pypdf>=4.0.0",
]

[project.scripts]
llm-wiki = "llm_wiki.cli:app"

[tool.uv]
dev-dependencies = [
    "pytest>=8.2.0",
    "pytest-mock>=3.14.0",
    "respx>=0.21.0",
    "pytest-cov>=5.0.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/llm_wiki"]

# ── Ruff ──────────────────────────────────────────────────────────────────────
[tool.ruff]
target-version = "py313"
line-length = 100
src = ["src"]

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "F",    # pyflakes
    "W",    # pycodestyle warnings
    "I",    # isort
    "N",    # pep8-naming
    "UP",   # pyupgrade (enforces 3.13 idioms)
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "SIM",  # flake8-simplify
    "ANN",  # flake8-annotations (enforces type hints)
    "RUF",  # ruff-specific rules
    "PTH",  # flake8-use-pathlib (no os.path)
    "TCH",  # flake8-type-checking (TYPE_CHECKING guards)
]
ignore = [
    "ANN101",  # self type annotation not required
    "ANN102",  # cls type annotation not required
    "ANN401",  # allow Any in specific cases
]

[tool.ruff.lint.isort]
known-first-party = ["llm_wiki"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

# ── ty (Astral type checker) ───────────────────────────────────────────────────
[tool.ty]
# Run: uv run ty check src/
environment.python = "3.13"
strict = true

# ── pytest ─────────────────────────────────────────────────────────────────────
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--tb=short -v --cov=src/llm_wiki --cov-report=term-missing"
```

### Step 0.3 — `.python-version`

```
3.13
```

### Step 0.4 — `.env.example`

```dotenv
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Optional overrides
# WIKI_DIR=wiki
# SOURCES_DIR=sources
# CLAUDE_MODEL=claude-opus-4-5
# EMBEDDING_MODEL=text-embedding-3-small
# QUERY_TOP_K=5
```

### Step 0.5 — `.gitignore`

```gitignore
__pycache__/
*.py[cod]
.venv/
.env
.coverage
htmlcov/
dist/
.ruff_cache/
.ty_cache/
*.egg-info/
```

---

## Phase 1 — Pydantic Models (Tests First)

> The models are the type system of this project. Every boundary — between pipeline
> steps, between LLM output and Python code, between disk and memory — is enforced
> by a Pydantic model. Write tests that document what is and isn't acceptable.

### Step 1.1 — `tests/conftest.py`

```python
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
    """A minimal but structurally valid schema for most tests."""
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

    Returns the *root* path (tmp_path), not wiki/ itself, so callers can
    resolve both wiki/ and sources/ from it.
    """
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    meta_dir = wiki_dir / ".meta"
    meta_dir.mkdir()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()

    (meta_dir / "schema.json").write_text(sample_schema.model_dump_json(indent=2))
    empty_index = EmbeddingIndexData()
    (meta_dir / "embeddings.json").write_text(empty_index.model_dump_json(indent=2))
    (wiki_dir / "index.md").write_text("# Index\n\nNo pages yet.\n")
    (wiki_dir / "log.md").write_text("# Log\n\n")

    return tmp_path


@pytest.fixture
def sample_embedding() -> PageEmbedding:
    """A valid normalised 1536-dim embedding for topic_a."""
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
```

### Step 1.2 — `tests/test_models/test_frontmatter.py`

```python
"""Tests for PageSpec, WikiSchema, and WikiFrontMatter."""

from __future__ import annotations

import json
from datetime import date

import pytest

from llm_wiki.models.frontmatter import PageSpec, WikiFrontMatter, WikiSchema


class TestPageSpec:
    def test_valid_lowercase_slug_accepted(self) -> None:
        spec = PageSpec(slug="attention_mechanism", title="Attention", description="About attention")
        assert spec.slug == "attention_mechanism"

    def test_slug_with_hyphen_accepted(self) -> None:
        spec = PageSpec(slug="back-prop", title="Backprop", description="Backpropagation basics")
        assert spec.slug == "back-prop"

    def test_slug_with_numbers_accepted(self) -> None:
        spec = PageSpec(slug="transformer2017", title="Transformer", description="The 2017 paper")
        assert spec.slug == "transformer2017"

    def test_uppercase_slug_rejected(self) -> None:
        with pytest.raises(ValueError, match="lowercase"):
            PageSpec(slug="Attention", title="A", description="d")

    def test_slug_with_space_rejected(self) -> None:
        with pytest.raises(ValueError):
            PageSpec(slug="attention mechanism", title="A", description="d")

    def test_slug_starting_with_hyphen_rejected(self) -> None:
        with pytest.raises(ValueError):
            PageSpec(slug="-bad", title="A", description="d")

    def test_empty_title_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            PageSpec(slug="valid", title="   ", description="d")

    def test_whitespace_only_description_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            PageSpec(slug="valid", title="Valid", description="  ")

    def test_title_leading_trailing_whitespace_stripped(self) -> None:
        spec = PageSpec(slug="valid", title="  My Title  ", description="d")
        assert spec.title == "My Title"

    def test_description_whitespace_stripped(self) -> None:
        spec = PageSpec(slug="valid", title="T", description="  desc  ")
        assert spec.description == "desc"


class TestWikiSchema:
    def test_valid_schema_round_trips_json(self, sample_schema: WikiSchema) -> None:
        serialised = sample_schema.model_dump_json()
        restored = WikiSchema.model_validate_json(serialised)
        assert restored.slug_set() == sample_schema.slug_set()

    def test_duplicate_slugs_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            WikiSchema(
                name="Bad",
                pages=[
                    PageSpec(slug="dup", title="A", description="d"),
                    PageSpec(slug="dup", title="B", description="d"),
                ],
            )

    def test_routing_context_one_line_per_page(self) -> None:
        schema = WikiSchema(
            name="T",
            pages=[
                PageSpec(slug="a", title="Topic A", description="About A"),
                PageSpec(slug="b", title="Topic B", description="About B"),
            ],
        )
        ctx = schema.routing_context()
        lines = ctx.splitlines()
        assert len(lines) == 2
        assert lines[0] == "a: Topic A — About A"
        assert lines[1] == "b: Topic B — About B"

    def test_get_page_returns_matching_spec(self) -> None:
        schema = WikiSchema(
            name="T",
            pages=[PageSpec(slug="found", title="Found", description="d")],
        )
        spec = schema.get_page("found")
        assert spec is not None
        assert spec.slug == "found"

    def test_get_page_returns_none_for_missing_slug(self) -> None:
        schema = WikiSchema(name="T", pages=[])
        assert schema.get_page("ghost") is None

    def test_slug_set_returns_frozenset(self) -> None:
        schema = WikiSchema(
            name="T",
            pages=[
                PageSpec(slug="a", title="A", description="d"),
                PageSpec(slug="b", title="B", description="d"),
            ],
        )
        assert schema.slug_set() == frozenset({"a", "b"})

    def test_empty_pages_list_is_valid(self) -> None:
        schema = WikiSchema(name="Empty Wiki", pages=[])
        assert schema.routing_context() == ""


class TestWikiFrontMatter:
    def test_valid_front_matter_created(self) -> None:
        fm = WikiFrontMatter(
            slug="test_page",
            title="Test Page",
            tags=["ml", "nlp"],
            sources=["paper.txt"],
            updated=date(2026, 4, 14),
        )
        assert fm.slug == "test_page"
        assert fm.updated == date(2026, 4, 14)

    def test_invalid_slug_raises(self) -> None:
        with pytest.raises(ValueError):
            WikiFrontMatter(slug="Invalid Slug", title="T", updated=date.today())

    def test_tags_normalised_to_lowercase(self) -> None:
        fm = WikiFrontMatter(
            slug="s",
            title="T",
            tags=["ML", "Deep-Learning", "NLP"],
            updated=date.today(),
        )
        assert fm.tags == ["ml", "deep-learning", "nlp"]

    def test_none_tags_defaults_to_empty_list(self) -> None:
        fm = WikiFrontMatter(slug="s", title="T", updated=date.today())
        assert fm.tags == []

    def test_add_source_appends_new_source(self) -> None:
        fm = WikiFrontMatter(
            slug="s", title="T", sources=["old.md"], updated=date(2020, 1, 1)
        )
        new_fm = fm.add_source("new.md")
        assert "old.md" in new_fm.sources
        assert "new.md" in new_fm.sources

    def test_add_source_is_idempotent(self) -> None:
        fm = WikiFrontMatter(
            slug="s", title="T", sources=["existing.md"], updated=date.today()
        )
        new_fm = fm.add_source("existing.md")
        assert new_fm.sources.count("existing.md") == 1

    def test_add_source_updates_date(self) -> None:
        fm = WikiFrontMatter(
            slug="s", title="T", sources=[], updated=date(2020, 1, 1)
        )
        new_fm = fm.add_source("new.md")
        assert new_fm.updated == date.today()

    def test_front_matter_serialises_date_as_string(self) -> None:
        fm = WikiFrontMatter(
            slug="s", title="T", updated=date(2026, 4, 14)
        )
        data = fm.model_dump(mode="json")
        assert data["updated"] == "2026-04-14"
```

### Step 1.3 — `tests/test_models/test_source.py`

```python
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
            text="a" * 4000,  # ~1000 tokens
        )
        assert src.token_estimate == 1000

    def test_empty_text_is_valid(self) -> None:
        # empty text is allowed — caller decides whether to skip
        src = ResolvedSource(
            raw="empty.txt",
            source_type=SourceType.LOCAL_FILE,
            name="empty.txt",
            text="",
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
```

### Step 1.4 — `tests/test_models/test_operations.py`

```python
"""Tests for RouteResult, IngestResult, QueryResult, LintResult."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from llm_wiki.models.operations import (
    IngestResult,
    LintIssue,
    LintIssueKind,
    LintResult,
    QueryResult,
    RouteResult,
    SimilarityHit,
    SynthesisResult,
)


class TestRouteResult:
    def test_valid_route_result(self) -> None:
        result = RouteResult(
            source_name="paper.md",
            relevant_slugs=["attention_mechanism", "transformers"],
        )
        assert len(result.relevant_slugs) == 2

    def test_slugs_normalised_to_lowercase(self) -> None:
        result = RouteResult(
            source_name="paper.md",
            relevant_slugs=["ATTENTION", "Transformers"],
        )
        assert result.relevant_slugs == ["attention", "transformers"]

    def test_empty_slugs_list_is_valid(self) -> None:
        result = RouteResult(source_name="paper.md", relevant_slugs=[])
        assert result.relevant_slugs == []

    def test_non_list_slugs_defaults_to_empty(self) -> None:
        # Simulates a malformed LLM response
        result = RouteResult.model_validate({"source_name": "f.md", "relevant_slugs": None})
        assert result.relevant_slugs == []


class TestSynthesisResult:
    def test_valid_synthesis_result(self) -> None:
        result = SynthesisResult(slug="transformers", new_body="## Overview\n\nContent here.")
        assert result.slug == "transformers"

    def test_empty_body_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SynthesisResult(slug="transformers", new_body="   ")


class TestIngestResult:
    def test_successful_result_has_defaults(self) -> None:
        result = IngestResult(source_name="paper.md", slugs_touched=["a", "b"])
        assert result.success is True
        assert result.error is None

    def test_failed_result_carries_error(self) -> None:
        result = IngestResult(source_name="bad.pdf", slugs_touched=[], success=False, error="File not found")
        assert not result.success
        assert result.error == "File not found"

    def test_log_line_format(self) -> None:
        ts = datetime(2026, 4, 14, 10, 30, 0, tzinfo=timezone.utc)
        result = IngestResult(source_name="paper.md", slugs_touched=["a", "b"], timestamp=ts)
        line = result.log_line()
        assert "2026-04-14 10:30:00 UTC" in line
        assert "paper.md" in line
        assert "[a, b]" in line

    def test_log_line_with_no_slugs(self) -> None:
        result = IngestResult(source_name="paper.md", slugs_touched=[])
        line = result.log_line()
        assert "none" in line


class TestQueryResult:
    def test_query_result_without_save(self) -> None:
        result = QueryResult(
            question="What is attention?",
            hits=[SimilarityHit(slug="attention_mechanism", title="Attention Mechanism", score=0.85)],
            answer="Attention is a mechanism that...",
        )
        assert result.saved_slug is None

    def test_query_result_with_save(self) -> None:
        result = QueryResult(
            question="What is attention?",
            hits=[],
            answer="...",
            saved_slug="what_is_attention",
        )
        assert result.saved_slug == "what_is_attention"

    def test_similarity_score_range_enforced(self) -> None:
        with pytest.raises(ValidationError):
            SimilarityHit(slug="a", title="A", score=1.5)  # > 1.0


class TestLintResult:
    def test_empty_lint_result_passes(self) -> None:
        result = LintResult(pages_checked=5)
        assert result.passed is True
        assert result.issue_count == 0

    def test_result_with_issues_fails(self) -> None:
        result = LintResult(
            issues=[LintIssue(kind=LintIssueKind.ORPHANED_PAGE, slug="ghost", detail="Not in schema")],
            pages_checked=3,
        )
        assert not result.passed
        assert result.issue_count == 1

    def test_issues_of_kind_filters_correctly(self) -> None:
        result = LintResult(
            issues=[
                LintIssue(kind=LintIssueKind.ORPHANED_PAGE, slug="a", detail="orphan"),
                LintIssue(kind=LintIssueKind.BROKEN_XREF, slug="b", detail="broken"),
                LintIssue(kind=LintIssueKind.ORPHANED_PAGE, slug="c", detail="orphan"),
            ],
            pages_checked=3,
        )
        orphans = result.issues_of_kind(LintIssueKind.ORPHANED_PAGE)
        assert len(orphans) == 2
        assert all(i.kind == LintIssueKind.ORPHANED_PAGE for i in orphans)

    def test_lint_issue_str_includes_slug(self) -> None:
        issue = LintIssue(kind=LintIssueKind.BROKEN_XREF, slug="transformers", detail="[[missing]] not found")
        assert "transformers" in str(issue)
        assert "broken_xref" in str(issue)
```

### Step 1.5 — `tests/test_models/test_embeddings.py`

```python
"""Tests for PageEmbedding and EmbeddingIndexData."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from llm_wiki.models.embeddings import EMBEDDING_DIM, EmbeddingIndexData, PageEmbedding


def make_vector(seed: int = 0) -> list[float]:
    """Return a normalised EMBEDDING_DIM vector."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


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
```

### Step 1.6 — `src/llm_wiki/models/frontmatter.py`

```python
"""Pydantic models for wiki page frontmatter and schema governance.

These types form the governance contract between human intent (schema)
and LLM execution (page writes). Every file write is validated here.
"""

from __future__ import annotations

import re
from datetime import date

from pydantic import BaseModel, Field, field_validator, model_validator

_SLUG_PATTERN = re.compile(r"[a-z0-9][a-z0-9_\-]*")


def _validate_slug(v: str) -> str:
    """Shared slug validation: lowercase alphanumeric with _ or -."""
    if not _SLUG_PATTERN.fullmatch(v):
        raise ValueError(
            f"Slug must be lowercase alphanumeric (underscores/hyphens allowed), got: {v!r}"
        )
    return v


class PageSpec(BaseModel):
    """A single page definition in the wiki schema.

    The schema is the contract between human intent and LLM execution.
    Humans add PageSpecs; the LLM handles how knowledge is written.
    """

    slug: str = Field(description="Machine-readable ID, e.g. 'attention_mechanism'")
    title: str = Field(description="Human-readable title, e.g. 'Attention Mechanism'")
    description: str = Field(description="One-line description used in routing context")

    @field_validator("slug")
    @classmethod
    def slug_is_valid(cls, v: str) -> str:
        return _validate_slug(v)

    @field_validator("title", "description")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Value must not be empty or whitespace only")
        return v.strip()


class WikiSchema(BaseModel):
    """The full page universe — stored at wiki/.meta/schema.json.

    This is the only file humans actively manage. Everything downstream
    (page creation, routing, indexing) is automated by the LLM.
    """

    name: str = Field(description="Wiki name, e.g. 'ML Fundamentals'")
    pages: list[PageSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def slugs_are_unique(self) -> WikiSchema:
        slugs = [p.slug for p in self.pages]
        if len(slugs) != len(set(slugs)):
            duplicates = sorted({s for s in slugs if slugs.count(s) > 1})
            raise ValueError(f"Duplicate slugs found in schema: {duplicates}")
        return self

    def routing_context(self) -> str:
        """Compact schema summary for the LLM routing prompt.

        Format per line: 'slug: title — description'
        Designed to be minimal — keeps routing cost low.
        """
        return "\n".join(
            f"{p.slug}: {p.title} — {p.description}" for p in self.pages
        )

    def get_page(self, slug: str) -> PageSpec | None:
        return next((p for p in self.pages if p.slug == slug), None)

    def slug_set(self) -> frozenset[str]:
        return frozenset(p.slug for p in self.pages)


class WikiFrontMatter(BaseModel):
    """YAML front matter schema enforced on every wiki page.

    Validated on both read (from disk) and write (before file I/O),
    so the LLM can never produce a structurally malformed page.

    Example YAML:
        ---
        slug: attention_mechanism
        title: Attention Mechanism
        tags:
          - ml
          - nlp
        sources:
          - attention_paper_excerpt.txt
        updated: "2026-04-14"
        ---
    """

    slug: str
    title: str
    tags: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    updated: date

    @field_validator("slug")
    @classmethod
    def slug_is_valid(cls, v: str) -> str:
        return _validate_slug(v)

    @field_validator("tags", mode="before")
    @classmethod
    def coerce_tags(cls, v: object) -> list[str]:
        if not isinstance(v, list):
            return []
        return [str(t).lower().strip() for t in v if t]

    @field_validator("sources", mode="before")
    @classmethod
    def coerce_sources(cls, v: object) -> list[str]:
        if not isinstance(v, list):
            return []
        return [str(s).strip() for s in v if s]

    def add_source(self, source_name: str) -> WikiFrontMatter:
        """Return a new WikiFrontMatter with source appended (immutable update)."""
        if source_name in self.sources:
            return self
        return self.model_copy(
            update={"sources": [*self.sources, source_name], "updated": date.today()}
        )
```

### Step 1.7 — `src/llm_wiki/models/source.py`

```python
"""Pydantic models for source resolution.

Every ingest starts with a raw path or URL. These types capture the
pipeline contract from raw input → type-detected → text extracted.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class SourceType(StrEnum):
    LOCAL_FILE = "local_file"
    YOUTUBE = "youtube"
    HTTP = "http"


class ResolvedSource(BaseModel):
    """A source after type detection and full text extraction.

    This is the output of resolve_source() and the input to route().
    """

    raw: str = Field(description="Original input: file path or URL")
    source_type: SourceType
    name: str = Field(description="Short display name for provenance tracking")
    text: str = Field(description="Extracted plain text, ready for the LLM")

    @property
    def token_estimate(self) -> int:
        """Rough token estimate (~4 chars per token)."""
        return len(self.text) // 4


class IngestRequest(BaseModel):
    """Validated input to the ingest command."""

    source_path: str = Field(description="File path or URL to ingest")
    force: bool = Field(
        default=False,
        description="Re-ingest even if source already appears in log.md",
    )

    @field_validator("source_path")
    @classmethod
    def path_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("source_path must not be empty")
        return v.strip()
```

### Step 1.8 — `src/llm_wiki/models/operations.py`

```python
"""Pydantic models for all pipeline operation inputs and outputs.

These are the typed contracts flowing between pipeline steps and
between the LLM's structured JSON output and Python code.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class RouteResult(BaseModel):
    """LLM routing step output: which pages are relevant to a source.

    The LLM returns a JSON array of slugs; this model validates that
    the response is well-formed before we proceed to synthesis.
    """

    source_name: str
    relevant_slugs: list[str] = Field(
        description="Slugs of pages the LLM determined are relevant"
    )

    @field_validator("relevant_slugs", mode="before")
    @classmethod
    def coerce_slugs(cls, v: object) -> list[str]:
        if not isinstance(v, list):
            return []
        return [str(s).lower().strip() for s in v if s]


class SynthesisResult(BaseModel):
    """LLM synthesis step output: an updated wiki page body."""

    slug: str
    new_body: str = Field(
        description="Full updated markdown body (no YAML front matter)"
    )

    @field_validator("new_body")
    @classmethod
    def body_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Synthesised body must not be empty")
        return v.strip()


class IngestResult(BaseModel):
    """Full outcome of one complete ingest pipeline run."""

    source_name: str
    slugs_touched: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    success: bool = True
    error: str | None = None

    def log_line(self) -> str:
        """Format as a single append-only log.md entry."""
        ts = self.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
        slugs = ", ".join(self.slugs_touched) if self.slugs_touched else "none"
        return f"- {ts} — Ingest: {self.source_name} → touched: [{slugs}]"


class SimilarityHit(BaseModel):
    """A single ranked retrieval result from the embedding index."""

    slug: str
    title: str
    score: float = Field(ge=0.0, le=1.0, description="Cosine similarity score")


class QueryResult(BaseModel):
    """Full outcome of a query pipeline run."""

    question: str
    hits: list[SimilarityHit] = Field(default_factory=list)
    answer: str
    saved_slug: str | None = Field(
        default=None,
        description="Slug of new page created by --save, if applicable",
    )


class LintIssueKind(StrEnum):
    ORPHANED_PAGE = "orphaned_page"
    MISSING_PAGE = "missing_page"
    BROKEN_XREF = "broken_xref"
    STALE_EMBEDDING = "stale_embedding"
    MISSING_PROVENANCE = "missing_provenance"
    CONTRADICTION = "contradiction"


class LintIssue(BaseModel):
    kind: LintIssueKind
    slug: str | None = None
    detail: str

    def __str__(self) -> str:
        prefix = f"[{self.slug}] " if self.slug else ""
        return f"{prefix}{self.kind}: {self.detail}"


class LintResult(BaseModel):
    """Complete output of a lint run."""

    issues: list[LintIssue] = Field(default_factory=list)
    pages_checked: int = 0
    deep: bool = False

    @property
    def passed(self) -> bool:
        return len(self.issues) == 0

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    def issues_of_kind(self, kind: LintIssueKind) -> list[LintIssue]:
        return [i for i in self.issues if i.kind == kind]
```

### Step 1.9 — `src/llm_wiki/models/embeddings.py`

```python
"""Pydantic models for the embedding index.

The index is a flat JSON file mapping slugs to 1536-dim vectors.
Strict dimension validation prevents silent shape bugs in cosine
similarity calculations.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field, field_validator

EMBEDDING_DIM: int = 1536  # text-embedding-3-small output dimension
EMBEDDING_MODEL: str = "text-embedding-3-small"


class PageEmbedding(BaseModel):
    """One wiki page's embedding vector, with dimension validation."""

    slug: str
    vector: list[float] = Field(description=f"Normalised {EMBEDDING_DIM}-dim vector")
    model: str = Field(default=EMBEDDING_MODEL)

    @field_validator("vector")
    @classmethod
    def vector_has_correct_dims(cls, v: list[float]) -> list[float]:
        if len(v) != EMBEDDING_DIM:
            raise ValueError(
                f"Expected {EMBEDDING_DIM}-dimensional vector, got {len(v)}"
            )
        return v

    def as_array(self) -> np.ndarray:
        """Return the vector as a float32 numpy array for computation."""
        return np.array(self.vector, dtype=np.float32)


class EmbeddingIndexData(BaseModel):
    """Serialisable form of the entire embedding index.

    Stored at wiki/.meta/embeddings.json. Loading this model validates
    every stored embedding on startup, catching any disk corruption.
    """

    model: str = Field(default=EMBEDDING_MODEL)
    embeddings: dict[str, PageEmbedding] = Field(default_factory=dict)

    @property
    def slug_count(self) -> int:
        return len(self.embeddings)
```

### Step 1.10 — `src/llm_wiki/models/__init__.py`

```python
"""Public re-exports for llm_wiki.models.

Import from here to avoid coupling callers to internal sub-modules.
"""

from llm_wiki.models.embeddings import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EmbeddingIndexData,
    PageEmbedding,
)
from llm_wiki.models.frontmatter import PageSpec, WikiFrontMatter, WikiSchema
from llm_wiki.models.operations import (
    IngestResult,
    LintIssue,
    LintIssueKind,
    LintResult,
    QueryResult,
    RouteResult,
    SimilarityHit,
    SynthesisResult,
)
from llm_wiki.models.source import IngestRequest, ResolvedSource, SourceType

__all__ = [
    # embeddings
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "EmbeddingIndexData",
    "PageEmbedding",
    # frontmatter
    "PageSpec",
    "WikiFrontMatter",
    "WikiSchema",
    # operations
    "IngestResult",
    "LintIssue",
    "LintIssueKind",
    "LintResult",
    "QueryResult",
    "RouteResult",
    "SimilarityHit",
    "SynthesisResult",
    # source
    "IngestRequest",
    "ResolvedSource",
    "SourceType",
]
```

---

## Phase 2 — Configuration

### Step 2.1 — `tests/test_config.py`

```python
"""Tests for WikiSettings and WikiPaths."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from llm_wiki.config import WikiPaths, WikiSettings


class TestWikiSettings:
    def test_settings_load_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        settings = WikiSettings()
        assert settings.anthropic_api_key == "sk-ant-test"
        assert settings.openai_api_key == "sk-openai-test"

    def test_default_models_are_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        monkeypatch.setenv("OPENAI_API_KEY", "x")
        settings = WikiSettings()
        assert settings.embedding_model == "text-embedding-3-small"
        assert settings.query_top_k == 5

    def test_missing_api_keys_raise(self) -> None:
        # Ensure no env vars are set
        env = {k: v for k, v in os.environ.items()
               if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")}
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
```

### Step 2.2 — `src/llm_wiki/config.py`

```python
"""Configuration and path management.

WikiSettings: validated env-var config via pydantic-settings.
WikiPaths: all filesystem paths derived from a single root, making
           the whole system relocatable and easy to test.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WikiSettings(BaseSettings):
    """All configuration, loaded from environment variables or .env file.

    Pydantic-settings validates types and raises on missing required fields,
    preventing the system from running with incomplete credentials.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Allow extra env vars without raising
        extra="ignore",
    )

    # ── Required API credentials ──────────────────────────────────────────────
    anthropic_api_key: str = Field(alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(alias="OPENAI_API_KEY")

    # ── Model selection ───────────────────────────────────────────────────────
    claude_model: str = Field(
        default="claude-opus-4-5",
        alias="CLAUDE_MODEL",
        description="Anthropic model used for routing, synthesis, and answers",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="EMBEDDING_MODEL",
    )

    # ── Pipeline tuning ───────────────────────────────────────────────────────
    query_top_k: int = Field(
        default=5,
        alias="QUERY_TOP_K",
        ge=1,
        le=20,
        description="Number of wiki pages retrieved per query",
    )
    max_source_chars: int = Field(
        default=50_000,
        alias="MAX_SOURCE_CHARS",
        description="Truncation limit for very long source documents",
    )


class WikiPaths:
    """All filesystem paths derived from a single root directory.

    Centralising path logic here means tests can pass any tmp_path
    as root and the entire path graph shifts with it — no globals.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path.cwd()

    @property
    def wiki_dir(self) -> Path:
        return self.root / "wiki"

    @property
    def sources_dir(self) -> Path:
        return self.root / "sources"

    @property
    def meta_dir(self) -> Path:
        return self.wiki_dir / ".meta"

    @property
    def schema_path(self) -> Path:
        return self.meta_dir / "schema.json"

    @property
    def embeddings_path(self) -> Path:
        return self.meta_dir / "embeddings.json"

    @property
    def index_path(self) -> Path:
        return self.wiki_dir / "index.md"

    @property
    def log_path(self) -> Path:
        return self.wiki_dir / "log.md"

    def page_path(self, slug: str) -> Path:
        return self.wiki_dir / f"{slug}.md"
```

---

## Phase 3 — Prompts

### Step 3.1 — `tests/test_prompts.py`

```python
"""Tests for prompt construction and the QUERY_TEMPLATES registry."""

from __future__ import annotations

import pytest

from llm_wiki.prompts import (
    QUERY_TEMPLATES,
    QueryTemplate,
    build_answer_prompt,
    build_routing_prompt,
    build_synthesis_prompt,
    get_template,
)


class TestRoutingPrompt:
    def test_routing_prompt_contains_schema_context(self) -> None:
        context = "a: Topic A — About A\nb: Topic B — About B"
        source_text = "Some source content."
        prompt = build_routing_prompt(schema_context=context, source_text=source_text)
        assert "a: Topic A" in prompt
        assert source_text in prompt

    def test_routing_prompt_requests_json_array(self) -> None:
        prompt = build_routing_prompt(schema_context="a: A — d", source_text="text")
        assert "JSON" in prompt or "json" in prompt


class TestSynthesisPrompt:
    def test_synthesis_prompt_includes_existing_body(self) -> None:
        prompt = build_synthesis_prompt(
            slug="transformers",
            title="Transformers",
            existing_body="## Existing content",
            source_text="New information.",
        )
        assert "## Existing content" in prompt
        assert "New information." in prompt

    def test_synthesis_prompt_enforces_preservation_invariant(self) -> None:
        prompt = build_synthesis_prompt(
            slug="t", title="T", existing_body="old", source_text="new"
        )
        # Must instruct the LLM never to discard existing knowledge
        assert "preserve" in prompt.lower() or "never discard" in prompt.lower()


class TestAnswerPrompt:
    def test_answer_prompt_includes_question_and_context(self) -> None:
        prompt = build_answer_prompt(
            question="What is attention?",
            wiki_context="## Attention Mechanism\n\nContent here.",
        )
        assert "What is attention?" in prompt
        assert "## Attention Mechanism" in prompt


class TestQueryTemplates:
    def test_all_template_categories_present(self) -> None:
        categories = {t.category for t in QUERY_TEMPLATES.values()}
        expected = {"synthesis", "gap-finding", "debate", "output", "health", "personal"}
        assert expected.issubset(categories)

    def test_each_template_has_required_fields(self) -> None:
        for name, template in QUERY_TEMPLATES.items():
            assert template.name == name
            assert len(template.description) > 0
            assert len(template.prompt) > 0
            assert template.category in {
                "synthesis", "gap-finding", "debate", "output", "health", "personal"
            }

    def test_get_template_returns_correct_template(self) -> None:
        t = get_template("master-summary")
        assert t is not None
        assert t.name == "master-summary"

    def test_get_template_returns_none_for_unknown(self) -> None:
        assert get_template("does-not-exist") is None

    def test_minimum_24_templates_defined(self) -> None:
        assert len(QUERY_TEMPLATES) >= 24
```

### Step 3.2 — `src/llm_wiki/prompts.py`

```python
"""All prompt text and query templates — the single source of truth.

No prompt text lives anywhere else in the codebase. This module is
the contract between what we ask the LLM to do and how it does it.
"""

from __future__ import annotations

from dataclasses import dataclass


# ── Prompt builders ────────────────────────────────────────────────────────────

def build_routing_prompt(schema_context: str, source_text: str) -> str:
    """Build the routing prompt that asks the LLM which pages are relevant.

    The LLM is given a compact schema summary and the source text.
    It returns a JSON array of slugs — the cheapest step in the pipeline.
    """
    return f"""\
You are a knowledge routing assistant. Your job is to decide which wiki pages
are genuinely relevant to the given source text.

Below is a list of all pages in the wiki. Each line has the format:
  slug: title — one-line description

WIKI SCHEMA:
{schema_context}

SOURCE TEXT:
{source_text}

Return ONLY a JSON array of the relevant page slugs. Include a slug only if the
source text contains meaningful information that would enrich that page.
Be selective — irrelevant pages should not be included.

Example response: ["attention_mechanism", "transformers"]

Return the JSON array only, with no other text or explanation."""


def build_synthesis_prompt(
    slug: str,
    title: str,
    existing_body: str,
    source_text: str,
) -> str:
    """Build the synthesis prompt that rewrites a wiki page.

    CRITICAL INVARIANT: The LLM must preserve all existing knowledge
    while integrating the new material. Knowledge compounds — it never
    overwrites or discards what was already there.
    """
    existing_section = (
        f"EXISTING PAGE CONTENT:\n{existing_body}"
        if existing_body.strip()
        else "EXISTING PAGE CONTENT:\n(This is a new page — no existing content yet.)"
    )
    return f"""\
You are maintaining a personal knowledge wiki. You must update the wiki page
for "{title}" (slug: {slug}) by integrating new information from the source.

CRITICAL RULES:
1. Preserve and EXTEND existing content — NEVER discard information already on the page.
2. Integrate new facts, expand explanations, and add [[slug]] cross-references
   to other related pages where appropriate.
3. Note any contradictions between new and existing information explicitly.
4. Use clear markdown formatting with headings and bullet points.
5. Do NOT include YAML front matter — return the markdown body only.

{existing_section}

NEW SOURCE TEXT:
{source_text}

Write the complete updated wiki page body (markdown only, no front matter):"""


def build_answer_prompt(question: str, wiki_context: str) -> str:
    """Build the query answer prompt using compiled wiki context."""
    return f"""\
You are answering a question using a compiled knowledge wiki.
The wiki pages below represent pre-synthesised, cross-referenced knowledge.

WIKI CONTEXT:
{wiki_context}

QUESTION: {question}

Answer based on the wiki content above. Where relevant, cite the page titles
you are drawing from using the format [Page Title]. Be clear and comprehensive."""


def build_contradiction_prompt(page_a_title: str, page_a_body: str,
                                page_b_title: str, page_b_body: str) -> str:
    """Build the deep lint contradiction-check prompt."""
    return f"""\
Compare the following two wiki pages and identify any factual contradictions.
A contradiction exists only when both pages make claims about the same topic
that cannot both be true.

PAGE 1: {page_a_title}
{page_a_body}

PAGE 2: {page_b_title}
{page_b_body}

List any genuine factual contradictions. If there are none, say so clearly.
Notational differences (different variable names for the same concept) are NOT
contradictions. Focus only on substantive factual disagreements."""


def build_index_update_prompt(
    schema_context: str, updated_pages: list[tuple[str, str, list[str]]]
) -> str:
    """Build the prompt to regenerate index.md after an ingest.

    updated_pages: list of (slug, title, tags) tuples.
    """
    page_list = "\n".join(
        f"- **{title}** (`{slug}`) — tags: {', '.join(tags)}"
        for slug, title, tags in updated_pages
    )
    return f"""\
Update the wiki index to reflect the current state of the knowledge base.

SCHEMA (all pages that should exist):
{schema_context}

RECENTLY UPDATED PAGES:
{page_list}

Generate a clean markdown index with a summary table. Format:
- Title row: | Slug | Title | Tags | Updated |
- One row per wiki page
- Keep descriptions concise (one line each)

Return the complete index.md content (markdown only):"""


# ── Query Templates ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class QueryTemplate:
    """A named, reusable query template for commissioning structured analysis."""

    name: str
    category: str  # synthesis | gap-finding | debate | output | health | personal
    description: str
    prompt: str


QUERY_TEMPLATES: dict[str, QueryTemplate] = {
    # ── Synthesis ──────────────────────────────────────────────────────────────
    "master-summary": QueryTemplate(
        name="master-summary",
        category="synthesis",
        description="The single most important insight that ties everything together",
        prompt=(
            "Read everything in the wiki and identify the single most important insight — "
            "the idea that ties all other concepts together. Explain why it is the keystone "
            "of this knowledge base, and what everything else follows from it."
        ),
    ),
    "concept-map": QueryTemplate(
        name="concept-map",
        category="synthesis",
        description="Map every major concept and how they connect",
        prompt=(
            "Identify every major concept in the wiki and describe how each concept relates "
            "to at least two others. Present this as a structured map showing dependencies, "
            "influences, and oppositions."
        ),
    ),
    "timeline": QueryTemplate(
        name="timeline",
        category="synthesis",
        description="Chronological development of ideas across the wiki",
        prompt=(
            "Reconstruct the historical or logical timeline of ideas in this wiki. "
            "Which concepts came first? What did later ideas build on? "
            "Present as a narrative timeline with approximate dates where known."
        ),
    ),
    "first-principles": QueryTemplate(
        name="first-principles",
        category="synthesis",
        description="Reduce everything to first principles",
        prompt=(
            "What are the fundamental axioms or first principles that underlie everything "
            "in this wiki? If you had to rebuild this knowledge base from scratch, "
            "what would be the minimal starting assumptions?"
        ),
    ),
    # ── Gap-Finding ────────────────────────────────────────────────────────────
    "blind-spot": QueryTemplate(
        name="blind-spot",
        category="gap-finding",
        description="Important topics missing from the wiki",
        prompt=(
            "What important topics are entirely absent from this wiki? "
            "Based on the concepts that ARE present, what closely related areas "
            "are conspicuously missing? List the top 5 gaps with brief explanations "
            "of why each matters."
        ),
    ),
    "weak-links": QueryTemplate(
        name="weak-links",
        category="gap-finding",
        description="Concepts with insufficient depth or sources",
        prompt=(
            "Which wiki pages have insufficient depth relative to their importance? "
            "Which concepts are referenced frequently in cross-links but have thin content? "
            "Rank by urgency for improvement."
        ),
    ),
    "source-gaps": QueryTemplate(
        name="source-gaps",
        category="gap-finding",
        description="Areas where more source material is needed",
        prompt=(
            "Based on the current state of the wiki, what types of source material "
            "would most improve the knowledge base? Be specific: name topics, "
            "suggest source types (papers, books, talks), and explain the expected impact."
        ),
    ),
    "stale-claims": QueryTemplate(
        name="stale-claims",
        category="gap-finding",
        description="Claims likely to have been superseded by newer developments",
        prompt=(
            "Which claims in the wiki are most likely to have been superseded or "
            "updated by recent developments? Flag any claims that include dates, "
            "benchmarks, or state-of-the-art assertions that may now be outdated."
        ),
    ),
    # ── Debate ─────────────────────────────────────────────────────────────────
    "biggest-disagreement": QueryTemplate(
        name="biggest-disagreement",
        category="debate",
        description="The biggest tension or disagreement between sources",
        prompt=(
            "What is the single biggest disagreement between the sources in this wiki? "
            "Steelman both sides — present each position as its strongest advocates would. "
            "Then give your assessment of which side has the stronger argument."
        ),
    ),
    "tradeoffs": QueryTemplate(
        name="tradeoffs",
        category="debate",
        description="The most important tradeoffs in this domain",
        prompt=(
            "What are the most important tradeoffs documented in this wiki? "
            "For each tradeoff, describe what you gain and what you lose, "
            "and what conditions determine the right choice."
        ),
    ),
    "competing-paradigms": QueryTemplate(
        name="competing-paradigms",
        category="debate",
        description="Different paradigms or schools of thought in the wiki",
        prompt=(
            "Are there competing paradigms, frameworks, or schools of thought in this wiki? "
            "Describe each paradigm, its core assumptions, its strengths and limitations, "
            "and which contexts favour each one."
        ),
    ),
    # ── Output ─────────────────────────────────────────────────────────────────
    "study-guide": QueryTemplate(
        name="study-guide",
        category="output",
        description="A structured study guide for mastering this knowledge base",
        prompt=(
            "Create a comprehensive study guide for mastering everything in this wiki. "
            "Structure it as: (1) Prerequisites, (2) Core concepts in learning order, "
            "(3) Key relationships to understand, (4) Common misconceptions to avoid, "
            "(5) Suggested exercises or projects."
        ),
    ),
    "cheat-sheet": QueryTemplate(
        name="cheat-sheet",
        category="output",
        description="A compact reference cheat sheet",
        prompt=(
            "Create a concise cheat sheet covering the most important facts, formulas, "
            "definitions, and rules of thumb from this wiki. "
            "Format for quick reference: use tables and bullet points. "
            "Aim for one dense page."
        ),
    ),
    "faq": QueryTemplate(
        name="faq",
        category="output",
        description="Anticipated frequently asked questions and answers",
        prompt=(
            "Generate a FAQ document from the wiki's content. "
            "Anticipate the 10 most likely questions a new learner would ask, "
            "and answer each based on the wiki's compiled knowledge. "
            "Order from foundational to advanced."
        ),
    ),
    "slide-outline": QueryTemplate(
        name="slide-outline",
        category="output",
        description="A slide deck outline for presenting this knowledge",
        prompt=(
            "Create a slide deck outline for a 20-minute presentation on the core "
            "topics of this wiki. Include: title slide, agenda, ~12 content slides "
            "with bullet points, and a conclusion. Keep each slide focused on one concept."
        ),
    ),
    "executive-summary": QueryTemplate(
        name="executive-summary",
        category="output",
        description="A non-technical executive summary of the wiki",
        prompt=(
            "Write a one-page executive summary of this knowledge base for a non-technical "
            "audience. Focus on: what this domain is about, why it matters, the key ideas "
            "in plain language, and the most important practical implications."
        ),
    ),
    # ── Health ─────────────────────────────────────────────────────────────────
    "integrity-report": QueryTemplate(
        name="integrity-report",
        category="health",
        description="Audit the wiki for internal consistency",
        prompt=(
            "Audit the wiki for internal consistency. Look for: "
            "(1) pages that define the same concept differently, "
            "(2) cross-references that seem logically inverted, "
            "(3) claims made on one page that are contradicted elsewhere, "
            "(4) pages whose scope significantly overlaps. "
            "Report findings with specific page references."
        ),
    ),
    "duplication-check": QueryTemplate(
        name="duplication-check",
        category="health",
        description="Identify overlapping or duplicated content",
        prompt=(
            "Which wiki pages cover substantially overlapping content? "
            "For each pair of pages with significant overlap, explain what is duplicated "
            "and suggest whether they should be merged, one should subsume the other, "
            "or the overlap should be resolved with a clear scope distinction."
        ),
    ),
    "provenance-audit": QueryTemplate(
        name="provenance-audit",
        category="health",
        description="Identify claims that lack source attribution",
        prompt=(
            "Review the wiki for claims that appear to lack source attribution. "
            "Which pages contain assertions that read as confident but are not "
            "traceable back to any source in the provenance metadata? "
            "Flag the most significant unsourced claims."
        ),
    ),
    # ── Personal application ───────────────────────────────────────────────────
    "unknown-unknowns": QueryTemplate(
        name="unknown-unknowns",
        category="personal",
        description="Likely mistakes or blind spots given this knowledge base",
        prompt=(
            "Based on everything in this wiki, what are the most likely mistakes "
            "I am currently making — even if I don't know it yet? "
            "What are the most dangerous assumptions embedded in this knowledge base "
            "that I am probably not questioning?"
        ),
    ),
    "next-steps": QueryTemplate(
        name="next-steps",
        category="personal",
        description="Recommended next actions based on this knowledge",
        prompt=(
            "Given everything in this wiki, what are the top 5 most valuable "
            "actions I could take right now? Prioritise by impact and feasibility. "
            "Explain the reasoning behind each recommendation."
        ),
    ),
    "decision-framework": QueryTemplate(
        name="decision-framework",
        category="personal",
        description="A decision framework derived from the wiki's knowledge",
        prompt=(
            "Synthesise a decision framework from this wiki's knowledge. "
            "When I face a decision in this domain, what questions should I ask? "
            "What are the key variables to evaluate? Present as a structured process."
        ),
    ),
    "teaching-script": QueryTemplate(
        name="teaching-script",
        category="personal",
        description="How I would teach this to someone else",
        prompt=(
            "If I had to teach the core ideas from this wiki to a smart colleague "
            "who knows nothing about this domain, how would I do it? "
            "Write a teaching script: what order, what analogies, what examples, "
            "what common misconceptions to address first."
        ),
    ),
    "progress-reflection": QueryTemplate(
        name="progress-reflection",
        category="personal",
        description="What this wiki reveals about my learning progress",
        prompt=(
            "Based on the content and gaps in this wiki, what does it reveal about "
            "where I am in my understanding of this domain? "
            "What concepts am I clearly confident in? Where is my understanding thin? "
            "What would a true expert notice missing?"
        ),
    ),
}


def get_template(name: str) -> QueryTemplate | None:
    """Look up a query template by name."""
    return QUERY_TEMPLATES.get(name)
```

---

## Phase 4 — Core Infrastructure

### Step 4.1 — `tests/test_embeddings_module.py`

```python
"""Tests for EmbeddingService."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from llm_wiki.embeddings import EmbeddingService
from llm_wiki.models.embeddings import EMBEDDING_DIM


def _make_raw_vector(seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(EMBEDDING_DIM).tolist()


class TestEmbeddingService:
    def _make_service(self, raw_vector: list[float] | None = None) -> EmbeddingService:
        """Return an EmbeddingService backed by a mock OpenAI client."""
        mock_client = MagicMock()
        vec = raw_vector or _make_raw_vector()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=vec)]
        mock_client.embeddings.create.return_value = mock_response
        return EmbeddingService(client=mock_client)

    def test_embed_returns_normalised_vector(self) -> None:
        svc = self._make_service()
        result = svc.embed("Some text to embed")
        arr = np.array(result)
        # Normalised vectors have unit length
        assert abs(np.linalg.norm(arr) - 1.0) < 1e-5

    def test_embed_returns_correct_dimension(self) -> None:
        svc = self._make_service()
        result = svc.embed("text")
        assert len(result) == EMBEDDING_DIM

    def test_embed_page_returns_page_embedding(self) -> None:
        svc = self._make_service()
        emb = svc.embed_page(slug="transformers", text="Transformer architecture")
        assert emb.slug == "transformers"
        assert len(emb.vector) == EMBEDDING_DIM

    def test_embed_truncates_very_long_text(self) -> None:
        mock_client = MagicMock()
        raw_vec = _make_raw_vector()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=raw_vec)]
        mock_client.embeddings.create.return_value = mock_response
        svc = EmbeddingService(client=mock_client)

        long_text = "x" * 100_000
        svc.embed(long_text)

        call_kwargs = mock_client.embeddings.create.call_args
        passed_input = call_kwargs.kwargs.get("input") or call_kwargs.args[1]
        # Should be truncated to ~32k chars (8191 tokens * ~4 chars/token)
        assert len(passed_input) <= 32_768
```

### Step 4.2 — `src/llm_wiki/embeddings.py`

```python
"""OpenAI embedding wrapper.

Wraps text-embedding-3-small with normalisation and truncation.
Isolated here so swapping to a local model only touches this file.
"""

from __future__ import annotations

import numpy as np
from openai import OpenAI

from llm_wiki.models.embeddings import EMBEDDING_DIM, EMBEDDING_MODEL, PageEmbedding

# text-embedding-3-small max input tokens; 4 chars/token is a safe approximation
_MAX_INPUT_CHARS = 8191 * 4


class EmbeddingService:
    """Thin wrapper around the OpenAI embeddings API.

    Responsibilities:
    - Truncate input to model limits
    - Normalise the returned vector to unit length
    - Return typed PageEmbedding objects
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = EMBEDDING_MODEL,
    ) -> None:
        self._client = client
        self._model = model

    def embed(self, text: str) -> list[float]:
        """Embed text and return a normalised float32 vector."""
        truncated = text[:_MAX_INPUT_CHARS]
        response = self._client.embeddings.create(
            model=self._model,
            input=truncated,
        )
        raw: list[float] = response.data[0].embedding
        arr = np.array(raw, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    def embed_page(self, slug: str, text: str) -> PageEmbedding:
        """Embed a wiki page and return a validated PageEmbedding."""
        vector = self.embed(text)
        return PageEmbedding(slug=slug, vector=vector, model=self._model)
```

### Step 4.3 — `tests/test_index.py`

```python
"""Tests for EmbeddingIndex."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from llm_wiki.index import EmbeddingIndex
from llm_wiki.models.embeddings import EMBEDDING_DIM, PageEmbedding


def _unit_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


class TestEmbeddingIndex:
    def test_empty_index_initialises_from_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "embeddings.json"
        idx = EmbeddingIndex(path)
        assert idx.slugs() == []

    def test_upsert_and_retrieve_slug(self, tmp_path: Path) -> None:
        path = tmp_path / "embeddings.json"
        idx = EmbeddingIndex(path)
        emb = PageEmbedding(slug="a", vector=_unit_vec(0))
        idx.upsert(emb)
        assert idx.contains("a")

    def test_save_and_reload_persists_data(self, tmp_path: Path) -> None:
        path = tmp_path / "embeddings.json"
        idx = EmbeddingIndex(path)
        idx.upsert(PageEmbedding(slug="a", vector=_unit_vec(0)))
        idx.save()

        idx2 = EmbeddingIndex(path)
        assert idx2.contains("a")

    def test_top_k_returns_sorted_by_similarity(self, tmp_path: Path) -> None:
        path = tmp_path / "embeddings.json"
        idx = EmbeddingIndex(path)

        # Create two vectors: one close to query, one far
        query_vec = _unit_vec(0)
        close_vec = [v + 0.001 for v in query_vec]  # very similar
        far_arr = np.random.default_rng(999).standard_normal(EMBEDDING_DIM).astype(np.float32)
        far_vec = (far_arr / np.linalg.norm(far_arr)).tolist()

        idx.upsert(PageEmbedding(slug="close", vector=_unit_vec(0)))
        idx.upsert(PageEmbedding(slug="far", vector=far_vec))

        results = idx.top_k(query_vec, k=2)
        assert results[0][0] == "close"
        assert results[0][1] > results[1][1]

    def test_top_k_respects_k_limit(self, tmp_path: Path) -> None:
        path = tmp_path / "embeddings.json"
        idx = EmbeddingIndex(path)
        for i in range(10):
            idx.upsert(PageEmbedding(slug=f"p{i}", vector=_unit_vec(i)))
        results = idx.top_k(_unit_vec(0), k=3)
        assert len(results) == 3

    def test_top_k_on_empty_index_returns_empty_list(self, tmp_path: Path) -> None:
        path = tmp_path / "embeddings.json"
        idx = EmbeddingIndex(path)
        results = idx.top_k(_unit_vec(0))
        assert results == []

    def test_contains_returns_false_for_missing_slug(self, tmp_path: Path) -> None:
        path = tmp_path / "embeddings.json"
        idx = EmbeddingIndex(path)
        assert not idx.contains("ghost")
```

### Step 4.4 — `src/llm_wiki/index.py`

```python
"""Embedding index — cosine similarity search over wiki pages.

Uses a flat JSON file with NumPy linear scan. This is fast and
requires no additional infrastructure for wikis up to ~500 pages.
Beyond that, swap top_k() to use FAISS without changing callers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from llm_wiki.models.embeddings import EmbeddingIndexData, PageEmbedding


class EmbeddingIndex:
    """Persistent embedding index backed by a JSON file.

    The index is loaded into memory on construction and written back
    on explicit save() calls. This avoids file I/O on every search.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: EmbeddingIndexData = self._load()

    def _load(self) -> EmbeddingIndexData:
        if self._path.exists() and self._path.stat().st_size > 0:
            return EmbeddingIndexData.model_validate_json(self._path.read_text())
        return EmbeddingIndexData()

    def save(self) -> None:
        """Persist the current index to disk."""
        self._path.write_text(self._data.model_dump_json(indent=2))

    def upsert(self, embedding: PageEmbedding) -> None:
        """Insert or overwrite a page embedding in memory."""
        self._data.embeddings[embedding.slug] = embedding

    def top_k(
        self, query_vector: list[float], k: int = 5
    ) -> list[tuple[str, float]]:
        """Return (slug, cosine_similarity) pairs, sorted descending.

        Vectors are assumed to be pre-normalised (unit length), so
        cosine similarity reduces to a dot product — fast with NumPy.
        """
        if not self._data.embeddings:
            return []

        q = np.array(query_vector, dtype=np.float32)
        scores: list[tuple[str, float]] = [
            (slug, float(np.dot(q, emb.as_array())))
            for slug, emb in self._data.embeddings.items()
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def slugs(self) -> list[str]:
        return list(self._data.embeddings.keys())

    def contains(self, slug: str) -> bool:
        return slug in self._data.embeddings

    def remove(self, slug: str) -> None:
        """Remove a page from the index (used when a page is deleted)."""
        self._data.embeddings.pop(slug, None)

    @property
    def size(self) -> int:
        return self._data.slug_count
```

### Step 4.5 — `tests/test_wiki.py`

```python
"""Tests for WikiPage and WikiRepository."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from llm_wiki.models.frontmatter import PageSpec, WikiFrontMatter, WikiSchema
from llm_wiki.wiki import WikiPage, WikiRepository


class TestWikiRepository:
    def test_read_returns_none_for_missing_page(self, tmp_wiki_dir: Path) -> None:
        repo = WikiRepository(tmp_wiki_dir / "wiki")
        assert repo.read("nonexistent") is None

    def test_write_creates_readable_page(self, tmp_wiki_dir: Path) -> None:
        repo = WikiRepository(tmp_wiki_dir / "wiki")
        repo.write(
            slug="topic_a",
            title="Topic A",
            body="## Overview\n\nContent here.",
            sources=["source.md"],
            tags=["test"],
        )
        page = repo.read("topic_a")
        assert page is not None
        assert page.front_matter.slug == "topic_a"
        assert "Content here." in page.body

    def test_write_persists_front_matter_correctly(self, tmp_wiki_dir: Path) -> None:
        repo = WikiRepository(tmp_wiki_dir / "wiki")
        repo.write(
            slug="topic_a",
            title="Topic A",
            body="Body text.",
            sources=["paper.txt", "notes.md"],
            tags=["ml", "test"],
        )
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
        repo = WikiRepository(tmp_wiki_dir / "wiki")
        slugs = repo.list_slugs()
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
        repo = WikiRepository(tmp_wiki_dir / "wiki")
        schema = repo.load_schema()
        assert isinstance(schema, WikiSchema)
        assert len(schema.pages) == 3  # from conftest sample_schema

    def test_append_log_writes_entry(self, tmp_wiki_dir: Path) -> None:
        repo = WikiRepository(tmp_wiki_dir / "wiki")
        repo.append_log("- 2026-04-14 — Test entry")
        log_content = (tmp_wiki_dir / "wiki" / "log.md").read_text()
        assert "Test entry" in log_content

    def test_full_text_property_includes_title(self, tmp_wiki_dir: Path) -> None:
        repo = WikiRepository(tmp_wiki_dir / "wiki")
        repo.write(slug="topic_a", title="My Title", body="Body.", sources=[], tags=[])
        page = repo.read("topic_a")
        assert page is not None
        assert "My Title" in page.full_text
```

### Step 4.6 — `src/llm_wiki/wiki.py`

```python
"""WikiPage and WikiRepository — CRUD for the wiki's markdown files.

The repository is the only place that touches wiki/*.md files.
All reads validate front matter via Pydantic; all writes go through
the same validated path, ensuring the LLM can never produce a
structurally malformed page.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import frontmatter

from llm_wiki.models.frontmatter import WikiFrontMatter, WikiSchema


class WikiPage:
    """In-memory representation of a single wiki page."""

    def __init__(self, path: Path, front_matter: WikiFrontMatter, body: str) -> None:
        self.path = path
        self.front_matter = front_matter
        self.body = body

    @property
    def full_text(self) -> str:
        """Title + body — the text used for embedding and query context."""
        return f"# {self.front_matter.title}\n\n{self.body}"

    def save(self) -> None:
        """Serialise to disk with validated YAML front matter."""
        fm_data = self.front_matter.model_dump(mode="json")
        post = frontmatter.Post(self.body, **fm_data)
        self.path.write_text(frontmatter.dumps(post))


class WikiRepository:
    """All filesystem operations for wiki pages.

    The single point of entry for reading and writing .md files.
    Tests inject tmp_path here to isolate all file I/O.
    """

    def __init__(self, wiki_dir: Path) -> None:
        self._dir = wiki_dir
        self._meta_dir = wiki_dir / ".meta"

    def _page_path(self, slug: str) -> Path:
        return self._dir / f"{slug}.md"

    # ── Schema ─────────────────────────────────────────────────────────────────

    def load_schema(self) -> WikiSchema:
        """Load and validate the schema from disk."""
        return WikiSchema.model_validate_json(
            (self._meta_dir / "schema.json").read_text()
        )

    def save_schema(self, schema: WikiSchema) -> None:
        (self._meta_dir / "schema.json").write_text(schema.model_dump_json(indent=2))

    # ── Pages ──────────────────────────────────────────────────────────────────

    def read(self, slug: str) -> WikiPage | None:
        """Read a page from disk, validating its front matter on load."""
        path = self._page_path(slug)
        if not path.exists():
            return None
        post = frontmatter.load(str(path))
        fm = WikiFrontMatter.model_validate(dict(post.metadata))
        return WikiPage(path=path, front_matter=fm, body=post.content)

    def write(
        self,
        slug: str,
        title: str,
        body: str,
        sources: list[str],
        tags: list[str],
    ) -> WikiPage:
        """Create or overwrite a wiki page with validated front matter."""
        fm = WikiFrontMatter(
            slug=slug,
            title=title,
            tags=tags,
            sources=sources,
            updated=date.today(),
        )
        page = WikiPage(path=self._page_path(slug), front_matter=fm, body=body)
        page.save()
        return page

    def list_slugs(self) -> list[str]:
        """Return slugs of all real wiki pages (excludes index and log)."""
        return [
            p.stem for p in self._dir.glob("*.md")
            if p.stem not in ("index", "log")
        ]

    # ── Index and Log ──────────────────────────────────────────────────────────

    def write_index(self, content: str) -> None:
        (self._dir / "index.md").write_text(content)

    def append_log(self, entry: str) -> None:
        """Append a timestamped entry to the append-only log."""
        log_path = self._dir / "log.md"
        with log_path.open("a") as f:
            f.write(f"\n{entry}")

    def read_log(self) -> str:
        return (self._dir / "log.md").read_text()
```

---

## Phase 5 — Pipeline Modules

### Step 5.1 — `tests/test_ingest.py`

```python
"""Tests for the 5-step ingest pipeline and resolve_source()."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_wiki.ingest import resolve_source, route, run_ingest, synthesize
from llm_wiki.models.operations import RouteResult, SynthesisResult
from llm_wiki.models.source import SourceType


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
    def _make_anthropic_client(self, slugs: list[str]) -> MagicMock:
        client = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=json.dumps(slugs))]
        client.messages.create.return_value = mock_msg
        return client

    def test_route_returns_valid_slugs(self) -> None:
        from llm_wiki.models.source import ResolvedSource, SourceType
        client = self._make_anthropic_client(["topic_a", "topic_b"])
        source = ResolvedSource(
            raw="f.md", source_type=SourceType.LOCAL_FILE, name="f.md", text="text"
        )
        result = route(client=client, source=source, schema_context="context", model="claude-test")
        assert isinstance(result, RouteResult)
        assert "topic_a" in result.relevant_slugs

    def test_route_handles_malformed_llm_response(self) -> None:
        from llm_wiki.models.source import ResolvedSource, SourceType
        client = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="not valid json")]
        client.messages.create.return_value = mock_msg
        source = ResolvedSource(
            raw="f.md", source_type=SourceType.LOCAL_FILE, name="f.md", text="text"
        )
        result = route(client=client, source=source, schema_context="ctx", model="m")
        # Should not raise; malformed response → empty slugs
        assert result.relevant_slugs == []


class TestRunIngest:
    def test_full_ingest_pipeline_writes_pages(
        self, tmp_wiki_dir: Path, mocker: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki.config import WikiPaths, WikiSettings
        from llm_wiki.embeddings import EmbeddingService
        from llm_wiki.index import EmbeddingIndex
        from llm_wiki.wiki import WikiRepository

        # Write a source file
        source_file = tmp_wiki_dir / "sources" / "test.md"
        source_file.write_text("# Test\nContent about topic A.")

        repo = WikiRepository(tmp_wiki_dir / "wiki")
        index = EmbeddingIndex(tmp_wiki_dir / "wiki" / ".meta" / "embeddings.json")

        mock_embed_svc = MagicMock(spec=EmbeddingService)
        mock_embed_svc.embed.return_value = [0.0] * 1536
        # Return a valid PageEmbedding
        from llm_wiki.models.embeddings import PageEmbedding
        import numpy as np
        rng = np.random.default_rng(0)
        vec = rng.standard_normal(1536).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        mock_embed_svc.embed_page.return_value = PageEmbedding(slug="topic_a", vector=vec.tolist())

        mock_anthropic = MagicMock()
        # Route response
        route_msg = MagicMock()
        route_msg.content = [MagicMock(text='["topic_a"]')]
        # Synthesis response
        synth_msg = MagicMock()
        synth_msg.content = [MagicMock(text="## Topic A\n\nSynthesised content.")]
        # Index update response
        index_msg = MagicMock()
        index_msg.content = [MagicMock(text="# Index\n\n| topic_a | Topic A |")]
        mock_anthropic.messages.create.side_effect = [route_msg, synth_msg, index_msg]

        from llm_wiki.config import WikiSettings
        import os
        os.environ.setdefault("ANTHROPIC_API_KEY", "x")
        os.environ.setdefault("OPENAI_API_KEY", "x")
        settings = WikiSettings()

        result = run_ingest(
            source_path=str(source_file),
            repo=repo,
            index=index,
            embedding_service=mock_embed_svc,
            anthropic_client=mock_anthropic,
            settings=settings,
        )
        assert result.success
        assert "topic_a" in result.slugs_touched
```

### Step 5.2 — `src/llm_wiki/ingest.py`

```python
"""Five-step ingest pipeline: Resolve → Route → Synthesize → Embed → Update.

This is the heart of the LLM Wiki pattern. Each ingest operation
compiles new source knowledge into the wiki permanently — it doesn't
start from scratch on the next query.

Step 0: resolve_source()  — detect source type and extract text
Step 1: route()           — ask LLM which slugs are relevant
Step 2: synthesize()      — rewrite each relevant page
Step 3: embed             — re-embed each updated page
Step 4: update index/log  — keep catalog and audit trail current
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
from anthropic import Anthropic
from youtube_transcript_api import YouTubeTranscriptApi

from llm_wiki.config import WikiSettings
from llm_wiki.embeddings import EmbeddingService
from llm_wiki.index import EmbeddingIndex
from llm_wiki.models.operations import IngestResult, RouteResult, SynthesisResult
from llm_wiki.models.source import ResolvedSource, SourceType
from llm_wiki.prompts import (
    build_index_update_prompt,
    build_routing_prompt,
    build_synthesis_prompt,
)
from llm_wiki.wiki import WikiRepository

# ── Custom exceptions ──────────────────────────────────────────────────────────

class IngestError(Exception):
    """Raised when the ingest pipeline encounters an unrecoverable error."""


# ── Step 0: Source resolution ──────────────────────────────────────────────────

def _extract_youtube_id(url: str) -> str | None:
    """Extract the video ID from a YouTube URL."""
    parsed = urlparse(url)
    if "youtube.com" in parsed.netloc:
        qs = parse_qs(parsed.query)
        ids = qs.get("v", [])
        return ids[0] if ids else None
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")
    return None


def _strip_html(html: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def resolve_source(source_path: str, max_chars: int = 50_000) -> ResolvedSource:
    """Step 0: Detect source type and extract plain text.

    Handles local files (txt, md, pdf), YouTube URLs, and HTTP URLs.
    Returns a ResolvedSource with the extracted text truncated to max_chars.
    """
    path = source_path.strip()

    # ── YouTube ────────────────────────────────────────────────────────────────
    if "youtube.com" in path or "youtu.be" in path:
        video_id = _extract_youtube_id(path)
        if not video_id:
            raise IngestError(f"Could not extract YouTube video ID from: {path!r}")
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join(item["text"] for item in transcript_data)
        return ResolvedSource(
            raw=path,
            source_type=SourceType.YOUTUBE,
            name=f"youtube:{video_id}",
            text=text[:max_chars],
        )

    # ── HTTP URL ───────────────────────────────────────────────────────────────
    if path.startswith("http://") or path.startswith("https://"):
        response = httpx.get(path, follow_redirects=True, timeout=30.0)
        response.raise_for_status()
        text = _strip_html(response.text)
        domain = urlparse(path).netloc
        return ResolvedSource(
            raw=path,
            source_type=SourceType.HTTP,
            name=domain,
            text=text[:max_chars],
        )

    # ── Local file ─────────────────────────────────────────────────────────────
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Source file not found: {path!r}")

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        # Covers .md, .txt, and any other plain-text format
        text = file_path.read_text(encoding="utf-8", errors="replace")

    return ResolvedSource(
        raw=path,
        source_type=SourceType.LOCAL_FILE,
        name=file_path.name,
        text=text[:max_chars],
    )


# ── Step 1: Routing ────────────────────────────────────────────────────────────

def route(
    client: Anthropic,
    source: ResolvedSource,
    schema_context: str,
    model: str,
) -> RouteResult:
    """Step 1: Ask the LLM which pages are relevant to this source.

    Returns a RouteResult with a (possibly empty) list of slugs.
    Malformed LLM responses are caught and returned as an empty list —
    never raising, so the pipeline can report and move on.
    """
    prompt = build_routing_prompt(schema_context=schema_context, source_text=source.text)
    response = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text: str = response.content[0].text.strip()
    try:
        # LLM should return a plain JSON array
        slugs: list[str] = json.loads(raw_text)
    except (json.JSONDecodeError, ValueError):
        slugs = []

    return RouteResult(source_name=source.name, relevant_slugs=slugs)


# ── Step 2: Synthesis ──────────────────────────────────────────────────────────

def synthesize(
    client: Anthropic,
    slug: str,
    title: str,
    existing_body: str,
    source: ResolvedSource,
    model: str,
) -> SynthesisResult:
    """Step 2: Rewrite a wiki page, integrating knowledge from the source.

    The synthesis prompt enforces the preservation invariant:
    existing knowledge is always kept; new knowledge is added.
    """
    prompt = build_synthesis_prompt(
        slug=slug,
        title=title,
        existing_body=existing_body,
        source_text=source.text,
    )
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    new_body: str = response.content[0].text.strip()
    return SynthesisResult(slug=slug, new_body=new_body)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_ingest(
    source_path: str,
    repo: WikiRepository,
    index: EmbeddingIndex,
    embedding_service: EmbeddingService,
    anthropic_client: Anthropic,
    settings: WikiSettings,
) -> IngestResult:
    """Run the full 5-step ingest pipeline for one source.

    Designed for dependency injection — callers (CLI, tests) provide
    all dependencies. No global state is accessed here.
    """
    try:
        # Step 0: Resolve source
        source = resolve_source(source_path, max_chars=settings.max_source_chars)

        # Load schema once — shared across routing and synthesis
        schema = repo.load_schema()
        schema_context = schema.routing_context()

        # Step 1: Route
        route_result = route(
            client=anthropic_client,
            source=source,
            schema_context=schema_context,
            model=settings.claude_model,
        )

        # Filter slugs to those that actually exist in the schema
        valid_slugs = [s for s in route_result.relevant_slugs if schema.get_page(s)]
        touched_slugs: list[str] = []

        # Steps 2 & 3: Synthesize each relevant page, then embed
        for slug in valid_slugs:
            page_spec = schema.get_page(slug)
            assert page_spec is not None  # filtered above

            existing = repo.read(slug)
            existing_body = existing.body if existing else ""
            existing_sources = existing.front_matter.sources if existing else []
            existing_tags = existing.front_matter.tags if existing else []

            synth = synthesize(
                client=anthropic_client,
                slug=slug,
                title=page_spec.title,
                existing_body=existing_body,
                source=source,
                model=settings.claude_model,
            )

            # Merge sources without duplicates
            merged_sources = list(dict.fromkeys([*existing_sources, source.name]))
            repo.write(
                slug=slug,
                title=page_spec.title,
                body=synth.new_body,
                sources=merged_sources,
                tags=existing_tags,
            )

            # Step 3: Embed
            updated_page = repo.read(slug)
            if updated_page:
                embedding = embedding_service.embed_page(slug, updated_page.full_text)
                index.upsert(embedding)

            touched_slugs.append(slug)

        # Save the updated index to disk
        index.save()

        # Step 4: Update index.md
        all_pages = [
            (p.slug, p.title, repo.read(p.slug).front_matter.tags if repo.read(p.slug) else [])
            for p in schema.pages
        ]
        index_prompt = build_index_update_prompt(schema_context, all_pages)
        index_response = anthropic_client.messages.create(
            model=settings.claude_model,
            max_tokens=2048,
            messages=[{"role": "user", "content": index_prompt}],
        )
        repo.write_index(index_response.content[0].text)

        # Step 4 (cont.): Append to log.md
        result = IngestResult(source_name=source.name, slugs_touched=touched_slugs)
        repo.append_log(result.log_line())

        return result

    except (FileNotFoundError, IngestError) as exc:
        return IngestResult(
            source_name=source_path,
            slugs_touched=[],
            success=False,
            error=str(exc),
        )
```

### Step 5.3 — `tests/test_query.py`

```python
"""Tests for the 4-step query pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from llm_wiki.models.embeddings import EMBEDDING_DIM, PageEmbedding
from llm_wiki.models.operations import QueryResult


def _unit_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


class TestRunQuery:
    def test_query_returns_answer(self, tmp_wiki_dir: Path) -> None:
        from llm_wiki.config import WikiSettings
        from llm_wiki.embeddings import EmbeddingService
        from llm_wiki.index import EmbeddingIndex
        from llm_wiki.query import run_query
        from llm_wiki.wiki import WikiRepository
        import os
        os.environ.setdefault("ANTHROPIC_API_KEY", "x")
        os.environ.setdefault("OPENAI_API_KEY", "x")

        repo = WikiRepository(tmp_wiki_dir / "wiki")
        repo.write(slug="topic_a", title="Topic A", body="## Overview\n\nContent.", sources=[], tags=[])

        index = EmbeddingIndex(tmp_wiki_dir / "wiki" / ".meta" / "embeddings.json")
        emb = PageEmbedding(slug="topic_a", vector=_unit_vec(0))
        index.upsert(emb)
        index.save()

        mock_embed_svc = MagicMock(spec=EmbeddingService)
        mock_embed_svc.embed.return_value = _unit_vec(0)

        mock_anthropic = MagicMock()
        answer_msg = MagicMock()
        answer_msg.content = [MagicMock(text="Topic A is about...")]
        mock_anthropic.messages.create.return_value = answer_msg

        settings = WikiSettings()
        result = run_query(
            question="What is Topic A?",
            repo=repo,
            index=index,
            embedding_service=mock_embed_svc,
            anthropic_client=mock_anthropic,
            settings=settings,
        )
        assert isinstance(result, QueryResult)
        assert "Topic A" in result.answer or len(result.answer) > 0

    def test_query_with_save_creates_new_page(self, tmp_wiki_dir: Path) -> None:
        from llm_wiki.config import WikiSettings
        from llm_wiki.embeddings import EmbeddingService
        from llm_wiki.index import EmbeddingIndex
        from llm_wiki.query import run_query
        from llm_wiki.wiki import WikiRepository
        import os
        os.environ.setdefault("ANTHROPIC_API_KEY", "x")
        os.environ.setdefault("OPENAI_API_KEY", "x")

        repo = WikiRepository(tmp_wiki_dir / "wiki")
        repo.write(slug="topic_a", title="Topic A", body="Content.", sources=[], tags=[])

        index = EmbeddingIndex(tmp_wiki_dir / "wiki" / ".meta" / "embeddings.json")
        index.upsert(PageEmbedding(slug="topic_a", vector=_unit_vec(0)))
        index.save()

        mock_embed_svc = MagicMock(spec=EmbeddingService)
        mock_embed_svc.embed.return_value = _unit_vec(0)
        mock_embed_svc.embed_page.return_value = PageEmbedding(slug="saved_answer", vector=_unit_vec(1))

        mock_anthropic = MagicMock()
        answer_msg = MagicMock()
        answer_msg.content = [MagicMock(text="The answer is...")]
        mock_anthropic.messages.create.return_value = answer_msg

        settings = WikiSettings()
        result = run_query(
            question="What is the answer?",
            repo=repo,
            index=index,
            embedding_service=mock_embed_svc,
            anthropic_client=mock_anthropic,
            settings=settings,
            save=True,
        )
        assert result.saved_slug is not None
```

### Step 5.4 — `src/llm_wiki/query.py`

```python
"""Four-step query pipeline: Embed → Search → Assemble → Stream.

This is where the architectural bet on the LLM Wiki pays off.
Queries run over pre-synthesised, cross-referenced wiki pages —
not raw document chunks. The LLM reads a compiled encyclopedia
entry, not a random fragment from a PDF.

Optional --save: file a valuable answer back as a new wiki page,
closing the compounding loop.
"""

from __future__ import annotations

import re

from anthropic import Anthropic

from llm_wiki.config import WikiSettings
from llm_wiki.embeddings import EmbeddingService
from llm_wiki.index import EmbeddingIndex
from llm_wiki.models.operations import QueryResult, SimilarityHit
from llm_wiki.prompts import QUERY_TEMPLATES, build_answer_prompt, get_template
from llm_wiki.wiki import WikiRepository


def _question_to_slug(question: str) -> str:
    """Convert a question string to a valid page slug for --save."""
    slug = question.lower()
    slug = re.sub(r"[^a-z0-9\s]", "", slug)
    slug = re.sub(r"\s+", "_", slug.strip())
    return slug[:60]  # keep slugs manageable


def run_query(
    question: str,
    repo: WikiRepository,
    index: EmbeddingIndex,
    embedding_service: EmbeddingService,
    anthropic_client: Anthropic,
    settings: WikiSettings,
    save: bool = False,
    template: str | None = None,
    verbose: bool = False,
) -> QueryResult:
    """Run the full query pipeline and return a typed QueryResult.

    If template is provided, the corresponding named prompt is used
    instead of the question string.
    """
    # Resolve the actual question text (template overrides free-form question)
    if template:
        tmpl = get_template(template)
        if tmpl is None:
            raise ValueError(f"Unknown template: {template!r}. Run `llm-wiki prompts` to see options.")
        effective_question = tmpl.prompt
    else:
        effective_question = question

    # Step 1: Embed the question
    query_vector = embedding_service.embed(effective_question)

    # Step 2: Cosine similarity search
    hits_raw = index.top_k(query_vector, k=settings.query_top_k)

    # Step 3: Assemble context from retrieved pages
    hits: list[SimilarityHit] = []
    context_parts: list[str] = []

    for slug, score in hits_raw:
        page = repo.read(slug)
        if page is None:
            continue
        hits.append(SimilarityHit(slug=slug, title=page.front_matter.title, score=score))
        context_parts.append(f"## {page.front_matter.title}\n\n{page.body}")

    wiki_context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant pages found."

    # Step 4: Generate answer
    prompt = build_answer_prompt(question=effective_question, wiki_context=wiki_context)
    response = anthropic_client.messages.create(
        model=settings.claude_model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    answer: str = response.content[0].text

    # Optional: file the answer back as a new wiki page
    saved_slug: str | None = None
    if save and answer.strip():
        slug = _question_to_slug(question or template or "query_answer")
        repo.write(
            slug=slug,
            title=question[:80] if question else template or "Query Answer",
            body=answer,
            sources=["query"],
            tags=["query-answer"],
        )
        page = repo.read(slug)
        if page:
            emb = embedding_service.embed_page(slug, page.full_text)
            index.upsert(emb)
            index.save()
        saved_slug = slug

    return QueryResult(
        question=effective_question,
        hits=hits,
        answer=answer,
        saved_slug=saved_slug,
    )
```

### Step 5.5 — `tests/test_lint.py`

```python
"""Tests for the lint health checks."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from llm_wiki.models.embeddings import EMBEDDING_DIM, PageEmbedding
from llm_wiki.models.operations import LintIssueKind, LintResult


def _unit_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


class TestRunLint:
    def test_clean_wiki_passes_lint(self, tmp_wiki_dir: Path) -> None:
        from llm_wiki.embeddings import EmbeddingService
        from llm_wiki.index import EmbeddingIndex
        from llm_wiki.lint import run_lint
        from llm_wiki.wiki import WikiRepository

        repo = WikiRepository(tmp_wiki_dir / "wiki")
        index = EmbeddingIndex(tmp_wiki_dir / "wiki" / ".meta" / "embeddings.json")

        # Write pages for all schema slugs and embed them
        for slug in ["topic_a", "topic_b", "topic_c"]:
            repo.write(
                slug=slug,
                title=slug.replace("_", " ").title(),
                body=f"Content about {slug}.",
                sources=["source.md"],
                tags=["test"],
            )
            page = repo.read(slug)
            if page:
                index.upsert(PageEmbedding(slug=slug, vector=_unit_vec(hash(slug) % 100)))
        index.save()

        mock_embed_svc = MagicMock(spec=EmbeddingService)
        result = run_lint(repo=repo, index=index, deep=False)
        assert result.passed

    def test_orphaned_page_detected(self, tmp_wiki_dir: Path) -> None:
        from llm_wiki.index import EmbeddingIndex
        from llm_wiki.lint import run_lint
        from llm_wiki.wiki import WikiRepository

        repo = WikiRepository(tmp_wiki_dir / "wiki")
        index = EmbeddingIndex(tmp_wiki_dir / "wiki" / ".meta" / "embeddings.json")

        # Write a page not in the schema
        repo.write(slug="ghost_page", title="Ghost", body="I am not in schema.", sources=[], tags=[])
        result = run_lint(repo=repo, index=index, deep=False)
        assert not result.passed
        orphans = result.issues_of_kind(LintIssueKind.ORPHANED_PAGE)
        assert any(i.slug == "ghost_page" for i in orphans)

    def test_missing_page_detected(self, tmp_wiki_dir: Path) -> None:
        from llm_wiki.index import EmbeddingIndex
        from llm_wiki.lint import run_lint
        from llm_wiki.wiki import WikiRepository

        repo = WikiRepository(tmp_wiki_dir / "wiki")
        index = EmbeddingIndex(tmp_wiki_dir / "wiki" / ".meta" / "embeddings.json")

        # Don't write any pages — all schema slugs are missing
        result = run_lint(repo=repo, index=index, deep=False)
        missing = result.issues_of_kind(LintIssueKind.MISSING_PAGE)
        assert len(missing) == 3  # all 3 schema pages missing

    def test_stale_embedding_detected(self, tmp_wiki_dir: Path) -> None:
        from llm_wiki.index import EmbeddingIndex
        from llm_wiki.lint import run_lint
        from llm_wiki.wiki import WikiRepository

        repo = WikiRepository(tmp_wiki_dir / "wiki")
        index = EmbeddingIndex(tmp_wiki_dir / "wiki" / ".meta" / "embeddings.json")

        # Write page but don't embed it
        repo.write(slug="topic_a", title="Topic A", body="Content.", sources=["s.md"], tags=[])
        result = run_lint(repo=repo, index=index, deep=False)
        stale = result.issues_of_kind(LintIssueKind.STALE_EMBEDDING)
        assert any(i.slug == "topic_a" for i in stale)

    def test_broken_xref_detected(self, tmp_wiki_dir: Path) -> None:
        from llm_wiki.index import EmbeddingIndex
        from llm_wiki.lint import run_lint
        from llm_wiki.wiki import WikiRepository

        repo = WikiRepository(tmp_wiki_dir / "wiki")
        index = EmbeddingIndex(tmp_wiki_dir / "wiki" / ".meta" / "embeddings.json")

        # Write a page with a [[broken_link]] to a non-existent page
        repo.write(
            slug="topic_a",
            title="Topic A",
            body="This links to [[does_not_exist]] which is missing.",
            sources=["s.md"],
            tags=[],
        )
        result = run_lint(repo=repo, index=index, deep=False)
        broken = result.issues_of_kind(LintIssueKind.BROKEN_XREF)
        assert any("does_not_exist" in i.detail for i in broken)
```

### Step 5.6 — `src/llm_wiki/lint.py`

```python
"""Lint — health checks for the wiki knowledge base.

Structural checks (orphaned pages, missing pages, broken cross-references,
stale embeddings) run without any API calls. The --deep mode adds an
LLM contradiction check across pairs of pages with overlapping tags.
"""

from __future__ import annotations

import re
from itertools import combinations

from anthropic import Anthropic

from llm_wiki.index import EmbeddingIndex
from llm_wiki.models.operations import LintIssue, LintIssueKind, LintResult
from llm_wiki.prompts import build_contradiction_prompt
from llm_wiki.wiki import WikiRepository

_XREF_PATTERN = re.compile(r"\[\[([a-z0-9_\-]+)\]\]")


def _check_orphaned_pages(
    repo: WikiRepository,
    schema_slugs: frozenset[str],
    issues: list[LintIssue],
) -> None:
    """Detect pages on disk that are not in the schema."""
    for slug in repo.list_slugs():
        if slug not in schema_slugs:
            issues.append(LintIssue(
                kind=LintIssueKind.ORPHANED_PAGE,
                slug=slug,
                detail=f"Page '{slug}.md' exists on disk but is not defined in schema.json",
            ))


def _check_missing_pages(
    repo: WikiRepository,
    schema_slugs: frozenset[str],
    issues: list[LintIssue],
) -> None:
    """Detect schema slugs that have no corresponding file on disk."""
    existing = set(repo.list_slugs())
    for slug in schema_slugs:
        if slug not in existing:
            issues.append(LintIssue(
                kind=LintIssueKind.MISSING_PAGE,
                slug=slug,
                detail=f"Schema defines '{slug}' but no wiki/{slug}.md file exists",
            ))


def _check_broken_xrefs(
    repo: WikiRepository,
    all_slugs: set[str],
    issues: list[LintIssue],
) -> None:
    """Detect [[slug]] links pointing to pages that don't exist."""
    for slug in repo.list_slugs():
        page = repo.read(slug)
        if page is None:
            continue
        for xref in _XREF_PATTERN.findall(page.body):
            if xref not in all_slugs:
                issues.append(LintIssue(
                    kind=LintIssueKind.BROKEN_XREF,
                    slug=slug,
                    detail=f"Cross-reference [[{xref}]] in '{slug}.md' points to a non-existent page",
                ))


def _check_stale_embeddings(
    repo: WikiRepository,
    index: EmbeddingIndex,
    issues: list[LintIssue],
) -> None:
    """Detect pages that have no entry in the embedding index."""
    for slug in repo.list_slugs():
        if not index.contains(slug):
            issues.append(LintIssue(
                kind=LintIssueKind.STALE_EMBEDDING,
                slug=slug,
                detail=f"Page '{slug}.md' has no embedding — run `lint --fix` to regenerate",
            ))


def _check_missing_provenance(
    repo: WikiRepository,
    issues: list[LintIssue],
) -> None:
    """Detect pages with no source provenance recorded in front matter."""
    for slug in repo.list_slugs():
        page = repo.read(slug)
        if page and not page.front_matter.sources:
            issues.append(LintIssue(
                kind=LintIssueKind.MISSING_PROVENANCE,
                slug=slug,
                detail=f"Page '{slug}.md' has no sources listed in front matter",
            ))


def _check_contradictions(
    repo: WikiRepository,
    anthropic_client: Anthropic,
    model: str,
    issues: list[LintIssue],
) -> None:
    """Deep check: use LLM to detect factual contradictions between page pairs.

    Only checks pairs that share at least one tag (likely-related pages).
    This is expensive — each pair costs one API call.
    """
    slugs = repo.list_slugs()
    pages = {slug: repo.read(slug) for slug in slugs}
    pages_clean = {k: v for k, v in pages.items() if v is not None}

    for slug_a, slug_b in combinations(pages_clean.keys(), 2):
        page_a = pages_clean[slug_a]
        page_b = pages_clean[slug_b]
        # Only check pages with overlapping tags (reduces API calls)
        shared_tags = set(page_a.front_matter.tags) & set(page_b.front_matter.tags)
        if not shared_tags:
            continue

        prompt = build_contradiction_prompt(
            page_a_title=page_a.front_matter.title,
            page_a_body=page_a.body,
            page_b_title=page_b.front_matter.title,
            page_b_body=page_b.body,
        )
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        analysis: str = response.content[0].text
        if "no" not in analysis.lower()[:50] or "contradict" in analysis.lower():
            issues.append(LintIssue(
                kind=LintIssueKind.CONTRADICTION,
                slug=f"{slug_a} <-> {slug_b}",
                detail=analysis,
            ))


def run_lint(
    repo: WikiRepository,
    index: EmbeddingIndex,
    deep: bool = False,
    fix: bool = False,
    anthropic_client: Anthropic | None = None,
    model: str = "claude-opus-4-5",
    embedding_service: object | None = None,
) -> LintResult:
    """Run all structural health checks on the wiki.

    If deep=True, also runs the LLM contradiction analysis.
    If fix=True, re-embeds all pages to heal a stale index.
    """
    schema = repo.load_schema()
    schema_slugs = schema.slug_set()
    all_existing_slugs = set(repo.list_slugs())
    issues: list[LintIssue] = []

    _check_orphaned_pages(repo, schema_slugs, issues)
    _check_missing_pages(repo, schema_slugs, issues)
    _check_broken_xrefs(repo, all_existing_slugs, issues)
    _check_stale_embeddings(repo, index, issues)
    _check_missing_provenance(repo, issues)

    if deep and anthropic_client is not None:
        _check_contradictions(repo, anthropic_client, model, issues)

    if fix and embedding_service is not None:
        # Re-embed all pages to fix a stale index
        for slug in all_existing_slugs:
            page = repo.read(slug)
            if page:
                emb = embedding_service.embed_page(slug, page.full_text)  # type: ignore[union-attr]
                index.upsert(emb)
        index.save()

    return LintResult(
        issues=issues,
        pages_checked=len(all_existing_slugs),
        deep=deep,
    )
```

---

## Phase 6 — CLI

### Step 6.1 — `src/llm_wiki/cli.py`

```python
"""Typer CLI — six commands wiring all pipeline modules together.

CLI commands are thin: they validate input, construct dependencies,
call the pipeline function, and display the result with Rich.
No business logic lives here.

Commands:
  init     — bootstrap the wiki from schema
  ingest   — run the 5-step ingest pipeline
  query    — run the 4-step query pipeline
  lint     — run structural health checks
  status   — display wiki dashboard
  prompts  — list available query templates
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from anthropic import Anthropic
from openai import OpenAI
from rich.console import Console
from rich.table import Table

from llm_wiki.config import WikiPaths, WikiSettings
from llm_wiki.embeddings import EmbeddingService
from llm_wiki.index import EmbeddingIndex
from llm_wiki.ingest import run_ingest
from llm_wiki.lint import run_lint
from llm_wiki.models.frontmatter import PageSpec, WikiSchema
from llm_wiki.prompts import QUERY_TEMPLATES
from llm_wiki.query import run_query
from llm_wiki.wiki import WikiRepository

app = typer.Typer(
    name="llm-wiki",
    help="Karpathy's LLM Wiki pattern — compiling knowledge that actually compounds.",
    add_completion=False,
)
console = Console()


def _build_dependencies(
    paths: WikiPaths,
    settings: WikiSettings,
) -> tuple[WikiRepository, EmbeddingIndex, EmbeddingService, Anthropic]:
    """Construct all pipeline dependencies from settings and paths."""
    repo = WikiRepository(paths.wiki_dir)
    index = EmbeddingIndex(paths.embeddings_path)
    openai_client = OpenAI(api_key=settings.openai_api_key)
    embedding_service = EmbeddingService(client=openai_client, model=settings.embedding_model)
    anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
    return repo, index, embedding_service, anthropic_client


# ── init ──────────────────────────────────────────────────────────────────────

@app.command()
def init(
    root: Annotated[Path, typer.Option("--root", help="Wiki root directory")] = Path("."),
) -> None:
    """Bootstrap the wiki directory structure from the schema template."""
    paths = WikiPaths(root=root)

    paths.wiki_dir.mkdir(parents=True, exist_ok=True)
    paths.meta_dir.mkdir(parents=True, exist_ok=True)
    paths.sources_dir.mkdir(parents=True, exist_ok=True)

    if not paths.schema_path.exists():
        default_schema = WikiSchema(
            name="My Wiki",
            pages=[
                PageSpec(
                    slug="example",
                    title="Example Page",
                    description="Replace this with your first real concept",
                ),
            ],
        )
        paths.schema_path.write_text(default_schema.model_dump_json(indent=2))
        console.print("[green]✓[/green] Created wiki/.meta/schema.json — add your PageSpecs here.")
    else:
        console.print("[yellow]schema.json already exists — skipping.[/yellow]")

    if not paths.embeddings_path.exists():
        from llm_wiki.models.embeddings import EmbeddingIndexData
        paths.embeddings_path.write_text(EmbeddingIndexData().model_dump_json(indent=2))

    if not paths.index_path.exists():
        paths.index_path.write_text("# Index\n\nNo pages yet.\n")

    if not paths.log_path.exists():
        paths.log_path.write_text("# Log\n\n")

    console.print("\n[bold green]Wiki initialised.[/bold green]")
    console.print(f"  Wiki dir:    [cyan]{paths.wiki_dir}[/cyan]")
    console.print(f"  Sources dir: [cyan]{paths.sources_dir}[/cyan]")
    console.print(f"  Schema:      [cyan]{paths.schema_path}[/cyan]")
    console.print("\nNext steps:")
    console.print("  1. Edit wiki/.meta/schema.json to define your page universe")
    console.print("  2. Add source documents to sources/")
    console.print("  3. Run [bold]llm-wiki ingest sources/<file>[/bold]")


# ── ingest ─────────────────────────────────────────────────────────────────────

@app.command()
def ingest(
    source: Annotated[str, typer.Argument(help="File path, directory, YouTube URL, or HTTP URL")],
    root: Annotated[Path, typer.Option("--root")] = Path("."),
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Ingest a source into the wiki (Route → Synthesize → Embed → Update)."""
    settings = WikiSettings()
    paths = WikiPaths(root=root)
    repo, index, embedding_service, anthropic_client = _build_dependencies(paths, settings)

    source_path = Path(source)

    # Support ingesting an entire directory
    if source_path.is_dir():
        files = list(source_path.glob("*.*"))
        console.print(f"Ingesting {len(files)} files from {source_path}/")
        for f in files:
            _run_single_ingest(str(f), repo, index, embedding_service, anthropic_client, settings)
        return

    _run_single_ingest(source, repo, index, embedding_service, anthropic_client, settings)


def _run_single_ingest(
    source_path: str,
    repo: WikiRepository,
    index: EmbeddingIndex,
    embedding_service: EmbeddingService,
    anthropic_client: Anthropic,
    settings: WikiSettings,
) -> None:
    name = Path(source_path).name if not source_path.startswith("http") else source_path
    console.rule(f"Ingesting: {name}")
    result = run_ingest(
        source_path=source_path,
        repo=repo,
        index=index,
        embedding_service=embedding_service,
        anthropic_client=anthropic_client,
        settings=settings,
    )
    if result.success:
        console.print(f"  Relevant pages: {', '.join(result.slugs_touched) or 'none'}")
        for slug in result.slugs_touched:
            console.print(f"  [green]✓[/green] wiki/{slug}.md written")
        console.print("[green]✓[/green] Embeddings saved")
        console.print("[green]✓[/green] wiki/index.md updated")
        console.print("[green]✓[/green] wiki/log.md updated")
        console.print("\n[bold]Ingest complete.[/bold]")
    else:
        console.print(f"[red]✗ Ingest failed:[/red] {result.error}")
        raise typer.Exit(1)


# ── query ──────────────────────────────────────────────────────────────────────

@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Question to ask the wiki")] = "",
    root: Annotated[Path, typer.Option("--root")] = Path("."),
    save: Annotated[bool, typer.Option("--save", help="File answer back as a wiki page")] = False,
    template: Annotated[str | None, typer.Option("--template", "-t")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    top_k: Annotated[int | None, typer.Option("--top-k")] = None,
) -> None:
    """Query the compiled wiki using natural language or a named template."""
    if not question and not template:
        console.print("[red]Provide a question or use --template <name>[/red]")
        raise typer.Exit(1)

    settings = WikiSettings()
    if top_k is not None:
        settings = settings.model_copy(update={"query_top_k": top_k})

    paths = WikiPaths(root=root)
    repo, index, embedding_service, anthropic_client = _build_dependencies(paths, settings)

    result = run_query(
        question=question,
        repo=repo,
        index=index,
        embedding_service=embedding_service,
        anthropic_client=anthropic_client,
        settings=settings,
        save=save,
        template=template,
        verbose=verbose,
    )

    if verbose and result.hits:
        table = Table(title="Retrieved Wiki Pages")
        table.add_column("Rank", style="cyan")
        table.add_column("Page")
        table.add_column("Slug")
        table.add_column("Score", justify="right")
        for i, hit in enumerate(result.hits, 1):
            table.add_row(str(i), hit.title, hit.slug, f"{hit.score:.4f}")
        console.print(table)

    console.print(result.answer)

    if result.saved_slug:
        console.print(f"\n[green]✓[/green] Answer saved as wiki/{result.saved_slug}.md")


# ── lint ───────────────────────────────────────────────────────────────────────

@app.command()
def lint(
    root: Annotated[Path, typer.Option("--root")] = Path("."),
    deep: Annotated[bool, typer.Option("--deep", help="Run LLM contradiction check")] = False,
    fix: Annotated[bool, typer.Option("--fix", help="Re-embed all pages")] = False,
) -> None:
    """Run structural health checks on the wiki."""
    settings = WikiSettings()
    paths = WikiPaths(root=root)
    repo, index, embedding_service, anthropic_client = _build_dependencies(paths, settings)

    if deep:
        console.print("Running contradiction check (deep)...")

    if fix:
        console.print("Re-embedding all pages...")
        for slug in repo.list_slugs():
            page = repo.read(slug)
            if page:
                emb = embedding_service.embed_page(slug, page.full_text)
                index.upsert(emb)
                console.print(f"  [green]✓[/green] Re-embedded {slug}")
        index.save()
        console.print("All embeddings updated.")

    result = run_lint(
        repo=repo,
        index=index,
        deep=deep,
        anthropic_client=anthropic_client if deep else None,
        model=settings.claude_model,
        embedding_service=embedding_service,
    )

    console.rule("Lint Report")
    if result.passed:
        console.print("\n[green]All checks passed.[/green]")
    else:
        for issue in result.issues:
            console.print(f"  [red]✗[/red] {issue}")

    console.print(f"\nTotal pages checked: {result.pages_checked}")
    if not result.passed:
        raise typer.Exit(1)


# ── status ─────────────────────────────────────────────────────────────────────

@app.command()
def status(
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:
    """Display a dashboard overview of the wiki state."""
    paths = WikiPaths(root=root)

    if not paths.wiki_dir.exists():
        console.print("[red]Wiki not initialised. Run `llm-wiki init` first.[/red]")
        raise typer.Exit(1)

    repo = WikiRepository(paths.wiki_dir)
    schema = repo.load_schema()
    index = EmbeddingIndex(paths.embeddings_path)
    slugs = repo.list_slugs()
    sources = list(paths.sources_dir.glob("*.*")) if paths.sources_dir.exists() else []

    console.rule("Wiki Status")
    console.print(f"\nWiki pages:     [bold]{len(slugs)}[/bold]")
    console.print(f"Schema pages:   [bold]{len(schema.pages)}[/bold]")
    console.print(f"Wiki name:      [bold]{schema.name}[/bold]")
    console.print(f"Source files:   [bold]{len(sources)}[/bold]")
    console.print(f"Embedded pages: [bold]{index.size}[/bold]")

    if slugs:
        table = Table(title="Wiki Pages")
        table.add_column("Slug", style="cyan")
        table.add_column("Title")
        table.add_column("Tags")
        table.add_column("Updated")
        for slug in sorted(slugs):
            page = repo.read(slug)
            if page:
                fm = page.front_matter
                table.add_row(
                    fm.slug,
                    fm.title,
                    ", ".join(fm.tags) if fm.tags else "—",
                    str(fm.updated),
                )
        console.print(table)

    # Show recent log entries
    log = repo.read_log()
    lines = [l for l in log.splitlines() if l.strip().startswith("-")]
    if lines:
        console.print("\nRecent activity (last 5):")
        for line in lines[-5:]:
            console.print(f"  {line}")


# ── prompts ────────────────────────────────────────────────────────────────────

@app.command()
def prompts(
    category: Annotated[str | None, typer.Option("--category", "-c")] = None,
) -> None:
    """List all available query templates, optionally filtered by category."""
    table = Table(title="Query Templates")
    table.add_column("Name", style="cyan")
    table.add_column("Category", style="yellow")
    table.add_column("Description")

    for name, template in sorted(QUERY_TEMPLATES.items()):
        if category and template.category.lower() != category.lower():
            continue
        table.add_row(name, template.category, template.description)

    console.print(table)
    console.print(f"\n[dim]Use: llm-wiki query --template <name>[/dim]")
```

### Step 6.2 — `src/llm_wiki/__init__.py`

```python
"""llm-wiki — Karpathy's LLM Wiki pattern.

Compiling knowledge that compounds.
"""

__version__ = "0.1.0"
```

### Step 6.3 — `main.py`

```python
"""Entry point shortcut — equivalent to `llm-wiki` CLI when installed."""

from llm_wiki.cli import app

if __name__ == "__main__":
    app()
```

---

## Phase 7 — CLAUDE.md

Create `CLAUDE.md` at the project root. This file is read by the Claude Code agent on every session and enforces the invariants that keep the wiki healthy over time.

### `CLAUDE.md`

```markdown
# LLM Wiki — Agent Runtime Rules

You are the LLM maintaining this wiki. Read this file before taking any
action that touches wiki/*.md files.

## Three-Layer Architecture

```
sources/        ← Raw input documents. IMMUTABLE. Never modify these.
wiki/*.md       ← Compiled knowledge base. You maintain this.
wiki/.meta/     ← Schema + embeddings. Human manages schema.json;
                  you keep embeddings.json in sync.
```

## Your Core Invariants

1. **Never discard knowledge.** When synthesising a page, preserve all
   existing content and extend it. If new information contradicts old,
   note the contradiction explicitly — never silently overwrite.

2. **Every write is validated.** All page writes go through `WikiRepository.write()`.
   Never use `Path.write_text()` directly on a wiki page.

3. **Cross-references use [[slug]] notation.** When you reference another
   wiki page, use `[[slug]]` format. Slugs are lowercase alphanumeric with
   underscores only.

4. **Provenance is always tracked.** Every page's front matter must list
   all sources that contributed to it. Never write a page with an empty
   `sources:` field after ingest.

5. **The schema is the page universe.** You do not create pages for slugs
   not defined in schema.json. If a concept deserves a page, note the gap
   and the human will add the PageSpec.

## YAML Front Matter Schema

Every wiki page begins with exactly this front matter structure:

```yaml
---
slug: example_concept       # lowercase_with_underscores
title: Example Concept      # Human-readable title
tags:
  - tag1
  - tag2
sources:
  - source_filename.md
updated: "YYYY-MM-DD"
---
```

## The Five-Step Ingest Pipeline

When ingesting a source:

1. **Resolve**: Detect source type (file / YouTube / HTTP) and extract text
2. **Route**: Read schema context, return JSON array of relevant slugs only
3. **Synthesize**: For each slug, rewrite the full page, extending existing content
4. **Embed**: Re-embed each updated page; upsert into embeddings.json
5. **Update**: Regenerate index.md table; append to log.md

## Quality Standards

- Pages should be dense with content, not thin summaries
- Every key claim should ideally trace back to a source
- Cross-references strengthen the knowledge graph — use them liberally
- Contradiction detection is healthy — surface it, don't hide it
- Run `llm-wiki lint` after every batch ingest to catch structural issues

## File Ownership

| File | Owner | Rule |
|------|-------|------|
| sources/*.* | Human | Never modify |
| wiki/.meta/schema.json | Human | Never modify without instruction |
| wiki/*.md | LLM | Managed via WikiRepository |
| wiki/.meta/embeddings.json | LLM | Kept in sync after every write |
| wiki/index.md | LLM | Regenerated after every ingest |
| wiki/log.md | LLM | Append-only |
```

---

## Phase 8 — Quality Check Pass

After all code is generated, run these checks in order. Fix every finding
before considering the implementation complete.

### Step 8.1 — Automated checks

```bash
# Install the project
uv sync

# Run the full test suite with coverage
uv run pytest --cov=src/llm_wiki --cov-report=term-missing

# Ruff lint — should be zero errors
uv run ruff check src/ tests/

# Ruff format check
uv run ruff format --check src/ tests/

# Type check with ty
uv run ty check src/
```

### Step 8.2 — Manual code review checklist

Work through each file and verify the following. For every finding, fix it
before moving on.

**Models (`src/llm_wiki/models/`)**

- [ ] The `_validate_slug` function is defined *once* in `frontmatter.py` and reused by both `PageSpec` and `WikiFrontMatter` — not duplicated.
- [ ] All `field_validator` functions have the `@classmethod` decorator.
- [ ] All `model_validator` functions return `Self` (or the model class), not `None`.
- [ ] `WikiFrontMatter.add_source()` uses `model_copy` (immutable update), not direct mutation.
- [ ] `EmbeddingIndexData` validates every stored embedding on load — not just the top-level structure.
- [ ] `RouteResult.coerce_slugs` handles `None`, non-list, and empty-string entries without raising.

**Config (`src/llm_wiki/config.py`)**

- [ ] `WikiSettings` uses `SettingsConfigDict(extra="ignore")` to allow arbitrary env vars without raising.
- [ ] `WikiPaths` has no mutable state — all properties are computed from `self.root`.
- [ ] There are no hardcoded `Path("wiki")` strings outside of `WikiPaths`.

**Prompts (`src/llm_wiki/prompts.py`)**

- [ ] Every `QUERY_TEMPLATES` entry has a `name` field that matches its dict key.
- [ ] The synthesis prompt contains language that enforces the preservation invariant ("preserve", "never discard", or equivalent).
- [ ] `get_template()` returns `None` for unknown names, never raises `KeyError`.
- [ ] No prompt text exists anywhere outside this file.

**Embeddings (`src/llm_wiki/embeddings.py`)**

- [ ] Input is truncated to `_MAX_INPUT_CHARS` *before* the API call, not after.
- [ ] The vector is normalised using `np.linalg.norm`, guarded against zero-norm vectors.
- [ ] The `EmbeddingService` class has no global state — the `OpenAI` client is injected.

**Index (`src/llm_wiki/index.py`)**

- [ ] `top_k()` returns an empty list (not raises) when the index is empty.
- [ ] Vectors are assumed pre-normalised — dot product is used, not full cosine formula.
- [ ] `save()` writes atomically (writes to the same path — acceptable for this scale; note limitation).
- [ ] `remove()` uses `.pop(slug, None)` — does not raise on missing slugs.

**Wiki (`src/llm_wiki/wiki.py`)**

- [ ] `read()` validates front matter via `WikiFrontMatter.model_validate()` on every load — never raw dict access.
- [ ] `write()` always sets `updated=date.today()` — never inherits a stale date from the caller.
- [ ] `list_slugs()` excludes `"index"` and `"log"` — these are not wiki pages.
- [ ] `append_log()` opens in append mode (`"a"`) and never overwrites.

**Ingest (`src/llm_wiki/ingest.py`)**

- [ ] `resolve_source()` correctly distinguishes YouTube URLs from generic HTTP URLs before falling through to the HTTP branch.
- [ ] `route()` catches `json.JSONDecodeError` and returns an empty `RouteResult` — never raises on malformed LLM output.
- [ ] `run_ingest()` merges source lists without duplicates using `dict.fromkeys()`.
- [ ] The `IngestError` exception is defined in this module — not imported from elsewhere.
- [ ] `run_ingest()` returns a failed `IngestResult` (not raises) when `FileNotFoundError` or `IngestError` occurs.

**Query (`src/llm_wiki/query.py`)**

- [ ] `_question_to_slug()` strips all non-alphanumeric characters before replacing spaces, preventing malformed slugs.
- [ ] The `--save` path calls `index.save()` after upserting the new embedding.
- [ ] If `template` and `question` are both provided, `template` takes precedence with a clear comment.
- [ ] `hits` contains only pages that were successfully `repo.read()` — not slugs from the index that have no file.

**Lint (`src/llm_wiki/lint.py`)**

- [ ] `_XREF_PATTERN` matches `[[slug]]` with lowercase alphanumeric slugs only — same format enforced by `_validate_slug`.
- [ ] `_check_contradictions()` only checks pairs sharing at least one tag, not all `O(n²)` pairs.
- [ ] `run_lint()` with `fix=True` but `embedding_service=None` should not raise — guard with `if fix and embedding_service is not None`.
- [ ] Each check function takes `issues: list[LintIssue]` as a mutable output parameter — not returning a list. This is consistent across all five check functions.

**CLI (`src/llm_wiki/cli.py`)**

- [ ] `_build_dependencies()` is the *single* place that constructs `Anthropic` and `OpenAI` clients — not duplicated across commands.
- [ ] Every command that calls the pipeline wraps the call in error handling and calls `raise typer.Exit(1)` on failure.
- [ ] The `ingest` command's directory path uses `source_path.is_dir()`, not string matching.
- [ ] `status` checks `paths.wiki_dir.exists()` and exits gracefully if the wiki is not initialised.
- [ ] No command accesses `settings.anthropic_api_key` or `settings.openai_api_key` directly — that belongs in `_build_dependencies`.

**Tests**

- [ ] Every test that touches the filesystem uses `tmp_wiki_dir` or `tmp_path` fixtures — never hardcoded paths.
- [ ] Every test that calls the Anthropic or OpenAI API uses a `MagicMock` — no live API calls in tests.
- [ ] `conftest.py` `sample_schema` has exactly 3 pages — tests that check "3 missing pages" depend on this.
- [ ] Tests for `RouteResult` with a non-list `relevant_slugs` value cover the `None` case (simulates a malformed LLM response).
- [ ] There are no `import *` statements in any test file.

### Step 8.3 — Code smell sweep

Look for these patterns and refactor if found:

**Duplication:**
- The slug validation regex (`[a-z0-9][a-z0-9_\-]*`) must appear exactly once, in `_validate_slug()` in `frontmatter.py`. If it appears anywhere else, centralise it.
- The `_unit_vec(seed)` helper in tests appears in both `test_index.py` and `test_lint.py` — move it to `conftest.py` as a fixture or helper function.

**Type annotation gaps:**
- All public functions must have complete return type annotations (ruff's `ANN` rules enforce this, but manually verify `run_ingest`, `run_query`, `run_lint`).
- All `list[...]` and `dict[...]` annotations must use the lowercase form (Python 3.13 style, not `List[...]`).

**Error handling:**
- Verify there are no bare `except Exception` blocks — each `except` should name a specific exception type.
- `ingest.py` catches `FileNotFoundError | IngestError` in `run_ingest` — make sure no other exceptions are silently swallowed.

**Magic numbers:**
- `50_000` (max source chars) must not appear as a literal anywhere — it must be a `WikiSettings` field with a named alias.
- `1536` (embedding dim) must only appear as `EMBEDDING_DIM` from `models/embeddings.py`.
- `8191 * 4` (max input chars for embeddings) is defined as `_MAX_INPUT_CHARS` in `embeddings.py` — verify it's not also hardcoded in tests.

**Pydantic usage:**
- Every LLM JSON response is parsed through a Pydantic model — there should be no raw `json.loads(...)` result used directly without validation.
- `model_dump(mode="json")` is used when passing data to YAML/JSON serialisation — verify no `model_dump()` without `mode="json"` is used for this purpose.

### Step 8.4 — Final verification run

```bash
# Full test suite must pass with ≥85% coverage
uv run pytest --cov=src/llm_wiki --cov-fail-under=85

# Zero ruff errors
uv run ruff check src/ tests/

# Zero ty errors
uv run ty check src/

# Verify the CLI is importable (catches import-time errors)
uv run python -c "from llm_wiki.cli import app; print('CLI OK')"

# Verify the models module exports everything
uv run python -c "from llm_wiki.models import *; print('Models OK')"
```

---

## Appendix — Demo Sources

Create these files in `sources/` to test the pipeline end-to-end:

### `sources/backprop_notes.md`

```markdown
# Backpropagation Notes

Backpropagation is the algorithm for computing gradients in neural networks.
It applies the chain rule of calculus recursively from the output layer back
to the input layer. For a loss L and weights W, the gradient ∂L/∂W tells us
how to update W to reduce the loss.

Key insight: gradients flow backwards through the computational graph.
Each layer receives the gradient from the layer above and passes a modified
gradient to the layer below.

Vanishing gradients are a major problem with deep networks using sigmoid
activations — gradients shrink exponentially as they propagate back through
many layers. ReLU activations largely solve this.
```

### `sources/attention_paper_excerpt.txt`

```
Excerpt: "Attention Is All You Need" (Vaswani et al., 2017)

The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks. We propose a new simple network architecture,
the Transformer, based solely on attention mechanisms, dispensing with
recurrence and convolutions entirely.

An attention function can be described as mapping a query and a set of
key-value pairs to an output. The output is computed as a weighted sum of
the values, where the weight assigned to each value is computed by a
compatibility function of the query with the corresponding key.

We call our particular attention "Scaled Dot-Product Attention":
  Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

The scaling factor 1/sqrt(d_k) counteracts the effect of dot products
growing large in magnitude, which would push the softmax into regions
with extremely small gradients.
```

---

## Quick Reference

```
uv sync                                    # install dependencies
uv run llm-wiki init                       # bootstrap wiki structure
uv run llm-wiki ingest sources/file.md    # ingest a local file
uv run llm-wiki ingest sources/           # ingest all sources at once
uv run llm-wiki ingest https://...        # ingest a URL or YouTube video
uv run llm-wiki query "your question"     # query the wiki
uv run llm-wiki query "question" --save   # query and file the answer back
uv run llm-wiki query --template blind-spot   # use a named template
uv run llm-wiki lint                      # structural health checks
uv run llm-wiki lint --deep               # + LLM contradiction analysis
uv run llm-wiki lint --fix                # re-embed all pages
uv run llm-wiki status                    # overview dashboard
uv run llm-wiki prompts                   # list all 24+ query templates
uv run pytest                             # run test suite
uv run ruff check src/ tests/             # lint
uv run ty check src/                      # type check
```
