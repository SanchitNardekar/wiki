"""Tests for PageSpec, WikiSchema, and WikiFrontMatter."""

from __future__ import annotations

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
        assert schema.get_page("found") is not None
        assert schema.get_page("found").slug == "found"  # type: ignore[union-attr]

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
            slug="s", title="T", tags=["ML", "Deep-Learning", "NLP"], updated=date.today()
        )
        assert fm.tags == ["ml", "deep-learning", "nlp"]

    def test_none_tags_defaults_to_empty_list(self) -> None:
        fm = WikiFrontMatter(slug="s", title="T", updated=date.today())
        assert fm.tags == []

    def test_add_source_appends_new_source(self) -> None:
        fm = WikiFrontMatter(slug="s", title="T", sources=["old.md"], updated=date(2020, 1, 1))
        new_fm = fm.add_source("new.md")
        assert "old.md" in new_fm.sources
        assert "new.md" in new_fm.sources

    def test_add_source_is_idempotent(self) -> None:
        fm = WikiFrontMatter(slug="s", title="T", sources=["existing.md"], updated=date.today())
        new_fm = fm.add_source("existing.md")
        assert new_fm.sources.count("existing.md") == 1

    def test_add_source_updates_date(self) -> None:
        fm = WikiFrontMatter(slug="s", title="T", sources=[], updated=date(2020, 1, 1))
        new_fm = fm.add_source("new.md")
        assert new_fm.updated == date.today()

    def test_front_matter_serialises_date_as_string(self) -> None:
        fm = WikiFrontMatter(slug="s", title="T", updated=date(2026, 4, 14))
        data = fm.model_dump(mode="json")
        assert data["updated"] == "2026-04-14"
