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
            source_name="paper.md", relevant_slugs=["attention_mechanism", "transformers"]
        )
        assert len(result.relevant_slugs) == 2

    def test_slugs_normalised_to_lowercase(self) -> None:
        result = RouteResult(source_name="paper.md", relevant_slugs=["ATTENTION", "Transformers"])
        assert result.relevant_slugs == ["attention", "transformers"]

    def test_empty_slugs_list_is_valid(self) -> None:
        result = RouteResult(source_name="paper.md", relevant_slugs=[])
        assert result.relevant_slugs == []

    def test_non_list_slugs_defaults_to_empty(self) -> None:
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
        result = IngestResult(
            source_name="bad.pdf", slugs_touched=[], success=False, error="File not found"
        )
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
            question="What is attention?", hits=[], answer="...", saved_slug="what_is_attention"
        )
        assert result.saved_slug == "what_is_attention"

    def test_similarity_score_range_enforced(self) -> None:
        with pytest.raises(ValidationError):
            SimilarityHit(slug="a", title="A", score=1.5)


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

    def test_lint_issue_str_includes_slug(self) -> None:
        issue = LintIssue(
            kind=LintIssueKind.BROKEN_XREF, slug="transformers", detail="[[missing]] not found"
        )
        assert "transformers" in str(issue)
        assert "broken_xref" in str(issue)
