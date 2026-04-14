"""Lint — health checks for the wiki knowledge base."""

from __future__ import annotations

import re
from itertools import combinations

from openai import OpenAI

from llm_wiki.index import EmbeddingIndex
from llm_wiki.models.operations import LintIssue, LintIssueKind, LintResult
from llm_wiki.prompts import build_contradiction_prompt
from llm_wiki.wiki import WikiRepository

_XREF_PATTERN = re.compile(r"\[\[([a-z0-9_\-]+)\]\]")


def _check_orphaned_pages(
    repo: WikiRepository, schema_slugs: frozenset[str], issues: list[LintIssue],
) -> None:
    for slug in repo.list_slugs():
        if slug not in schema_slugs:
            issues.append(LintIssue(kind=LintIssueKind.ORPHANED_PAGE, slug=slug, detail=f"Page '{slug}.md' not in schema"))


def _check_missing_pages(
    repo: WikiRepository, schema_slugs: frozenset[str], issues: list[LintIssue],
) -> None:
    existing = set(repo.list_slugs())
    for slug in schema_slugs:
        if slug not in existing:
            issues.append(LintIssue(kind=LintIssueKind.MISSING_PAGE, slug=slug, detail=f"Schema defines '{slug}' but no file exists"))


def _check_broken_xrefs(
    repo: WikiRepository, all_slugs: set[str], issues: list[LintIssue],
) -> None:
    for slug in repo.list_slugs():
        page = repo.read(slug)
        if page is None:
            continue
        for xref in _XREF_PATTERN.findall(page.body):
            if xref not in all_slugs:
                issues.append(LintIssue(kind=LintIssueKind.BROKEN_XREF, slug=slug, detail=f"[[{xref}]] points to non-existent page"))


def _check_stale_embeddings(
    repo: WikiRepository, index: EmbeddingIndex, issues: list[LintIssue],
) -> None:
    for slug in repo.list_slugs():
        if not index.contains(slug):
            issues.append(LintIssue(kind=LintIssueKind.STALE_EMBEDDING, slug=slug, detail=f"Page '{slug}.md' has no embedding"))


def _check_missing_provenance(repo: WikiRepository, issues: list[LintIssue]) -> None:
    for slug in repo.list_slugs():
        page = repo.read(slug)
        if page and not page.front_matter.sources:
            issues.append(LintIssue(kind=LintIssueKind.MISSING_PROVENANCE, slug=slug, detail=f"Page '{slug}.md' has no sources"))


def _check_contradictions(
    repo: WikiRepository, openai_client: OpenAI, model: str, issues: list[LintIssue],
) -> None:
    slugs = repo.list_slugs()
    pages = {s: repo.read(s) for s in slugs}
    pages_clean = {k: v for k, v in pages.items() if v is not None}

    for slug_a, slug_b in combinations(pages_clean, 2):
        page_a, page_b = pages_clean[slug_a], pages_clean[slug_b]
        if not set(page_a.front_matter.tags) & set(page_b.front_matter.tags):
            continue
        prompt = build_contradiction_prompt(
            page_a.front_matter.title, page_a.body, page_b.front_matter.title, page_b.body,
        )
        response = openai_client.chat.completions.create(
            model=model, max_completion_tokens=1024, messages=[{"role": "user", "content": prompt}],
        )
        analysis: str = response.choices[0].message.content or ""
        if "no" not in analysis.lower()[:50] or "contradict" in analysis.lower():
            issues.append(LintIssue(kind=LintIssueKind.CONTRADICTION, slug=f"{slug_a} <-> {slug_b}", detail=analysis))


def run_lint(
    repo: WikiRepository,
    index: EmbeddingIndex,
    deep: bool = False,
    openai_client: OpenAI | None = None,
    model: str = "gpt-5.2",
) -> LintResult:
    schema = repo.load_schema()
    schema_slugs = schema.slug_set()
    all_existing = set(repo.list_slugs())
    issues: list[LintIssue] = []

    _check_orphaned_pages(repo, schema_slugs, issues)
    _check_missing_pages(repo, schema_slugs, issues)
    _check_broken_xrefs(repo, all_existing, issues)
    _check_stale_embeddings(repo, index, issues)
    _check_missing_provenance(repo, issues)

    if deep and openai_client is not None:
        _check_contradictions(repo, openai_client, model, issues)

    return LintResult(issues=issues, pages_checked=len(all_existing), deep=deep)
