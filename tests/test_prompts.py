"""Tests for prompt construction and the QUERY_TEMPLATES registry."""

from __future__ import annotations

from llm_wiki.prompts import (
    QUERY_TEMPLATES,
    build_answer_prompt,
    build_routing_prompt,
    build_synthesis_prompt,
    get_template,
)


class TestRoutingPrompt:
    def test_routing_prompt_contains_schema_context(self) -> None:
        context = "a: Topic A — About A\nb: Topic B — About B"
        prompt = build_routing_prompt(schema_context=context, source_text="Some source content.")
        assert "a: Topic A" in prompt
        assert "Some source content." in prompt

    def test_routing_prompt_requests_json_array(self) -> None:
        prompt = build_routing_prompt(schema_context="a: A — d", source_text="text")
        assert "JSON" in prompt or "json" in prompt


class TestSynthesisPrompt:
    def test_synthesis_prompt_includes_existing_body(self) -> None:
        prompt = build_synthesis_prompt(
            slug="transformers", title="Transformers",
            existing_body="## Existing content", source_text="New information.",
        )
        assert "## Existing content" in prompt
        assert "New information." in prompt

    def test_synthesis_prompt_enforces_preservation_invariant(self) -> None:
        prompt = build_synthesis_prompt(slug="t", title="T", existing_body="old", source_text="new")
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

    def test_get_template_returns_correct_template(self) -> None:
        t = get_template("master-summary")
        assert t is not None
        assert t.name == "master-summary"

    def test_get_template_returns_none_for_unknown(self) -> None:
        assert get_template("does-not-exist") is None

    def test_minimum_24_templates_defined(self) -> None:
        assert len(QUERY_TEMPLATES) >= 24
