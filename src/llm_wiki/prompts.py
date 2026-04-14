"""All prompt text and query templates — the single source of truth."""

from __future__ import annotations

from dataclasses import dataclass


def build_routing_prompt(schema_context: str, source_text: str) -> str:
    return f"""\
You are a knowledge routing assistant. Decide which wiki pages are relevant to the source text.

Each line: slug: title — description

WIKI SCHEMA:
{schema_context}

SOURCE TEXT:
{source_text}

Return ONLY a JSON array of relevant page slugs. Be selective.
Example: ["attention_mechanism", "transformers"]"""


def build_synthesis_prompt(
    slug: str, title: str, existing_body: str, source_text: str,
) -> str:
    existing_section = (
        f"EXISTING PAGE CONTENT:\n{existing_body}"
        if existing_body.strip()
        else "EXISTING PAGE CONTENT:\n(New page — no existing content yet.)"
    )
    return f"""\
Update the wiki page for "{title}" (slug: {slug}) by integrating new source information.

CRITICAL RULES:
1. Preserve and EXTEND existing content — NEVER discard information.
2. Add [[slug]] cross-references to related pages where appropriate.
3. Note contradictions between new and existing information explicitly.
4. Use clear markdown with headings and bullet points.
5. Return markdown body only — no YAML front matter.

{existing_section}

NEW SOURCE TEXT:
{source_text}

Write the complete updated wiki page body:"""


def build_answer_prompt(question: str, wiki_context: str) -> str:
    return f"""\
Answer using the compiled knowledge wiki below.

WIKI CONTEXT:
{wiki_context}

QUESTION: {question}

Cite page titles using [Page Title] format. Be clear and comprehensive."""


def build_contradiction_prompt(
    page_a_title: str, page_a_body: str, page_b_title: str, page_b_body: str,
) -> str:
    return f"""\
Compare these two wiki pages for factual contradictions.

PAGE 1: {page_a_title}
{page_a_body}

PAGE 2: {page_b_title}
{page_b_body}

List genuine factual contradictions only. Notational differences are NOT contradictions."""


def build_index_update_prompt(
    schema_context: str, updated_pages: list[tuple[str, str, list[str]]],
) -> str:
    page_list = "\n".join(
        f"- **{title}** (`{slug}`) — tags: {', '.join(tags)}"
        for slug, title, tags in updated_pages
    )
    return f"""\
Update the wiki index.

SCHEMA:
{schema_context}

PAGES:
{page_list}

Generate a markdown index with a summary table. Return complete index.md content:"""


@dataclass(frozen=True)
class QueryTemplate:
    name: str
    category: str
    description: str
    prompt: str


def get_template(name: str) -> QueryTemplate | None:
    return QUERY_TEMPLATES.get(name)


QUERY_TEMPLATES: dict[str, QueryTemplate] = {
    # ── Synthesis ──────────────────────────────────────────────────────────────
    "master-summary": QueryTemplate(
        name="master-summary", category="synthesis",
        description="The single most important insight tying everything together",
        prompt="Read everything in the wiki and identify the single most important insight — the idea that ties all other concepts together. Explain why it is the keystone.",
    ),
    "concept-map": QueryTemplate(
        name="concept-map", category="synthesis",
        description="Map every major concept and how they connect",
        prompt="Identify every major concept and describe how each relates to at least two others. Present as a structured map.",
    ),
    "timeline": QueryTemplate(
        name="timeline", category="synthesis",
        description="Chronological development of ideas",
        prompt="Reconstruct the historical or logical timeline of ideas. Which came first? What did later ideas build on?",
    ),
    "first-principles": QueryTemplate(
        name="first-principles", category="synthesis",
        description="Reduce everything to first principles",
        prompt="What are the fundamental axioms underlying everything in this wiki? What minimal starting assumptions would rebuild it?",
    ),
    # ── Gap-Finding ────────────────────────────────────────────────────────────
    "blind-spot": QueryTemplate(
        name="blind-spot", category="gap-finding",
        description="Important topics missing from the wiki",
        prompt="What important topics are absent? List the top 5 gaps with explanations of why each matters.",
    ),
    "weak-links": QueryTemplate(
        name="weak-links", category="gap-finding",
        description="Concepts with insufficient depth",
        prompt="Which pages have insufficient depth relative to importance? Rank by urgency for improvement.",
    ),
    "source-gaps": QueryTemplate(
        name="source-gaps", category="gap-finding",
        description="Areas where more source material is needed",
        prompt="What types of source material would most improve the knowledge base? Name topics and suggest source types.",
    ),
    "stale-claims": QueryTemplate(
        name="stale-claims", category="gap-finding",
        description="Claims likely superseded by newer developments",
        prompt="Which claims are most likely outdated? Flag dates, benchmarks, or state-of-the-art assertions.",
    ),
    # ── Debate ─────────────────────────────────────────────────────────────────
    "biggest-disagreement": QueryTemplate(
        name="biggest-disagreement", category="debate",
        description="The biggest tension between sources",
        prompt="What is the single biggest disagreement? Steelman both sides, then assess which is stronger.",
    ),
    "tradeoffs": QueryTemplate(
        name="tradeoffs", category="debate",
        description="The most important tradeoffs in this domain",
        prompt="What are the most important tradeoffs? For each, describe gains, losses, and deciding conditions.",
    ),
    "competing-paradigms": QueryTemplate(
        name="competing-paradigms", category="debate",
        description="Different paradigms or schools of thought",
        prompt="Are there competing paradigms? Describe each, its assumptions, strengths, limitations, and favourable contexts.",
    ),
    # ── Output ─────────────────────────────────────────────────────────────────
    "study-guide": QueryTemplate(
        name="study-guide", category="output",
        description="A structured study guide",
        prompt="Create a study guide: prerequisites, core concepts in order, key relationships, misconceptions, exercises.",
    ),
    "cheat-sheet": QueryTemplate(
        name="cheat-sheet", category="output",
        description="A compact reference cheat sheet",
        prompt="Create a concise cheat sheet of the most important facts, formulas, definitions, and rules of thumb.",
    ),
    "faq": QueryTemplate(
        name="faq", category="output",
        description="Anticipated frequently asked questions",
        prompt="Generate a FAQ with the 10 most likely questions a new learner would ask, ordered foundational to advanced.",
    ),
    "slide-outline": QueryTemplate(
        name="slide-outline", category="output",
        description="A slide deck outline for presenting this knowledge",
        prompt="Create a slide deck outline for a 20-minute presentation: title, agenda, ~12 content slides, conclusion.",
    ),
    "executive-summary": QueryTemplate(
        name="executive-summary", category="output",
        description="A non-technical executive summary",
        prompt="Write a one-page executive summary for a non-technical audience: what, why, key ideas, implications.",
    ),
    # ── Health ─────────────────────────────────────────────────────────────────
    "integrity-report": QueryTemplate(
        name="integrity-report", category="health",
        description="Audit the wiki for internal consistency",
        prompt="Audit for: conflicting definitions, inverted cross-references, contradictions, overlapping scope.",
    ),
    "duplication-check": QueryTemplate(
        name="duplication-check", category="health",
        description="Identify overlapping or duplicated content",
        prompt="Which pages cover overlapping content? For each pair, suggest merge, subsume, or scope distinction.",
    ),
    "provenance-audit": QueryTemplate(
        name="provenance-audit", category="health",
        description="Identify claims lacking source attribution",
        prompt="Which pages contain confident assertions not traceable to any source? Flag the most significant.",
    ),
    # ── Personal ───────────────────────────────────────────────────────────────
    "unknown-unknowns": QueryTemplate(
        name="unknown-unknowns", category="personal",
        description="Likely mistakes or blind spots",
        prompt="What mistakes am I likely making? What dangerous assumptions am I probably not questioning?",
    ),
    "next-steps": QueryTemplate(
        name="next-steps", category="personal",
        description="Recommended next actions",
        prompt="What are the top 5 most valuable actions I could take right now? Prioritise by impact and feasibility.",
    ),
    "decision-framework": QueryTemplate(
        name="decision-framework", category="personal",
        description="A decision framework from the wiki's knowledge",
        prompt="Synthesise a decision framework: what questions to ask, key variables to evaluate, structured process.",
    ),
    "teaching-script": QueryTemplate(
        name="teaching-script", category="personal",
        description="How to teach this to someone else",
        prompt="Write a teaching script for a smart colleague: order, analogies, examples, misconceptions to address.",
    ),
    "progress-reflection": QueryTemplate(
        name="progress-reflection", category="personal",
        description="What the wiki reveals about learning progress",
        prompt="Based on content and gaps, where am I in understanding? What would a true expert notice missing?",
    ),
    "analogy-bridge": QueryTemplate(
        name="analogy-bridge", category="synthesis",
        description="Connect wiki concepts to everyday analogies",
        prompt="For each major concept, provide an everyday analogy that captures its essence. Test each analogy for where it breaks down.",
    ),
}
