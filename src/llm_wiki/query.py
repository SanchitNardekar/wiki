"""Four-step query pipeline: Embed → Search → Assemble → Stream."""

from __future__ import annotations

import re

from openai import OpenAI

from llm_wiki.config import WikiSettings
from llm_wiki.embeddings import EmbeddingService
from llm_wiki.index import EmbeddingIndex
from llm_wiki.models.operations import QueryResult, SimilarityHit
from llm_wiki.prompts import build_answer_prompt, get_template
from llm_wiki.wiki import WikiRepository


def _question_to_slug(question: str) -> str:
    slug = re.sub(r"[^a-z0-9\s]", "", question.lower())
    slug = re.sub(r"\s+", "_", slug.strip())
    return slug[:60]


def run_query(
    question: str,
    repo: WikiRepository,
    index: EmbeddingIndex,
    embedding_service: EmbeddingService,
    openai_client: OpenAI,
    settings: WikiSettings,
    save: bool = False,
    template: str | None = None,
    verbose: bool = False,
) -> QueryResult:
    # Template overrides free-form question
    if template:
        tmpl = get_template(template)
        if tmpl is None:
            raise ValueError(f"Unknown template: {template!r}. Run `llm-wiki prompts` to see options.")
        effective_question = tmpl.prompt
    else:
        effective_question = question

    query_vector = embedding_service.embed(effective_question)
    hits_raw = index.top_k(query_vector, k=settings.query_top_k)

    hits: list[SimilarityHit] = []
    context_parts: list[str] = []
    for slug, score in hits_raw:
        page = repo.read(slug)
        if page is None:
            continue
        hits.append(SimilarityHit(slug=slug, title=page.front_matter.title, score=min(score, 1.0)))
        context_parts.append(f"## {page.front_matter.title}\n\n{page.body}")

    wiki_context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant pages found."

    prompt = build_answer_prompt(question=effective_question, wiki_context=wiki_context)
    response = openai_client.chat.completions.create(
        model=settings.openai_model, max_completion_tokens=2048, messages=[{"role": "user", "content": prompt}],
    )
    answer: str = response.choices[0].message.content or ""

    saved_slug: str | None = None
    if save and answer.strip():
        slug = _question_to_slug(question or template or "query_answer")
        repo.write(
            slug=slug, title=question[:80] if question else template or "Query Answer",
            body=answer, sources=["query"], tags=["query-answer"],
        )
        page = repo.read(slug)
        if page:
            emb = embedding_service.embed_page(slug, page.full_text)
            index.upsert(emb)
            index.save()
        saved_slug = slug

    return QueryResult(question=effective_question, hits=hits, answer=answer, saved_slug=saved_slug)
