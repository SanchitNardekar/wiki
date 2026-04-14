"""Five-step ingest pipeline: Resolve → Route → Synthesize → Embed → Update."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi

from llm_wiki.config import WikiSettings
from llm_wiki.embeddings import EmbeddingService
from llm_wiki.index import EmbeddingIndex
from llm_wiki.models.operations import IngestResult, RouteResult, SynthesisResult
from llm_wiki.models.source import ResolvedSource, SourceType
from llm_wiki.prompts import build_index_update_prompt, build_routing_prompt, build_synthesis_prompt
from llm_wiki.wiki import WikiRepository


class IngestError(Exception):
    """Unrecoverable ingest pipeline error."""


def _extract_youtube_id(url: str) -> str | None:
    parsed = urlparse(url)
    if "youtube.com" in parsed.netloc:
        ids = parse_qs(parsed.query).get("v", [])
        return ids[0] if ids else None
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")
    return None


def _strip_html(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def resolve_source(source_path: str, max_chars: int = 50_000) -> ResolvedSource:
    path = source_path.strip()

    if "youtube.com" in path or "youtu.be" in path:
        video_id = _extract_youtube_id(path)
        if not video_id:
            raise IngestError(f"Could not extract YouTube video ID from: {path!r}")
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join(item["text"] for item in transcript_data)
        return ResolvedSource(
            raw=path, source_type=SourceType.YOUTUBE, name=f"youtube:{video_id}", text=text[:max_chars],
        )

    if path.startswith(("http://", "https://")):
        response = httpx.get(path, follow_redirects=True, timeout=30.0)
        response.raise_for_status()
        text = _strip_html(response.text)
        return ResolvedSource(
            raw=path, source_type=SourceType.HTTP, name=urlparse(path).netloc, text=text[:max_chars],
        )

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Source file not found: {path!r}")

    if file_path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = file_path.read_text(encoding="utf-8", errors="replace")

    return ResolvedSource(
        raw=path, source_type=SourceType.LOCAL_FILE, name=file_path.name, text=text[:max_chars],
    )


def _chat(client: OpenAI, model: str, prompt: str, max_completion_tokens: int = 2048) -> str:
    """Single-turn OpenAI chat completion helper."""
    response = client.chat.completions.create(
        model=model, max_completion_tokens=max_completion_tokens, messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def route(
    client: OpenAI, source: ResolvedSource, schema_context: str, model: str,
) -> RouteResult:
    prompt = build_routing_prompt(schema_context=schema_context, source_text=source.text)
    raw_text = _chat(client, model, prompt, max_completion_tokens=512).strip()
    try:
        slugs: list[str] = json.loads(raw_text)
    except (json.JSONDecodeError, ValueError):
        slugs = []
    return RouteResult(source_name=source.name, relevant_slugs=slugs)


def synthesize(
    client: OpenAI, slug: str, title: str, existing_body: str, source: ResolvedSource, model: str,
) -> SynthesisResult:
    prompt = build_synthesis_prompt(slug=slug, title=title, existing_body=existing_body, source_text=source.text)
    new_body = _chat(client, model, prompt, max_completion_tokens=4096).strip()
    return SynthesisResult(slug=slug, new_body=new_body)


def run_ingest(
    source_path: str,
    repo: WikiRepository,
    index: EmbeddingIndex,
    embedding_service: EmbeddingService,
    openai_client: OpenAI,
    settings: WikiSettings,
) -> IngestResult:
    try:
        source = resolve_source(source_path, max_chars=settings.max_source_chars)
        schema = repo.load_schema()
        schema_context = schema.routing_context()

        route_result = route(client=openai_client, source=source, schema_context=schema_context, model=settings.openai_model)
        valid_slugs = [s for s in route_result.relevant_slugs if schema.get_page(s)]
        touched_slugs: list[str] = []

        for slug in valid_slugs:
            page_spec = schema.get_page(slug)
            assert page_spec is not None

            existing = repo.read(slug)
            existing_body = existing.body if existing else ""
            existing_sources = existing.front_matter.sources if existing else []
            existing_tags = existing.front_matter.tags if existing else []

            synth = synthesize(
                client=openai_client, slug=slug, title=page_spec.title,
                existing_body=existing_body, source=source, model=settings.openai_model,
            )

            merged_sources = list(dict.fromkeys([*existing_sources, source.name]))
            repo.write(slug=slug, title=page_spec.title, body=synth.new_body, sources=merged_sources, tags=existing_tags)

            updated_page = repo.read(slug)
            if updated_page:
                embedding = embedding_service.embed_page(slug, updated_page.full_text)
                index.upsert(embedding)

            touched_slugs.append(slug)

        index.save()

        all_pages = [
            (p.slug, p.title, repo.read(p.slug).front_matter.tags if repo.read(p.slug) else [])
            for p in schema.pages
        ]
        index_prompt = build_index_update_prompt(schema_context, all_pages)
        index_content = _chat(openai_client, settings.openai_model, index_prompt)
        repo.write_index(index_content)

        result = IngestResult(source_name=source.name, slugs_touched=touched_slugs)
        repo.append_log(result.log_line())
        return result

    except (FileNotFoundError, IngestError) as exc:
        return IngestResult(source_name=source_path, slugs_touched=[], success=False, error=str(exc))
