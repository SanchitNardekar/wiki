# LLM Wiki

Karpathy's LLM Wiki pattern — compiling knowledge that compounds.

Feed in sources (files, PDFs, YouTube, URLs), and the LLM synthesises them into a cross-referenced markdown wiki. Unlike RAG, knowledge is compiled once and built upon — queries run over pre-synthesised pages, not raw document chunks.

## Architecture

```
sources/           ← Raw input (human-managed, immutable)
wiki/*.md          ← Compiled knowledge (LLM-maintained)
wiki/.meta/        ← Schema + embedding index
```

## Quick Start

```bash
uv sync
llm-wiki init                              # bootstrap directory structure
# edit wiki/.meta/schema.json to define your pages
llm-wiki ingest sources/paper.pdf          # ingest a file
llm-wiki ingest sources/                   # ingest a whole directory
llm-wiki ingest "https://youtube.com/..."  # ingest a URL
llm-wiki query "What is attention?"        # query the wiki
llm-wiki query --template blind-spot       # use a named template
llm-wiki lint                              # structural health checks
llm-wiki status                            # dashboard overview
```

## Commands

| Command | Description |
|---------|-------------|
| `init` | Bootstrap wiki directory structure from schema |
| `ingest <source>` | Route → Synthesize → Embed → Update index & log |
| `query <question>` | Embed → Search → Assemble → Answer (optional `--save`) |
| `lint` | Structural checks; `--deep` for LLM contradiction analysis |
| `status` | Wiki dashboard |
| `prompts` | List 24+ query templates across 6 categories |

## Supported Sources

| Type | Example |
|------|---------|
| Text / Markdown | `sources/notes.md` |
| PDF | `sources/paper.pdf` |
| YouTube | `https://youtube.com/watch?v=...` (transcript) |
| HTTP | `https://arxiv.org/abs/...` (HTML → text) |

## Planned Improvements

- **Image support** — Add vision model integration (e.g. Claude vision API) to `resolve_source()` for ingesting diagrams, screenshots, and images with auto-generated descriptions. Currently the pipeline is text-only.
- **Inline source citations** — Extend `build_synthesis_prompt()` to instruct the LLM to attribute individual claims to their source within the page body. Currently provenance is tracked at the page level (front matter `sources:` field) but not per-claim.

## Stack

Python 3.13 · uv · OpenAI API (GPT-5.2 + embeddings) · Pydantic v2 · Typer · Rich
