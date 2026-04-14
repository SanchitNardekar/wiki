# LLM Wiki — Agent Runtime Rules

Read this before touching wiki/*.md files.

## Architecture

```
sources/        ← Raw input. IMMUTABLE. Never modify.
wiki/*.md       ← Compiled knowledge. You maintain this.
wiki/.meta/     ← Schema + embeddings. Human manages schema.json.
```

## Core Invariants

1. **Never discard knowledge.** Preserve all existing content when synthesising. Note contradictions explicitly — never silently overwrite.
2. **Every write is validated.** Use `WikiRepository.write()` only. Never `Path.write_text()` on wiki pages.
3. **Cross-references use `[[slug]]` notation.** Slugs are lowercase alphanumeric with underscores/hyphens.
4. **Provenance is always tracked.** Every page's front matter `sources:` field must list all contributing sources.
5. **The schema is the page universe.** Never create pages for slugs not in schema.json.

## Front Matter Schema

```yaml
---
slug: example_concept
title: Example Concept
tags: [tag1, tag2]
sources: [source_file.md]
updated: "YYYY-MM-DD"
---
```

## Ingest Pipeline (5 steps)

1. **Resolve** — detect source type, extract text
2. **Route** — return JSON array of relevant slugs
3. **Synthesize** — rewrite each page, extending existing content
4. **Embed** — re-embed updated pages into embeddings.json
5. **Update** — regenerate index.md, append to log.md

## File Ownership

| File | Owner | Rule |
|------|-------|------|
| sources/*.* | Human | Never modify |
| wiki/.meta/schema.json | Human | Never modify without instruction |
| wiki/*.md | LLM | Via WikiRepository |
| wiki/.meta/embeddings.json | LLM | Sync after every write |
| wiki/index.md | LLM | Regenerate after ingest |
| wiki/log.md | LLM | Append-only |

## Quality

- Dense content, not thin summaries
- Liberal cross-references to strengthen the knowledge graph
- Surface contradictions, don't hide them
- Run `llm-wiki lint` after batch ingests
