"""Pydantic models for wiki page frontmatter and schema governance."""

from __future__ import annotations

import re
from datetime import date

from pydantic import BaseModel, Field, field_validator, model_validator

_SLUG_PATTERN = re.compile(r"[a-z0-9][a-z0-9_\-]*")


def _validate_slug(v: str) -> str:
    if not _SLUG_PATTERN.fullmatch(v):
        raise ValueError(
            f"Slug must be lowercase alphanumeric (underscores/hyphens allowed), got: {v!r}"
        )
    return v


class PageSpec(BaseModel):
    slug: str
    title: str
    description: str

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
    name: str
    pages: list[PageSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def slugs_are_unique(self) -> WikiSchema:
        slugs = [p.slug for p in self.pages]
        if len(slugs) != len(set(slugs)):
            duplicates = sorted({s for s in slugs if slugs.count(s) > 1})
            raise ValueError(f"Duplicate slugs found in schema: {duplicates}")
        return self

    def routing_context(self) -> str:
        return "\n".join(f"{p.slug}: {p.title} — {p.description}" for p in self.pages)

    def get_page(self, slug: str) -> PageSpec | None:
        return next((p for p in self.pages if p.slug == slug), None)

    def slug_set(self) -> frozenset[str]:
        return frozenset(p.slug for p in self.pages)


class WikiFrontMatter(BaseModel):
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
        if source_name in self.sources:
            return self
        return self.model_copy(
            update={"sources": [*self.sources, source_name], "updated": date.today()}
        )
