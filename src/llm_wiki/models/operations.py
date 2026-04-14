"""Pydantic models for all pipeline operation inputs and outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class RouteResult(BaseModel):
    source_name: str
    relevant_slugs: list[str] = Field(default_factory=list)

    @field_validator("relevant_slugs", mode="before")
    @classmethod
    def coerce_slugs(cls, v: object) -> list[str]:
        if not isinstance(v, list):
            return []
        return [str(s).lower().strip() for s in v if s]


class SynthesisResult(BaseModel):
    slug: str
    new_body: str

    @field_validator("new_body")
    @classmethod
    def body_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Synthesised body must not be empty")
        return v.strip()


class IngestResult(BaseModel):
    source_name: str
    slugs_touched: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    success: bool = True
    error: str | None = None

    def log_line(self) -> str:
        ts = self.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
        slugs = ", ".join(self.slugs_touched) if self.slugs_touched else "none"
        return f"- {ts} — Ingest: {self.source_name} → touched: [{slugs}]"


class SimilarityHit(BaseModel):
    slug: str
    title: str
    score: float = Field(ge=0.0, le=1.0)


class QueryResult(BaseModel):
    question: str
    hits: list[SimilarityHit] = Field(default_factory=list)
    answer: str
    saved_slug: str | None = None


class LintIssueKind(StrEnum):
    ORPHANED_PAGE = "orphaned_page"
    MISSING_PAGE = "missing_page"
    BROKEN_XREF = "broken_xref"
    STALE_EMBEDDING = "stale_embedding"
    MISSING_PROVENANCE = "missing_provenance"
    CONTRADICTION = "contradiction"


class LintIssue(BaseModel):
    kind: LintIssueKind
    slug: str | None = None
    detail: str

    def __str__(self) -> str:
        prefix = f"[{self.slug}] " if self.slug else ""
        return f"{prefix}{self.kind}: {self.detail}"


class LintResult(BaseModel):
    issues: list[LintIssue] = Field(default_factory=list)
    pages_checked: int = 0
    deep: bool = False

    @property
    def passed(self) -> bool:
        return len(self.issues) == 0

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    def issues_of_kind(self, kind: LintIssueKind) -> list[LintIssue]:
        return [i for i in self.issues if i.kind == kind]
