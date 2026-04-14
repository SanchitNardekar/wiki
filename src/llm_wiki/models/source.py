"""Pydantic models for source resolution."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class SourceType(StrEnum):
    LOCAL_FILE = "local_file"
    YOUTUBE = "youtube"
    HTTP = "http"


class ResolvedSource(BaseModel):
    raw: str
    source_type: SourceType
    name: str
    text: str

    @property
    def token_estimate(self) -> int:
        return len(self.text) // 4


class IngestRequest(BaseModel):
    source_path: str
    force: bool = False

    @field_validator("source_path")
    @classmethod
    def path_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("source_path must not be empty")
        return v.strip()
