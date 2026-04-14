"""Configuration and path management."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WikiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = Field(alias="OPENAI_API_KEY")

    openai_model: str = Field(default="gpt-5.2", alias="OPENAI_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")

    query_top_k: int = Field(default=5, alias="QUERY_TOP_K", ge=1, le=20)
    max_source_chars: int = Field(default=50_000, alias="MAX_SOURCE_CHARS")


class WikiPaths:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path.cwd()

    @property
    def wiki_dir(self) -> Path:
        return self.root / "wiki"

    @property
    def sources_dir(self) -> Path:
        return self.root / "sources"

    @property
    def meta_dir(self) -> Path:
        return self.wiki_dir / ".meta"

    @property
    def schema_path(self) -> Path:
        return self.meta_dir / "schema.json"

    @property
    def embeddings_path(self) -> Path:
        return self.meta_dir / "embeddings.json"

    @property
    def index_path(self) -> Path:
        return self.wiki_dir / "index.md"

    @property
    def log_path(self) -> Path:
        return self.wiki_dir / "log.md"

    def page_path(self, slug: str) -> Path:
        return self.wiki_dir / f"{slug}.md"
