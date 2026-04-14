"""WikiPage and WikiRepository — CRUD for the wiki's markdown files."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import frontmatter

from llm_wiki.models.frontmatter import WikiFrontMatter, WikiSchema


class WikiPage:
    def __init__(self, path: Path, front_matter: WikiFrontMatter, body: str) -> None:
        self.path = path
        self.front_matter = front_matter
        self.body = body

    @property
    def full_text(self) -> str:
        return f"# {self.front_matter.title}\n\n{self.body}"

    def save(self) -> None:
        post = frontmatter.Post(self.body, **self.front_matter.model_dump(mode="json"))
        self.path.write_text(frontmatter.dumps(post))


class WikiRepository:
    def __init__(self, wiki_dir: Path) -> None:
        self._dir = wiki_dir
        self._meta_dir = wiki_dir / ".meta"

    def _page_path(self, slug: str) -> Path:
        return self._dir / f"{slug}.md"

    def load_schema(self) -> WikiSchema:
        return WikiSchema.model_validate_json((self._meta_dir / "schema.json").read_text())

    def save_schema(self, schema: WikiSchema) -> None:
        (self._meta_dir / "schema.json").write_text(schema.model_dump_json(indent=2))

    def read(self, slug: str) -> WikiPage | None:
        path = self._page_path(slug)
        if not path.exists():
            return None
        post = frontmatter.load(str(path))
        fm = WikiFrontMatter.model_validate(dict(post.metadata))
        return WikiPage(path=path, front_matter=fm, body=post.content)

    def write(
        self, slug: str, title: str, body: str, sources: list[str], tags: list[str],
    ) -> WikiPage:
        fm = WikiFrontMatter(slug=slug, title=title, tags=tags, sources=sources, updated=date.today())
        page = WikiPage(path=self._page_path(slug), front_matter=fm, body=body)
        page.save()
        return page

    def list_slugs(self) -> list[str]:
        return [p.stem for p in self._dir.glob("*.md") if p.stem not in ("index", "log")]

    def write_index(self, content: str) -> None:
        (self._dir / "index.md").write_text(content)

    def append_log(self, entry: str) -> None:
        with (self._dir / "log.md").open("a") as f:
            f.write(f"\n{entry}")

    def read_log(self) -> str:
        return (self._dir / "log.md").read_text()
