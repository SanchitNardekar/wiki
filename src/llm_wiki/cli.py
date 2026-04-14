"""Typer CLI — six commands wiring all pipeline modules together."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from openai import OpenAI
from rich.console import Console
from rich.table import Table

from llm_wiki.config import WikiPaths, WikiSettings
from llm_wiki.embeddings import EmbeddingService
from llm_wiki.index import EmbeddingIndex
from llm_wiki.ingest import run_ingest
from llm_wiki.lint import run_lint
from llm_wiki.models.embeddings import EmbeddingIndexData
from llm_wiki.models.frontmatter import PageSpec, WikiSchema
from llm_wiki.prompts import QUERY_TEMPLATES
from llm_wiki.query import run_query
from llm_wiki.wiki import WikiRepository

app = typer.Typer(name="llm-wiki", help="Compiling knowledge that compounds.", add_completion=False)
console = Console()


def _build_dependencies(
    paths: WikiPaths, settings: WikiSettings,
) -> tuple[WikiRepository, EmbeddingIndex, EmbeddingService, OpenAI]:
    repo = WikiRepository(paths.wiki_dir)
    index = EmbeddingIndex(paths.embeddings_path)
    client = OpenAI(api_key=settings.openai_api_key)
    embedding_service = EmbeddingService(client=client, model=settings.embedding_model)
    return repo, index, embedding_service, client


@app.command()
def init(
    root: Annotated[Path, typer.Option("--root", help="Wiki root directory")] = Path("."),
) -> None:
    """Bootstrap the wiki directory structure from the schema template."""
    paths = WikiPaths(root=root)
    paths.wiki_dir.mkdir(parents=True, exist_ok=True)
    paths.meta_dir.mkdir(parents=True, exist_ok=True)
    paths.sources_dir.mkdir(parents=True, exist_ok=True)

    if not paths.schema_path.exists():
        schema = WikiSchema(name="My Wiki", pages=[PageSpec(slug="example", title="Example Page", description="Replace with your first concept")])
        paths.schema_path.write_text(schema.model_dump_json(indent=2))
        console.print("[green]✓[/green] Created wiki/.meta/schema.json")
    else:
        console.print("[yellow]schema.json already exists — skipping.[/yellow]")

    if not paths.embeddings_path.exists():
        paths.embeddings_path.write_text(EmbeddingIndexData().model_dump_json(indent=2))
    if not paths.index_path.exists():
        paths.index_path.write_text("# Index\n\nNo pages yet.\n")
    if not paths.log_path.exists():
        paths.log_path.write_text("# Log\n\n")

    console.print("[bold green]Wiki initialised.[/bold green]")
    console.print(f"  Schema: [cyan]{paths.schema_path}[/cyan]")
    console.print("\nNext: edit schema.json, add sources, run [bold]llm-wiki ingest[/bold]")


@app.command()
def ingest(
    source: Annotated[str, typer.Argument(help="File path, directory, YouTube URL, or HTTP URL")],
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:
    """Ingest a source into the wiki."""
    settings = WikiSettings()
    paths = WikiPaths(root=root)
    repo, index, embedding_service, client = _build_dependencies(paths, settings)

    source_path = Path(source)
    if source_path.is_dir():
        files = list(source_path.glob("*.*"))
        console.print(f"Ingesting {len(files)} files from {source_path}/")
        for f in files:
            _run_single_ingest(str(f), repo, index, embedding_service, client, settings)
        return

    _run_single_ingest(source, repo, index, embedding_service, client, settings)


def _run_single_ingest(
    source_path: str, repo: WikiRepository, index: EmbeddingIndex,
    embedding_service: EmbeddingService, client: OpenAI, settings: WikiSettings,
) -> None:
    name = Path(source_path).name if not source_path.startswith("http") else source_path
    console.rule(f"Ingesting: {name}")
    result = run_ingest(
        source_path=source_path, repo=repo, index=index,
        embedding_service=embedding_service, openai_client=client, settings=settings,
    )
    if result.success:
        for slug in result.slugs_touched:
            console.print(f"  [green]✓[/green] wiki/{slug}.md written")
        console.print("[bold]Ingest complete.[/bold]")
    else:
        console.print(f"[red]✗ Ingest failed:[/red] {result.error}")
        raise typer.Exit(1)


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Question to ask the wiki")] = "",
    root: Annotated[Path, typer.Option("--root")] = Path("."),
    save: Annotated[bool, typer.Option("--save")] = False,
    template: Annotated[str | None, typer.Option("--template", "-t")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    top_k: Annotated[int | None, typer.Option("--top-k")] = None,
) -> None:
    """Query the compiled wiki."""
    if not question and not template:
        console.print("[red]Provide a question or use --template <name>[/red]")
        raise typer.Exit(1)

    settings = WikiSettings()
    if top_k is not None:
        settings = settings.model_copy(update={"query_top_k": top_k})

    paths = WikiPaths(root=root)
    repo, index, embedding_service, client = _build_dependencies(paths, settings)

    result = run_query(
        question=question, repo=repo, index=index, embedding_service=embedding_service,
        openai_client=client, settings=settings, save=save, template=template, verbose=verbose,
    )

    if verbose and result.hits:
        table = Table(title="Retrieved Wiki Pages")
        table.add_column("Rank", style="cyan")
        table.add_column("Page")
        table.add_column("Score", justify="right")
        for i, hit in enumerate(result.hits, 1):
            table.add_row(str(i), hit.title, f"{hit.score:.4f}")
        console.print(table)

    console.print(result.answer)
    if result.saved_slug:
        console.print(f"\n[green]✓[/green] Saved as wiki/{result.saved_slug}.md")


@app.command()
def lint(
    root: Annotated[Path, typer.Option("--root")] = Path("."),
    deep: Annotated[bool, typer.Option("--deep")] = False,
    fix: Annotated[bool, typer.Option("--fix")] = False,
) -> None:
    """Run structural health checks on the wiki."""
    settings = WikiSettings()
    paths = WikiPaths(root=root)
    repo, index, embedding_service, client = _build_dependencies(paths, settings)

    if fix:
        console.print("Re-embedding all pages...")
        for slug in repo.list_slugs():
            page = repo.read(slug)
            if page:
                index.upsert(embedding_service.embed_page(slug, page.full_text))
                console.print(f"  [green]✓[/green] {slug}")
        index.save()

    result = run_lint(
        repo=repo, index=index, deep=deep,
        openai_client=client if deep else None, model=settings.openai_model,
    )

    console.rule("Lint Report")
    if result.passed:
        console.print("[green]All checks passed.[/green]")
    else:
        for issue in result.issues:
            console.print(f"  [red]✗[/red] {issue}")
    console.print(f"\nPages checked: {result.pages_checked}")
    if not result.passed:
        raise typer.Exit(1)


@app.command()
def status(
    root: Annotated[Path, typer.Option("--root")] = Path("."),
) -> None:
    """Display a dashboard overview of the wiki state."""
    paths = WikiPaths(root=root)
    if not paths.wiki_dir.exists():
        console.print("[red]Wiki not initialised. Run `llm-wiki init` first.[/red]")
        raise typer.Exit(1)

    repo = WikiRepository(paths.wiki_dir)
    schema = repo.load_schema()
    index = EmbeddingIndex(paths.embeddings_path)
    slugs = repo.list_slugs()
    sources = list(paths.sources_dir.glob("*.*")) if paths.sources_dir.exists() else []

    console.rule("Wiki Status")
    console.print(f"  Wiki: [bold]{schema.name}[/bold]")
    console.print(f"  Pages: {len(slugs)}/{len(schema.pages)} | Embedded: {index.size} | Sources: {len(sources)}")

    if slugs:
        table = Table(title="Pages")
        table.add_column("Slug", style="cyan")
        table.add_column("Title")
        table.add_column("Tags")
        table.add_column("Updated")
        for slug in sorted(slugs):
            page = repo.read(slug)
            if page:
                fm = page.front_matter
                table.add_row(fm.slug, fm.title, ", ".join(fm.tags) or "—", str(fm.updated))
        console.print(table)

    log = repo.read_log()
    lines = [l for l in log.splitlines() if l.strip().startswith("-")]
    if lines:
        console.print("\nRecent activity:")
        for line in lines[-5:]:
            console.print(f"  {line}")


@app.command()
def prompts(
    category: Annotated[str | None, typer.Option("--category", "-c")] = None,
) -> None:
    """List all available query templates."""
    table = Table(title="Query Templates")
    table.add_column("Name", style="cyan")
    table.add_column("Category", style="yellow")
    table.add_column("Description")

    for name, tmpl in sorted(QUERY_TEMPLATES.items()):
        if category and tmpl.category.lower() != category.lower():
            continue
        table.add_row(name, tmpl.category, tmpl.description)

    console.print(table)
    console.print("\n[dim]Use: llm-wiki query --template <name>[/dim]")
