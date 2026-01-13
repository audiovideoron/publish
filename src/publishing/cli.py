"""Distillyzer CLI - Personal learning accelerator."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from . import db, harvest as harv, transcribe, embed as emb, query as q, visualize as viz, extract as ext, artifacts as art, search_queries, scoring

app = typer.Typer(help="Distillyzer - Harvest knowledge, query it, use it.")
artifacts_app = typer.Typer(help="Manage and use extracted artifacts.")
app.add_typer(artifacts_app, name="artifacts")
skills_app = typer.Typer(help="Manage presentation skills.")
app.add_typer(skills_app, name="skills")
projects_app = typer.Typer(help="Manage projects with faceted organization.")
app.add_typer(projects_app, name="project")
sources_app = typer.Typer(help="Manage knowledge sources (channels, repos).")
app.add_typer(sources_app, name="sources")
items_app = typer.Typer(help="Manage harvested items (videos, articles, code files).")
app.add_typer(items_app, name="items")
suggest_app = typer.Typer(help="Get content suggestions based on project facets.")
app.add_typer(suggest_app, name="suggest")
console = Console()


@app.command()
def search(query: str, limit: int = 10):
    """Search YouTube for videos on a topic."""
    console.print(f"[yellow]Searching YouTube for:[/yellow] {query}\n")

    try:
        videos = harv.search_youtube(query, limit=limit)
        if not videos:
            console.print("[dim]No results found[/dim]")
            return

        table = Table(show_header=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Title", style="cyan", no_wrap=False)
        table.add_column("Channel", style="green")
        table.add_column("Duration", style="yellow", justify="right")

        for i, video in enumerate(videos, 1):
            duration = video.get("duration")
            if duration:
                duration = int(duration)
                duration_str = f"{duration // 60}:{duration % 60:02d}"
            else:
                duration_str = "?"
            table.add_row(
                str(i),
                video["title"][:60] + ("..." if len(video.get("title", "")) > 60 else ""),
                video.get("channel", "?")[:20],
                duration_str,
            )

        console.print(table)
        console.print("\n[dim]To harvest a video:[/dim] dz harvest <url>")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@app.command()
def harvest(
    url: str,
    skip_transcribe: bool = typer.Option(False, "--skip-transcribe", help="Skip transcription"),
):
    """Harvest a YouTube video or GitHub repo."""
    console.print(f"[yellow]Harvesting:[/yellow] {url}\n")

    try:
        # Detect URL type
        if "github.com" in url:
            # GitHub repo
            console.print("[cyan]Detected GitHub repo[/cyan]")
            result = harv.harvest_repo(url)

            if result["status"] == "already_exists":
                console.print(f"[yellow]Already harvested:[/yellow] {result['name']}")
                return

            console.print(f"[green]Cloned:[/green] {result['name']}")
            console.print(f"[green]Files indexed:[/green] {result['files_indexed']}")

            # Embed code files
            if result.get("file_items"):
                console.print("\n[yellow]Embedding code files...[/yellow]")

                def progress(current, total, path, chunks):
                    console.print(f"  [{current}/{total}] {path} ({chunks} chunks)")

                embed_result = emb.embed_repo_files(
                    result["file_items"],
                    progress_callback=progress,
                )

                console.print(f"\n[green]Embedded:[/green] {embed_result['total_files']} files, {embed_result['total_chunks']} chunks")

                if embed_result["errors"]:
                    console.print(f"[yellow]Errors:[/yellow] {len(embed_result['errors'])}")
                    for err in embed_result["errors"][:3]:
                        console.print(f"  - {err['path']}: {err['error']}")

                # Auto-regenerate index
                console.print("[yellow]Updating index...[/yellow]")
                _regenerate_index()
                console.print("[green]Index updated[/green]")

        elif "youtube.com" in url or "youtu.be" in url:
            # YouTube video
            console.print("[cyan]Detected YouTube video[/cyan]")
            result = harv.harvest_video(url)

            if result["status"] == "already_exists":
                console.print(f"[yellow]Already harvested:[/yellow] {result['title']}")
                return

            console.print(f"[green]Downloaded:[/green] {result['title']}")
            console.print(f"[dim]Channel:[/dim] {result['channel']}")
            dur = int(result['duration']) if result.get('duration') else 0
            console.print(f"[dim]Duration:[/dim] {dur // 60}:{dur % 60:02d}")

            if not skip_transcribe:
                console.print("\n[yellow]Transcribing...[/yellow]")
                transcript = transcribe.transcribe_audio(result["audio_path"])
                console.print(f"[green]Transcribed:[/green] {len(transcript['text'])} characters")

                # Convert to timed chunks and embed
                console.print("[yellow]Embedding...[/yellow]")
                timed_chunks = transcribe.segments_to_timed_chunks(transcript["segments"])
                num_chunks = emb.embed_transcript_chunks(result["item_id"], timed_chunks)
                console.print(f"[green]Stored:[/green] {num_chunks} chunks")

                # Cleanup audio file
                Path(result["audio_path"]).unlink(missing_ok=True)

                # Auto-regenerate index
                console.print("[yellow]Updating index...[/yellow]")
                _regenerate_index()
                console.print("[green]Index updated[/green]")
        else:
            # Assume it's an article URL
            console.print("[cyan]Detected article URL[/cyan]")
            result = harv.harvest_article(url)

            if result["status"] == "already_exists":
                console.print(f"[yellow]Already harvested:[/yellow] {result['title']}")
                return

            console.print(f"[green]Harvested:[/green] {result['title']}")
            if result.get("author"):
                console.print(f"[dim]Author:[/dim] {result['author']}")
            console.print(f"[dim]Site:[/dim] {result.get('sitename', 'Unknown')}")

            # Embed the article content
            console.print("\n[yellow]Embedding...[/yellow]")
            num_chunks = emb.embed_text_content(
                result["item_id"],
                result["content"],
                is_code=False,
            )
            console.print(f"[green]Stored:[/green] {num_chunks} chunks")

            # Auto-regenerate index
            console.print("[yellow]Updating index...[/yellow]")
            _regenerate_index()
            console.print("[green]Index updated[/green]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise


@app.command("harvest-channel")
def harvest_channel(
    channel_url: str,
    limit: int = typer.Option(10, "--limit", "-l", help="Max videos to list"),
):
    """List videos from a YouTube channel (for selective harvesting)."""
    console.print(f"[yellow]Fetching channel videos:[/yellow] {channel_url}\n")

    try:
        videos = harv.harvest_channel(channel_url, limit=limit)

        table = Table(show_header=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Title", style="cyan")
        table.add_column("URL", style="dim")

        for i, video in enumerate(videos, 1):
            table.add_row(str(i), video["title"][:50], video["url"])

        console.print(table)
        console.print("\n[dim]To harvest a video:[/dim] dz harvest <url>")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@app.command()
def query(
    question: str,
    sources: int = typer.Option(5, "--sources", "-s", help="Number of sources to retrieve"),
):
    """Query your knowledge base."""
    console.print(f"[yellow]Querying:[/yellow] {question}\n")

    try:
        result = q.ask(question, num_sources=sources)

        # Display answer
        console.print(Panel(Markdown(result["answer"]), title="Answer", border_style="green"))

        # Display sources
        if result["sources"]:
            console.print("\n[dim]Sources:[/dim]")
            for i, src in enumerate(result["sources"], 1):
                title = src["item_title"][:50]
                ts = q.format_timestamp(src.get("timestamp_start"))
                ts_str = f" @ {ts}" if ts else ""
                sim = f"{src['similarity']:.2f}"
                console.print(f"  {i}. [cyan]{title}[/cyan]{ts_str} [dim](sim: {sim})[/dim]")

        console.print(f"\n[dim]Tokens used: {result['tokens_used']}[/dim]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@app.command()
def chat():
    """Interactive chat with your knowledge base."""
    console.print("[yellow]Starting chat...[/yellow]")
    console.print("[dim]Type 'quit' or 'exit' to end. Press Ctrl+C to cancel.[/dim]\n")

    history = []

    try:
        while True:
            question = console.input("[bold blue]You:[/bold blue] ")

            if question.lower() in ("quit", "exit", "q"):
                console.print("[yellow]Goodbye![/yellow]")
                break

            if not question.strip():
                continue

            result = q.chat_turn(question, history, num_sources=5)

            # Display answer
            console.print(f"\n[bold green]Assistant:[/bold green] {result['answer']}\n")

            # Update history
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": result["answer"]})

            # Keep history manageable
            if len(history) > 20:
                history = history[-20:]

    except KeyboardInterrupt:
        console.print("\n[yellow]Chat ended.[/yellow]")


def _format_timestamp(seconds: float | None) -> str:
    """Format seconds as MM:SS."""
    if seconds is None:
        return "0:00"
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"


def _make_timestamp_link(url: str, timestamp: float | None) -> str:
    """Create a URL with timestamp for YouTube, or plain URL for others."""
    if timestamp is None:
        return url
    if "youtube.com" in url or "youtu.be" in url:
        # YouTube supports &t=SECONDS
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}t={int(timestamp)}"
    # Other sites: just return the URL (timestamp shown separately)
    return url


def _generate_index_html_grouped(grouped_items: dict[str, list[dict]], output_path: Path) -> None:
    """Generate HTML index file grouped by source with type-aware rendering."""
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        "<title>Distillyzer Knowledge Index</title>",
        "<style>",
        "body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; line-height: 1.6; }",
        "h1 { color: #333; }",
        "ul { list-style: none; padding: 0; }",
        "li { margin: 0.5em 0; padding: 0.5em; background: #f9f9f9; border-radius: 4px; }",
        "a { color: #0066cc; text-decoration: none; }",
        "a:hover { text-decoration: underline; }",
        ".preview { color: #333; }",
        ".source-section { margin: 2em 0; padding: 1em; border: 1px solid #ddd; border-radius: 8px; }",
        ".source-header { font-size: 1.3em; font-weight: bold; color: #333; margin-bottom: 0.5em; }",
        ".source-type { font-size: 0.75em; color: #666; font-weight: normal; margin-left: 0.5em; }",
        "details { margin: 0.5em 0; }",
        "summary { cursor: pointer; font-size: 1.05em; font-weight: 600; color: #555; padding: 0.5em; background: #f0f0f0; border-radius: 4px; }",
        "summary:hover { background: #e8e8e8; }",
        ".item-type { font-size: 0.7em; font-weight: normal; color: #888; margin-left: 0.5em; padding: 0.1em 0.4em; background: #e0e0e0; border-radius: 3px; }",
        ".chunk-count { font-size: 0.7em; font-weight: normal; color: #888; margin-left: 0.5em; }",
        "/* Video styles */",
        ".video .timestamp { font-family: monospace; color: #0066cc; margin-right: 1em; }",
        "/* Article styles */",
        ".article .part-ref { font-family: monospace; color: #666; margin-right: 1em; font-size: 0.9em; }",
        "/* Code styles */",
        ".code .chunk-ref { font-family: monospace; color: #666; margin-right: 1em; font-size: 0.85em; }",
        ".code .preview { font-family: monospace; font-size: 0.85em; background: #f5f5f5; padding: 0.2em 0.4em; border-radius: 2px; }",
        "</style>",
        "</head><body>",
        "<h1>Distillyzer Knowledge Index</h1>",
    ]

    for source_name, items in grouped_items.items():
        total_chunks = sum(len(item["chunks"]) for item in items)
        # Determine source type from first item
        source_type = items[0].get("type", "unknown") if items else "unknown"
        source_type_label = {"video": "YouTube", "article": "Website", "code_file": "Code"}.get(source_type, source_type)

        html_parts.append("<div class='source-section'>")
        html_parts.append(f"<div class='source-header'>{source_name}<span class='source-type'>{source_type_label}</span></div>")

        for item in items:
            chunk_count = len(item["chunks"])
            item_type = item.get("type", "unknown")
            type_label = {"video": "video", "article": "article", "code_file": "code"}.get(item_type, item_type)

            html_parts.append(f"<details class='{type_label}'>")
            html_parts.append(f"<summary>{item['title']}<span class='item-type'>{type_label}</span><span class='chunk-count'>({chunk_count} chunks)</span></summary>")
            html_parts.append("<ul>")

            for i, chunk in enumerate(item["chunks"]):
                preview = chunk["content"][:80].replace("<", "&lt;").replace(">", "&gt;")
                if len(chunk["content"]) > 80:
                    preview += "..."

                if item_type == "video":
                    # Video: timestamp link
                    ts = chunk["timestamp_start"]
                    ts_str = _format_timestamp(ts)
                    link = _make_timestamp_link(item["url"], ts) if item["url"] else "#"
                    html_parts.append(
                        f'<li><a href="{link}"><span class="timestamp">{ts_str}</span></a>'
                        f'<span class="preview">{preview}</span></li>'
                    )
                elif item_type == "article":
                    # Article: part reference, no timestamp
                    part_num = i + 1
                    link = item["url"] or "#"
                    html_parts.append(
                        f'<li><a href="{link}"><span class="part-ref">[Part {part_num}]</span></a>'
                        f'<span class="preview">{preview}</span></li>'
                    )
                elif item_type == "code_file":
                    # Code: chunk reference with monospace preview
                    chunk_num = i + 1
                    link = item["url"] or "#"
                    html_parts.append(
                        f'<li><a href="{link}"><span class="chunk-ref">[{chunk_num}]</span></a>'
                        f'<span class="preview">{preview}</span></li>'
                    )
                else:
                    # Unknown type: basic rendering
                    link = item["url"] or "#"
                    html_parts.append(
                        f'<li><a href="{link}">'
                        f'<span class="preview">{preview}</span></a></li>'
                    )

            html_parts.append("</ul>")
            html_parts.append("</details>")

        html_parts.append("</div>")

    html_parts.extend(["</body></html>"])
    output_path.write_text("\n".join(html_parts))


def _regenerate_index(output_path: Path = Path("index.html")) -> None:
    """Regenerate the HTML index file."""
    grouped = db.get_items_grouped_by_source()
    _generate_index_html_grouped(grouped, output_path)


@app.command()
def index(
    output: str = typer.Option("index.html", "--output", "-o", help="Output file path"),
):
    """Generate HTML index with timestamp links, grouped by source."""
    console.print("[yellow]Generating index...[/yellow]\n")

    try:
        output_path = Path(output)
        _regenerate_index(output_path)

        grouped = db.get_items_grouped_by_source()
        total_items = sum(len(items) for items in grouped.values())
        total_chunks = sum(
            sum(len(item["chunks"]) for item in items)
            for items in grouped.values()
        )

        console.print(f"[green]Generated:[/green] {output_path}")
        console.print(f"[dim]Sources:[/dim] {len(grouped)}")
        console.print(f"[dim]Items:[/dim] {total_items}")
        console.print(f"[dim]Chunks:[/dim] {total_chunks}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise


@app.command()
def stats():
    """Show statistics about your knowledge base."""
    console.print("[yellow]Knowledge base stats[/yellow]\n")

    try:
        s = db.get_stats()
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green", justify="right")

        table.add_row("Sources (channels, repos)", str(s["sources"]))
        table.add_row("Items (videos, files)", str(s["items"]))
        table.add_row("Chunks (searchable)", str(s["chunks"]))

        console.print(table)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@app.command()
def embed(
    item_id: int = typer.Argument(None, help="Item ID to re-embed (omit for --all)"),
    all_items: bool = typer.Option(False, "--all", "-a", help="Re-embed all items"),
):
    """Re-embed existing items without re-harvesting.

    Deletes old chunks and creates new embeddings using current settings.
    Useful when embedding model or chunking strategy changes.

    Examples:
        dz embed 5           # Re-embed item with ID 5
        dz embed --all       # Re-embed all items
    """
    try:
        if all_items:
            console.print("[yellow]Re-embedding all items...[/yellow]\n")

            def progress(current, total, title, status):
                status_icon = "[green]OK[/green]" if status == "success" else "[red]FAIL[/red]"
                title_short = title[:50] if title else "?"
                console.print(f"  [{current}/{total}] {title_short} {status_icon}")

            result = emb.reembed_all_items(progress_callback=progress)

            console.print(f"\n[green]Completed:[/green] {result['successful']}/{result['total_items']} items")
            console.print(f"[dim]Old chunks deleted:[/dim] {result['total_old_chunks']}")
            console.print(f"[dim]New chunks created:[/dim] {result['total_new_chunks']}")

            if result["errors"]:
                console.print(f"\n[yellow]Errors ({len(result['errors'])}):[/yellow]")
                for err in result["errors"][:5]:
                    console.print(f"  - [{err['item_id']}] {err['title']}: {err['error']}")
                if len(result["errors"]) > 5:
                    console.print(f"  ... and {len(result['errors']) - 5} more")

            # Update index after re-embedding
            console.print("\n[yellow]Updating index...[/yellow]")
            _regenerate_index()
            console.print("[green]Index updated[/green]")

        elif item_id is not None:
            console.print(f"[yellow]Re-embedding item {item_id}...[/yellow]\n")

            result = emb.reembed_item(item_id)

            if result["status"] == "success":
                console.print(f"[green]Re-embedded:[/green] {result['title']}")
                console.print(f"[dim]Type:[/dim] {result['type']}")
                console.print(f"[dim]Old chunks deleted:[/dim] {result['old_chunks']}")
                console.print(f"[dim]New chunks created:[/dim] {result['new_chunks']}")

                # Update index after re-embedding
                console.print("\n[yellow]Updating index...[/yellow]")
                _regenerate_index()
                console.print("[green]Index updated[/green]")
            else:
                console.print(f"[red]Error:[/red] {result['error']}")

        else:
            console.print("[red]Error:[/red] Provide an item ID or use --all")
            console.print("[dim]Examples:[/dim]")
            console.print("  dz embed 5       # Re-embed item 5")
            console.print("  dz embed --all   # Re-embed all items")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise


@app.command()
def visualize(
    concept: str,
    output_dir: str = typer.Option("images", "--output", "-o", help="Output directory"),
    chunk_id: int = typer.Option(None, "--chunk", "-c", help="Generate from specific chunk ID"),
    no_context: bool = typer.Option(False, "--no-context", help="Skip knowledge base search"),
    num_sources: int = typer.Option(3, "--sources", "-s", help="Number of sources for context"),
    artistic: bool = typer.Option(False, "--artistic", "-a", help="Generate artistic imagery instead of diagrams"),
):
    """Generate images using Gemini (Nano Banana).

    By default generates informational diagrams. Use --artistic for creative/evocative imagery.
    """
    try:
        if chunk_id:
            console.print(f"[yellow]Generating image for chunk {chunk_id}...[/yellow]\n")
            result = viz.generate_from_chunk(chunk_id, output_dir=output_dir)
        else:
            mode = "[cyan]artistic[/cyan]" if artistic else "[cyan]diagram[/cyan]"
            console.print(f"[yellow]Generating {mode} image for:[/yellow] {concept}\n")
            if not no_context:
                console.print("[dim]Searching knowledge base for context...[/dim]")
            result = viz.generate_image(
                concept,
                output_dir=output_dir,
                use_context=not no_context,
                num_chunks=num_sources,
                artistic=artistic,
            )

        if result["status"] == "success":
            console.print(f"[green]Generated:[/green] {result['image_path']}")
            if result.get("text_response"):
                console.print(f"\n[dim]Model notes:[/dim] {result['text_response'][:200]}...")
            # Open the image
            import subprocess
            subprocess.run(["open", result["image_path"]], check=False)
        else:
            console.print(f"[red]Failed:[/red] {result['status']}")
            if result.get("text_response"):
                console.print(f"[dim]Response:[/dim] {result['text_response']}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise


@app.command()
def extract(
    topic: str = typer.Argument(None, help="Topic to search for artifacts"),
    artifact_type: str = typer.Option("all", "--type", "-t", help="Artifact type: prompt, pattern, checklist, rule, tool, all"),
    sources: int = typer.Option(10, "--sources", "-s", help="Number of source chunks to search"),
    item_id: int = typer.Option(None, "--item", "-i", help="Extract from specific item ID"),
    output: str = typer.Option(None, "--output", "-o", help="Save to file (JSON)"),
):
    """Extract implementation artifacts from your knowledge base.

    Examples:
        dz extract "agentic engineering"
        dz extract "prompt design" --type prompt
        dz extract --item 3 --type checklist
    """
    try:
        if not topic and not item_id:
            console.print("[red]Error:[/red] Provide a topic or --item ID")
            return

        if item_id:
            console.print(f"[yellow]Extracting from item {item_id}...[/yellow]\n")
            result = ext.extract_from_item(item_id, artifact_type=artifact_type)
        else:
            console.print(f"[yellow]Extracting artifacts for:[/yellow] {topic}\n")
            result = ext.extract_artifacts(topic, artifact_type=artifact_type, num_sources=sources)

        if result["status"] != "success":
            console.print(f"[red]{result.get('message', 'Extraction failed')}[/red]")
            return

        # Display artifacts
        artifacts = result.get("artifacts", [])
        if not artifacts:
            console.print("[dim]No artifacts found[/dim]")
            return

        for i, artifact in enumerate(artifacts, 1):
            atype = artifact.get("type", "unknown").upper()
            name = artifact.get("name", "Unnamed")
            content = artifact.get("content", "")
            context = artifact.get("context", "")

            # Create panel for each artifact
            panel_content = f"{content}\n\n[dim]Context: {context}[/dim]"
            console.print(Panel(
                panel_content,
                title=f"[bold]{atype}[/bold] {name}",
                border_style="cyan" if atype != "RAW" else "yellow",
            ))
            console.print()

        # Show sources
        if result.get("sources"):
            console.print("[dim]Sources used:[/dim]")
            for src in result["sources"][:5]:
                title = src.get("title", "?")[:40]
                console.print(f"  - {title}")

        # Show notes
        if result.get("notes"):
            console.print(f"\n[dim]Notes: {result['notes']}[/dim]")

        console.print(f"\n[dim]Artifacts: {len(artifacts)} | Tokens: {result.get('tokens_used', '?')}[/dim]")

        # Save to file if requested
        if output:
            import json
            output_path = Path(output)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            console.print(f"[green]Saved to:[/green] {output_path}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise


# --- Artifacts subcommands ---

@artifacts_app.command("list")
def artifacts_list():
    """List all stored artifacts."""
    try:
        artifacts = art.list_all_artifacts()

        if not artifacts:
            console.print("[dim]No artifacts found. Extract some first:[/dim]")
            console.print("  dz extract \"topic\" -o artifacts/topic.json")
            return

        # Group by source file
        by_file = {}
        for a in artifacts:
            source = a.get("_source_file", "unknown")
            if source not in by_file:
                by_file[source] = []
            by_file[source].append(a)

        for source, file_artifacts in by_file.items():
            console.print(f"\n[bold cyan]{source}.json[/bold cyan]")
            table = Table(show_header=True, box=None)
            table.add_column("#", style="dim", width=3)
            table.add_column("Type", style="yellow", width=10)
            table.add_column("Name", style="green")

            for i, a in enumerate(file_artifacts, 1):
                table.add_row(
                    str(i),
                    a.get("type", "?")[:10],
                    a.get("name", "Unnamed")[:50],
                )
            console.print(table)

        console.print(f"\n[dim]Total: {len(artifacts)} artifacts[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@artifacts_app.command("show")
def artifacts_show(name: str):
    """Show a specific artifact by name."""
    try:
        artifact = art.find_artifact(name)

        if not artifact:
            console.print(f"[red]Artifact not found:[/red] {name}")
            console.print("[dim]Use 'dz artifacts list' to see available artifacts[/dim]")
            return

        atype = artifact.get("type", "unknown").upper()
        aname = artifact.get("name", "Unnamed")
        content = artifact.get("content", "")
        context = artifact.get("context", "")
        source = artifact.get("_source_file", "")

        console.print(Panel(
            f"{content}\n\n[dim]Context: {context}[/dim]",
            title=f"[bold]{atype}[/bold] {aname}",
            subtitle=f"[dim]from {source}.json[/dim]" if source else None,
            border_style="cyan",
        ))

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@artifacts_app.command("apply")
def artifacts_apply(
    name: str,
    context: str = typer.Option(..., "--context", "-c", help="Your project/situation context"),
):
    """Apply an artifact to your specific context.

    Example:
        dz artifacts apply "Agentic Layer Pattern" -c "I have a FastAPI backend"
    """
    try:
        artifact = art.find_artifact(name)

        if not artifact:
            console.print(f"[red]Artifact not found:[/red] {name}")
            return

        console.print(f"[yellow]Applying:[/yellow] {artifact.get('name')}")
        console.print(f"[dim]To context:[/dim] {context}\n")

        result = art.apply_artifact(artifact, context)

        console.print(Panel(
            Markdown(result["applied_guidance"]),
            title=f"Applied: {result['artifact_name']}",
            border_style="green",
        ))

        console.print(f"\n[dim]Tokens: {result['tokens_used']}[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise


@artifacts_app.command("search")
def artifacts_search(query: str):
    """Search artifacts by keyword."""
    try:
        results = art.search_artifacts(query)

        if not results:
            console.print(f"[dim]No artifacts matching:[/dim] {query}")
            return

        console.print(f"[green]Found {len(results)} artifacts:[/green]\n")

        for artifact in results:
            atype = artifact.get("type", "?").upper()
            aname = artifact.get("name", "Unnamed")
            source = artifact.get("_source_file", "")
            console.print(f"  [{atype}] {aname} [dim]({source})[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@artifacts_app.command("scaffold")
def artifacts_scaffold(
    name: str,
    project_name: str = typer.Option(None, "--name", "-n", help="Project name (defaults to artifact name)"),
    output_dir: str = typer.Option(None, "--output", "-o", help="Output directory (default: ~/projects)"),
):
    """Generate a working test project from an artifact.

    Creates a runnable demo that implements the technique.

    Example:
        dz artifacts scaffold "Core Four" -n core-four-demo
    """
    try:
        artifact = art.find_artifact(name)

        if not artifact:
            console.print(f"[red]Artifact not found:[/red] {name}")
            return

        # Default project name from artifact
        if not project_name:
            project_name = artifact.get("name", "demo").lower().replace(" ", "-")[:30]

        # Default output directory
        if not output_dir:
            output_dir = Path.home() / "projects"

        console.print(f"[yellow]Scaffolding:[/yellow] {artifact.get('name')}")
        console.print(f"[dim]Project:[/dim] {project_name}\n")

        result = art.scaffold_project(
            artifact,
            project_name,
            Path(output_dir),
        )

        if result["status"] != "success":
            console.print(f"[red]Failed:[/red] {result.get('message', 'Unknown error')}")
            if result.get("raw_response"):
                console.print(f"[dim]{result['raw_response'][:500]}...[/dim]")
            return

        console.print(f"[green]Created:[/green] {result['project_dir']}\n")

        console.print("[cyan]Files:[/cyan]")
        for f in result["files_created"]:
            console.print(f"  {f}")

        if result.get("run_command"):
            console.print(f"\n[yellow]To run:[/yellow]")
            console.print(f"  cd {result['project_dir']}")
            console.print(f"  {result['run_command']}")

        if result.get("description"):
            console.print(f"\n[dim]{result['description']}[/dim]")

        console.print(f"\n[dim]Tokens: {result.get('tokens_used', '?')}[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise


@app.command()
def demo(
    topic: str,
    project_name: str = typer.Option(None, "--name", "-n", help="Project name"),
    output_dir: str = typer.Option(None, "--output", "-o", help="Output directory"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Build a hello world demo from a lesson topic.

    Searches your knowledge base, identifies the core lesson,
    proposes a minimal demo, and builds it after confirmation.

    Example:
        dz demo "agentic layer"
        dz demo "prompt engineering" -n prompt-demo -o ~/projects/
    """
    try:
        console.print(f"[yellow]Analyzing:[/yellow] {topic}\n")

        # Step 1: Analyze and propose
        analysis = art.analyze_for_demo(topic)

        if analysis["status"] != "success":
            console.print(f"[red]{analysis.get('message', 'Analysis failed')}[/red]")
            return

        # Show what we found
        console.print(f"[cyan]Found {len(analysis['sources'])} sources:[/cyan]")
        for src in analysis["sources"][:3]:
            console.print(f"  - {src['title']}")

        console.print(f"\n[green]Core lesson:[/green]")
        console.print(f"  {analysis['core_lesson']}")

        console.print(f"\n[green]Proposed hello world:[/green]")
        console.print(f"  {analysis['demo_concept']}")

        if analysis.get("proves"):
            console.print(f"\n[dim]This proves: {analysis['proves']}[/dim]")

        # Step 2: Confirm
        if not yes:
            console.print()
            proceed = typer.confirm("Build this demo?")
            if not proceed:
                console.print("[yellow]Cancelled[/yellow]")
                return

        # Step 3: Build
        if not project_name:
            project_name = topic.lower().replace(" ", "-")[:20] + "-demo"

        if not output_dir:
            output_dir = Path.home() / "projects"

        output_path = Path(output_dir)
        # Directory created lazily by build_demo only when output is produced

        console.print(f"\n[yellow]Building demo...[/yellow]")

        result = art.build_demo(
            topic=topic,
            core_lesson=analysis["core_lesson"],
            demo_concept=analysis["demo_concept"],
            project_name=project_name,
            output_dir=output_path,
        )

        if result["status"] != "success":
            console.print(f"[red]Failed:[/red] {result.get('message', 'Unknown error')}")
            return

        console.print(f"\n[green]Created:[/green] {result['project_dir']}")

        console.print(f"\n[cyan]Files:[/cyan]")
        for f in result["files_created"]:
            console.print(f"  {f}")

        if result.get("run_command"):
            console.print(f"\n[yellow]To run:[/yellow]")
            console.print(f"  cd {result['project_dir']}")
            console.print(f"  {result['run_command']}")

        total_tokens = analysis.get("tokens_used", 0) + result.get("tokens_used", 0)
        console.print(f"\n[dim]Tokens: {total_tokens}[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise


# --- Skills subcommands ---

@skills_app.command("list")
def skills_list():
    """List all skills."""
    try:
        skills = db.list_skills()

        if not skills:
            console.print("[dim]No skills found.[/dim]")
            console.print("[dim]Create one with:[/dim] dz skills create")
            return

        table = Table(show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Description", style="dim")
        table.add_column("Created", style="dim")

        for skill in skills:
            desc = skill.get("description") or ""
            if len(desc) > 40:
                desc = desc[:40] + "..."
            created = skill.get("created_at")
            created_str = created.strftime("%Y-%m-%d") if created else ""
            table.add_row(
                skill["name"],
                skill["type"],
                desc,
                created_str,
            )

        console.print(table)
        console.print(f"\n[dim]Total: {len(skills)} skills[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@skills_app.command("show")
def skills_show(name: str):
    """Show a skill's content."""
    try:
        skill = db.get_skill(name)

        if not skill:
            console.print(f"[red]Skill not found:[/red] {name}")
            return

        console.print(Panel(
            skill["content"],
            title=f"[bold]{skill['name']}[/bold] ({skill['type']})",
            subtitle=skill.get("description"),
            border_style="cyan",
        ))

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@skills_app.command("create")
def skills_create(
    name: str = typer.Option(..., "--name", "-n", help="Skill name"),
    skill_type: str = typer.Option(..., "--type", "-t", help="Skill type (diagnostic, tutorial, etc.)"),
    description: str = typer.Option(None, "--description", "-d", help="Skill description"),
    content_file: str = typer.Option(None, "--file", "-f", help="Read content from file"),
):
    """Create a new skill.

    Example:
        dz skills create -n "diagnostic-troubleshooting" -t "diagnostic" -d "Outputs diagnostic frameworks as PDF" -f skill.md
    """
    try:
        # Get content
        if content_file:
            content = Path(content_file).read_text()
        else:
            console.print("[yellow]Enter skill content (Ctrl+D when done):[/yellow]")
            import sys
            content = sys.stdin.read()

        if not content.strip():
            console.print("[red]Error:[/red] No content provided")
            return

        skill_id = db.create_skill(
            name=name,
            type=skill_type,
            content=content,
            description=description,
        )

        console.print(f"[green]Created skill:[/green] {name} (id: {skill_id})")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@skills_app.command("update")
def skills_update(
    name: str,
    content_file: str = typer.Option(None, "--file", "-f", help="Read new content from file"),
    description: str = typer.Option(None, "--description", "-d", help="Update description"),
):
    """Update a skill's content."""
    try:
        skill = db.get_skill(name)
        if not skill:
            console.print(f"[red]Skill not found:[/red] {name}")
            return

        if content_file:
            content = Path(content_file).read_text()
        else:
            console.print("[yellow]Enter new content (Ctrl+D when done):[/yellow]")
            import sys
            content = sys.stdin.read()

        if db.update_skill(name, content, description):
            console.print(f"[green]Updated:[/green] {name}")
        else:
            console.print(f"[red]Failed to update:[/red] {name}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@skills_app.command("delete")
def skills_delete(name: str):
    """Delete a skill."""
    try:
        skill = db.get_skill(name)
        if not skill:
            console.print(f"[red]Skill not found:[/red] {name}")
            return

        if typer.confirm(f"Delete skill '{name}'?"):
            if db.delete_skill(name):
                console.print(f"[green]Deleted:[/green] {name}")
            else:
                console.print(f"[red]Failed to delete:[/red] {name}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


# --- Projects subcommands ---

def _parse_facet(value: str | None) -> list[str]:
    """Parse comma-separated facet values into a list."""
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


@projects_app.command("create")
def projects_create(
    name: str = typer.Argument(..., help="Project name (unique identifier)"),
    about: str = typer.Option(None, "--about", "-a", help="Comma-separated 'about' facets (what it's about)"),
    uses: str = typer.Option(None, "--uses", "-u", help="Comma-separated 'uses' facets (technologies used)"),
    needs: str = typer.Option(None, "--needs", "-n", help="Comma-separated 'needs' facets (requirements)"),
    description: str = typer.Option(None, "--description", "-d", help="Project description"),
    status: str = typer.Option("active", "--status", "-s", help="Project status (active, completed, archived)"),
):
    """Create a new project with faceted organization.

    Example:
        dz project create power-quality --about "power quality,flicker" --uses "python,signal processing" --needs "data collection"
    """
    try:
        # Check if project already exists
        existing = db.get_project(name)
        if existing:
            console.print(f"[red]Project already exists:[/red] {name}")
            return

        project_id = db.create_project(
            name=name,
            description=description,
            status=status,
            facet_about=_parse_facet(about),
            facet_uses=_parse_facet(uses),
            facet_needs=_parse_facet(needs),
        )

        console.print(f"[green]Created project:[/green] {name} (id: {project_id})")

        # Show facets if any were provided
        if about:
            console.print(f"  [dim]About:[/dim] {about}")
        if uses:
            console.print(f"  [dim]Uses:[/dim] {uses}")
        if needs:
            console.print(f"  [dim]Needs:[/dim] {needs}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@projects_app.command("list")
def projects_list(
    status: str = typer.Option(None, "--status", "-s", help="Filter by status (active, completed, archived)"),
):
    """List all projects."""
    try:
        projects = db.list_projects(status=status)

        if not projects:
            if status:
                console.print(f"[dim]No {status} projects found.[/dim]")
            else:
                console.print("[dim]No projects found.[/dim]")
            console.print("[dim]Create one with:[/dim] dz project create <name>")
            return

        table = Table(show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Status", style="yellow")
        table.add_column("About", style="dim")
        table.add_column("Uses", style="dim")
        table.add_column("Needs", style="dim")

        for proj in projects:
            about_str = ", ".join(proj.get("facet_about", [])[:3])
            if len(proj.get("facet_about", [])) > 3:
                about_str += "..."
            uses_str = ", ".join(proj.get("facet_uses", [])[:3])
            if len(proj.get("facet_uses", [])) > 3:
                uses_str += "..."
            needs_str = ", ".join(proj.get("facet_needs", [])[:3])
            if len(proj.get("facet_needs", [])) > 3:
                needs_str += "..."

            table.add_row(
                proj["name"],
                proj["status"],
                about_str,
                uses_str,
                needs_str,
            )

        console.print(table)
        console.print(f"\n[dim]Total: {len(projects)} projects[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@projects_app.command("show")
def projects_show(name: str):
    """Show project details including linked items and skills."""
    try:
        project = db.get_project(name)

        if not project:
            console.print(f"[red]Project not found:[/red] {name}")
            return

        # Build project info panel
        info_lines = []
        if project.get("description"):
            info_lines.append(f"[bold]Description:[/bold] {project['description']}")
        info_lines.append(f"[bold]Status:[/bold] {project['status']}")

        if project.get("facet_about"):
            info_lines.append(f"\n[bold]About:[/bold] {', '.join(project['facet_about'])}")
        if project.get("facet_uses"):
            info_lines.append(f"[bold]Uses:[/bold] {', '.join(project['facet_uses'])}")
        if project.get("facet_needs"):
            info_lines.append(f"[bold]Needs:[/bold] {', '.join(project['facet_needs'])}")

        if project.get("beads_epic_id"):
            info_lines.append(f"\n[bold]Beads Epic:[/bold] {project['beads_epic_id']}")

        created = project.get("created_at")
        updated = project.get("updated_at")
        if created:
            info_lines.append(f"\n[dim]Created: {created.strftime('%Y-%m-%d %H:%M')}[/dim]")
        if updated:
            info_lines.append(f"[dim]Updated: {updated.strftime('%Y-%m-%d %H:%M')}[/dim]")

        console.print(Panel(
            "\n".join(info_lines),
            title=f"[bold]{project['name']}[/bold]",
            border_style="cyan",
        ))

        # Show linked items
        items = db.get_project_items(project["id"])
        if items:
            console.print(f"\n[bold]Linked Items ({len(items)}):[/bold]")
            for item in items:
                console.print(f"  [{item['id']}] {item['title']} ({item['type']})")

        # Show linked skills
        skills = db.get_project_skills(project["id"])
        if skills:
            console.print(f"\n[bold]Linked Skills ({len(skills)}):[/bold]")
            for skill in skills:
                console.print(f"  {skill['name']} ({skill['type']})")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@projects_app.command("link")
def projects_link(
    name: str = typer.Argument(..., help="Project name"),
    item: int = typer.Option(None, "--item", "-i", help="Item ID to link"),
    skill: str = typer.Option(None, "--skill", "-s", help="Skill name to link"),
):
    """Link an item or skill to a project.

    Example:
        dz project link my-project --item 5
        dz project link my-project --skill "diagnostic-troubleshooting"
    """
    try:
        project = db.get_project(name)
        if not project:
            console.print(f"[red]Project not found:[/red] {name}")
            return

        if not item and not skill:
            console.print("[red]Provide --item or --skill to link[/red]")
            return

        if item:
            if db.link_project_item(project["id"], item):
                console.print(f"[green]Linked item {item} to project {name}[/green]")
            else:
                console.print(f"[red]Failed to link item {item}[/red]")

        if skill:
            skill_obj = db.get_skill(skill)
            if not skill_obj:
                console.print(f"[red]Skill not found:[/red] {skill}")
                return
            if db.link_project_skill(project["id"], skill_obj["id"]):
                console.print(f"[green]Linked skill '{skill}' to project {name}[/green]")
            else:
                console.print(f"[red]Failed to link skill '{skill}'[/red]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@projects_app.command("unlink")
def projects_unlink(
    name: str = typer.Argument(..., help="Project name"),
    item: int = typer.Option(None, "--item", "-i", help="Item ID to unlink"),
    skill: str = typer.Option(None, "--skill", "-s", help="Skill name to unlink"),
):
    """Unlink an item or skill from a project.

    Example:
        dz project unlink my-project --item 5
        dz project unlink my-project --skill "diagnostic-troubleshooting"
    """
    try:
        project = db.get_project(name)
        if not project:
            console.print(f"[red]Project not found:[/red] {name}")
            return

        if not item and not skill:
            console.print("[red]Provide --item or --skill to unlink[/red]")
            return

        if item:
            if db.unlink_project_item(project["id"], item):
                console.print(f"[green]Unlinked item {item} from project {name}[/green]")
            else:
                console.print(f"[yellow]Item {item} was not linked to project {name}[/yellow]")

        if skill:
            skill_obj = db.get_skill(skill)
            if not skill_obj:
                console.print(f"[red]Skill not found:[/red] {skill}")
                return
            if db.unlink_project_skill(project["id"], skill_obj["id"]):
                console.print(f"[green]Unlinked skill '{skill}' from project {name}[/green]")
            else:
                console.print(f"[yellow]Skill '{skill}' was not linked to project {name}[/yellow]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@projects_app.command("update")
def projects_update(
    name: str = typer.Argument(..., help="Project name"),
    about: str = typer.Option(None, "--about", "-a", help="Replace 'about' facets (comma-separated)"),
    uses: str = typer.Option(None, "--uses", "-u", help="Replace 'uses' facets (comma-separated)"),
    needs: str = typer.Option(None, "--needs", "-n", help="Replace 'needs' facets (comma-separated)"),
    description: str = typer.Option(None, "--description", "-d", help="Update description"),
    status: str = typer.Option(None, "--status", "-s", help="Update status (active, completed, archived)"),
):
    """Update a project's facets, description, or status.

    Example:
        dz project update my-project --status completed
        dz project update my-project --about "new topic,another topic"
    """
    try:
        project = db.get_project(name)
        if not project:
            console.print(f"[red]Project not found:[/red] {name}")
            return

        # Parse facets only if provided
        facet_about = _parse_facet(about) if about is not None else None
        facet_uses = _parse_facet(uses) if uses is not None else None
        facet_needs = _parse_facet(needs) if needs is not None else None

        if db.update_project(
            name=name,
            description=description,
            status=status,
            facet_about=facet_about,
            facet_uses=facet_uses,
            facet_needs=facet_needs,
        ):
            console.print(f"[green]Updated:[/green] {name}")
        else:
            console.print(f"[red]Failed to update:[/red] {name}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@projects_app.command("delete")
def projects_delete(name: str):
    """Delete a project."""
    try:
        project = db.get_project(name)
        if not project:
            console.print(f"[red]Project not found:[/red] {name}")
            return

        if typer.confirm(f"Delete project '{name}'?"):
            if db.delete_project(name):
                console.print(f"[green]Deleted:[/green] {name}")
            else:
                console.print(f"[red]Failed to delete:[/red] {name}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@projects_app.command("embed")
def projects_embed(
    name: str = typer.Argument(None, help="Project name (or 'all' for all projects)"),
):
    """Embed project facets for cross-pollination discovery.

    Example:
        dz project embed my-project
        dz project embed all
    """
    try:
        if name == "all" or name is None:
            console.print("[yellow]Embedding all projects needing embeddings...[/yellow]\n")
            results = emb.embed_all_projects()

            if not results:
                console.print("[dim]No projects need embedding[/dim]")
                return

            for r in results:
                if r.get("error"):
                    console.print(f"[red]{r['name']}:[/red] {r['error']}")
                else:
                    facets = [k for k, v in r["embedded"].items() if v]
                    console.print(f"[green]{r['name']}:[/green] embedded {', '.join(facets)}")

            console.print(f"\n[dim]Embedded {len(results)} projects[/dim]")
        else:
            project = db.get_project(name)
            if not project:
                console.print(f"[red]Project not found:[/red] {name}")
                return

            console.print(f"[yellow]Embedding project:[/yellow] {name}\n")
            result = emb.embed_project_facets(project["id"])

            facets = [k for k, v in result.items() if v]
            if facets:
                console.print(f"[green]Embedded:[/green] {', '.join(facets)}")
            else:
                console.print("[dim]No facets to embed[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@projects_app.command("discover")
def projects_discover(
    name: str = typer.Argument(..., help="Project name"),
    facet: str = typer.Option("needs", "--facet", "-f", help="Facet to match: about, uses, needs"),
    limit: int = typer.Option(10, "--limit", "-l", help="Max results"),
    threshold: float = typer.Option(0.3, "--threshold", "-t", help="Min similarity (0-1)"),
):
    """Discover knowledge that matches a project's facets.

    Cross-pollination: Find content that could help your project.
    Default searches for content matching what the project NEEDS.

    Example:
        dz project discover my-project
        dz project discover my-project --facet uses
        dz project discover my-project --threshold 0.4
    """
    try:
        project = db.get_project(name)
        if not project:
            console.print(f"[red]Project not found:[/red] {name}")
            return

        # Check if project has embeddings
        facet_list = project.get(f"facet_{facet}", [])
        if not facet_list:
            console.print(f"[yellow]Project has no '{facet}' facets defined[/yellow]")
            return

        console.print(f"[yellow]Discovering for:[/yellow] {name}")
        console.print(f"[dim]Matching {facet}: {', '.join(facet_list)}[/dim]\n")

        # Ensure project is embedded
        results = db.discover_for_project(
            project["id"],
            facet=facet,
            limit=limit,
            min_similarity=threshold,
        )

        if not results:
            # Maybe project isn't embedded yet
            console.print("[dim]No matches found. Try embedding the project first:[/dim]")
            console.print(f"  dz project embed {name}")
            return

        # Display results
        table = Table(show_header=True)
        table.add_column("Sim", style="yellow", width=5)
        table.add_column("Source", style="cyan")
        table.add_column("Content Preview", style="dim")

        for r in results:
            sim = f"{r['similarity']:.2f}"
            title = r["item_title"][:30] if r["item_title"] else "?"
            if r.get("timestamp_start"):
                ts = int(r["timestamp_start"])
                title += f" @{ts // 60}:{ts % 60:02d}"
            preview = r["content"][:60].replace("\n", " ")
            if len(r["content"]) > 60:
                preview += "..."

            table.add_row(sim, title, preview)

        console.print(table)
        console.print(f"\n[dim]Found {len(results)} matches[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise


@app.command("crosspolinate")
def crosspolinate(
    query_text: str = typer.Argument(..., help="Text to find matching projects for"),
    facet: str = typer.Option("needs", "--facet", "-f", help="Facet to match: about, uses, needs"),
    limit: int = typer.Option(5, "--limit", "-l", help="Max results"),
):
    """Find projects that could benefit from given content/topic.

    Reverse cross-pollination: Given a topic or chunk of content,
    find which active projects might need it.

    Example:
        dz crosspolinate "Python decorators for caching"
        dz crosspolinate "vector embeddings" --facet uses
    """
    try:
        console.print(f"[yellow]Finding projects that {facet}:[/yellow] {query_text}\n")

        # Get embedding for the query
        query_embedding = emb.get_embedding(query_text)

        # Find matching projects
        projects = db.discover_projects_for_content(
            query_embedding,
            facet=facet,
            limit=limit,
        )

        if not projects:
            console.print("[dim]No matching projects found[/dim]")
            console.print("[dim]Ensure projects are embedded:[/dim] dz project embed all")
            return

        table = Table(show_header=True)
        table.add_column("Sim", style="yellow", width=5)
        table.add_column("Project", style="cyan")
        table.add_column(facet.capitalize(), style="dim")

        for p in projects:
            sim = f"{p['similarity']:.2f}"
            facet_values = ", ".join(p.get(f"facet_{facet}", [])[:3])
            table.add_row(sim, p["name"], facet_values)

        console.print(table)
        console.print(f"\n[dim]Found {len(projects)} projects[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise


# --- Sources subcommands ---

@sources_app.command("list")
def sources_list():
    """List all sources (channels, repos)."""
    try:
        sources = db.list_sources()

        if not sources:
            console.print("[dim]No sources found.[/dim]")
            console.print("[dim]Harvest content to create sources:[/dim] dz harvest <url>")
            return

        table = Table(show_header=True)
        table.add_column("ID", style="dim", width=4)
        table.add_column("Type", style="yellow", width=10)
        table.add_column("Name", style="cyan")
        table.add_column("URL", style="dim")

        for source in sources:
            url = source.get("url") or ""
            if len(url) > 50:
                url = url[:47] + "..."
            table.add_row(
                str(source["id"]),
                source["type"],
                source["name"],
                url,
            )

        console.print(table)
        console.print(f"\n[dim]Total: {len(sources)} sources[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@sources_app.command("delete")
def sources_delete(
    source_id: int = typer.Argument(..., help="Source ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete a source by ID.

    Note: This only deletes the source record. Associated items and chunks
    remain in the database (they will show as having no source).

    Example:
        dz sources delete 3
        dz sources delete 3 --force
    """
    try:
        # First check if source exists
        sources = db.list_sources()
        source = next((s for s in sources if s["id"] == source_id), None)

        if not source:
            console.print(f"[red]Source not found:[/red] ID {source_id}")
            console.print("[dim]Use 'dz sources list' to see available sources[/dim]")
            return

        # Show source details
        console.print(f"[yellow]Source to delete:[/yellow]")
        console.print(f"  ID: {source['id']}")
        console.print(f"  Type: {source['type']}")
        console.print(f"  Name: {source['name']}")
        if source.get("url"):
            console.print(f"  URL: {source['url']}")

        if not force:
            if not typer.confirm("\nDelete this source?"):
                console.print("[yellow]Cancelled[/yellow]")
                return

        if db.delete_source(source_id):
            console.print(f"[green]Deleted:[/green] {source['name']}")
        else:
            console.print(f"[red]Failed to delete source[/red]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


# --- Items subcommands ---

@items_app.command("list")
def items_list(
    source_id: int = typer.Option(None, "--source", "-s", help="Filter by source ID"),
):
    """List all items (videos, articles, code files)."""
    try:
        items = db.list_items(source_id=source_id)

        if not items:
            if source_id:
                console.print(f"[dim]No items found for source ID {source_id}.[/dim]")
            else:
                console.print("[dim]No items found.[/dim]")
            console.print("[dim]Harvest content to create items:[/dim] dz harvest <url>")
            return

        table = Table(show_header=True)
        table.add_column("ID", style="dim", width=4)
        table.add_column("Type", style="yellow", width=10)
        table.add_column("Title", style="cyan")
        table.add_column("Source", style="dim", width=8)

        for item in items:
            title = item.get("title") or ""
            if len(title) > 50:
                title = title[:47] + "..."
            source = str(item.get("source_id") or "-")
            table.add_row(
                str(item["id"]),
                item["type"],
                title,
                source,
            )

        console.print(table)
        console.print(f"\n[dim]Total: {len(items)} items[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@items_app.command("delete")
def items_delete(
    item_id: int = typer.Argument(..., help="Item ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete an item by ID.

    This will also delete all associated chunks (embeddings) for this item.

    Example:
        dz items delete 5
        dz items delete 5 --force
    """
    try:
        # First check if item exists
        items = db.list_items()
        item = next((i for i in items if i["id"] == item_id), None)

        if not item:
            console.print(f"[red]Item not found:[/red] ID {item_id}")
            console.print("[dim]Use 'dz items list' to see available items[/dim]")
            return

        # Show item details
        console.print(f"[yellow]Item to delete:[/yellow]")
        console.print(f"  ID: {item['id']}")
        console.print(f"  Type: {item['type']}")
        console.print(f"  Title: {item['title']}")
        if item.get("url"):
            console.print(f"  URL: {item['url']}")
        if item.get("source_id"):
            console.print(f"  Source ID: {item['source_id']}")

        console.print("\n[yellow]Warning:[/yellow] This will also delete all chunks for this item.")

        if not force:
            if not typer.confirm("\nDelete this item?"):
                console.print("[yellow]Cancelled[/yellow]")
                return

        if db.delete_item(item_id):
            console.print(f"[green]Deleted:[/green] {item['title']}")

            # Regenerate index after deletion
            console.print("[yellow]Updating index...[/yellow]")
            _regenerate_index()
            console.print("[green]Index updated[/green]")
        else:
            console.print(f"[red]Failed to delete item[/red]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


# --- Suggest subcommands ---

def _format_views(view_count: int | None) -> str:
    """Format view count for display."""
    if view_count is None:
        return "?"
    if view_count >= 1_000_000:
        return f"{view_count / 1_000_000:.1f}M"
    if view_count >= 1_000:
        return f"{view_count / 1_000:.1f}K"
    return str(view_count)


def _display_project_facets(project: dict) -> None:
    """Display project facets as context for suggestions."""
    needs = project.get("facet_needs", [])
    uses = project.get("facet_uses", [])
    about = project.get("facet_about", [])

    facet_parts = []
    if needs:
        # Escape brackets for Rich markup (use \[ for literal [)
        facet_parts.append(f"NEEDS=\\[{', '.join(needs)}]")
    if uses:
        facet_parts.append(f"USES=\\[{', '.join(uses)}]")
    if about:
        facet_parts.append(f"ABOUT=\\[{', '.join(about)}]")

    if facet_parts:
        console.print(f"[dim]Based on: {', '.join(facet_parts)}[/dim]\n")
    else:
        console.print("[yellow]Warning: Project has no facets defined.[/yellow]")
        console.print("[dim]Add facets with: dz project update <name> --needs 'topic1,topic2'[/dim]\n")


def _suggest_youtube(
    project: dict,
    min_score: int = 40,
    limit: int = 10,
    max_queries: int = 3,
) -> list[dict]:
    """Search YouTube and score videos based on project facets.

    Args:
        project: Project dictionary with facets
        min_score: Minimum quality score (0-100)
        limit: Maximum results to return
        max_queries: Maximum number of search queries to run

    Returns:
        List of scored video dictionaries
    """
    # Generate search queries from project facets
    query_set = search_queries.generate_from_project(project)
    youtube_queries = query_set.youtube_queries()[:max_queries]

    if not youtube_queries:
        return []

    all_videos = []
    seen_ids = set()

    for query in youtube_queries:
        try:
            # Search YouTube (limit per query to avoid too many results)
            videos = harv.search_youtube(query, limit=min(limit, 10))

            for video in videos:
                video_id = video.get("id")
                if video_id and video_id not in seen_ids:
                    seen_ids.add(video_id)

                    # Get more detailed info for scoring
                    try:
                        info = harv.get_video_info(video["url"])
                        view_count = info.get("view_count")
                        like_count = info.get("like_count")
                        duration = info.get("duration")
                        description = info.get("description", "")
                    except Exception:
                        # Fall back to basic info from search
                        view_count = None
                        like_count = None
                        duration = video.get("duration")
                        description = ""

                    # Score the video
                    breakdown = scoring.score_video(
                        title=video.get("title", ""),
                        view_count=view_count,
                        like_count=like_count,
                        duration=duration,
                        description=description,
                    )

                    if breakdown.total >= min_score:
                        video["score"] = breakdown.total
                        video["score_breakdown"] = breakdown.to_dict()
                        video["view_count"] = view_count
                        video["duration"] = duration
                        video["search_query"] = query
                        all_videos.append(video)

        except Exception as e:
            console.print(f"[yellow]Warning: Search failed for '{query}': {e}[/yellow]")

    # Sort by score and limit
    all_videos.sort(key=lambda v: v.get("score", 0), reverse=True)
    return all_videos[:limit]


def _suggest_github(
    project: dict,
    min_score: int = 40,
    limit: int = 10,
    max_queries: int = 3,
) -> list[dict]:
    """Search GitHub and score repos based on project facets.

    Note: This is a placeholder that returns empty results since
    GitHub search is not yet implemented in the harvest module.

    Args:
        project: Project dictionary with facets
        min_score: Minimum quality score (0-100)
        limit: Maximum results to return
        max_queries: Maximum number of search queries to run

    Returns:
        List of scored repository dictionaries (currently empty)
    """
    # Generate search queries from project facets
    query_set = search_queries.generate_from_project(project)
    github_queries = query_set.github_queries()[:max_queries]

    if not github_queries:
        return []

    # Note: GitHub search is not yet implemented in harvest.py
    # This is a placeholder that could be extended with GitHub API integration
    console.print("[dim]GitHub search is not yet implemented.[/dim]")
    console.print("[dim]Generated queries that would be used:[/dim]")
    for i, query in enumerate(github_queries[:5], 1):
        console.print(f"  {i}. {query}")

    return []


@suggest_app.command("youtube")
def suggest_youtube(
    project_name: str = typer.Argument(..., help="Project name to get suggestions for"),
    min_score: int = typer.Option(40, "--min-score", "-m", help="Minimum quality score (0-100)"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results to show"),
    max_queries: int = typer.Option(3, "--queries", "-q", help="Maximum search queries to run"),
):
    """Suggest YouTube videos based on project facets.

    Searches YouTube using queries generated from the project's NEEDS, USES,
    and ABOUT facets, then scores and ranks the results by educational quality.

    Example:
        dz suggest youtube my-project
        dz suggest youtube my-project --min-score 60 --limit 5
    """
    try:
        # Load project
        project = db.get_project(project_name)
        if not project:
            console.print(f"[red]Project not found:[/red] {project_name}")
            console.print("[dim]Use 'dz project list' to see available projects[/dim]")
            return

        console.print(f"[yellow]Suggestions for project:[/yellow] {project_name}\n")
        _display_project_facets(project)

        console.print("[dim]Searching YouTube...[/dim]\n")

        videos = _suggest_youtube(
            project,
            min_score=min_score,
            limit=limit,
            max_queries=max_queries,
        )

        if not videos:
            console.print("[dim]No videos found matching your criteria.[/dim]")
            console.print("[dim]Try lowering --min-score or adding more project facets.[/dim]")
            return

        # Display results table
        console.print(f"[green]YouTube Videos ({len(videos)} results):[/green]")
        table = Table(show_header=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Title", style="cyan", no_wrap=False, max_width=50)
        table.add_column("Score", style="yellow", justify="right", width=6)
        table.add_column("Views", style="green", justify="right", width=8)
        table.add_column("Channel", style="dim", max_width=20)

        for i, video in enumerate(videos, 1):
            title = video.get("title", "?")
            if len(title) > 50:
                title = title[:47] + "..."
            channel = video.get("channel", "?")
            if len(channel) > 20:
                channel = channel[:17] + "..."

            table.add_row(
                str(i),
                title,
                str(video.get("score", "?")),
                _format_views(video.get("view_count")),
                channel,
            )

        console.print(table)
        console.print("\n[dim]To harvest a video:[/dim] dz harvest <url>")

        # Show URLs for easy copying
        console.print("\n[dim]Video URLs:[/dim]")
        for i, video in enumerate(videos[:5], 1):
            console.print(f"  {i}. {video.get('url', '?')}")
        if len(videos) > 5:
            console.print(f"  [dim]... and {len(videos) - 5} more[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise


@suggest_app.command("github")
def suggest_github(
    project_name: str = typer.Argument(..., help="Project name to get suggestions for"),
    min_score: int = typer.Option(40, "--min-score", "-m", help="Minimum quality score (0-100)"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results to show"),
    max_queries: int = typer.Option(3, "--queries", "-q", help="Maximum search queries to run"),
):
    """Suggest GitHub repositories based on project facets.

    Searches GitHub using queries generated from the project's NEEDS, USES,
    and ABOUT facets, then scores and ranks the results.

    Note: GitHub search is not yet fully implemented. This command shows
    the queries that would be used.

    Example:
        dz suggest github my-project
        dz suggest github my-project --min-score 60 --limit 5
    """
    try:
        # Load project
        project = db.get_project(project_name)
        if not project:
            console.print(f"[red]Project not found:[/red] {project_name}")
            console.print("[dim]Use 'dz project list' to see available projects[/dim]")
            return

        console.print(f"[yellow]Suggestions for project:[/yellow] {project_name}\n")
        _display_project_facets(project)

        repos = _suggest_github(
            project,
            min_score=min_score,
            limit=limit,
            max_queries=max_queries,
        )

        if not repos:
            console.print("\n[dim]No GitHub repositories to display.[/dim]")
            console.print("[dim]GitHub search integration coming soon![/dim]")
            return

        # Display results (when implemented)
        console.print(f"\n[green]GitHub Repositories ({len(repos)} results):[/green]")
        table = Table(show_header=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Repository", style="cyan", no_wrap=False)
        table.add_column("Score", style="yellow", justify="right", width=6)
        table.add_column("Stars", style="green", justify="right", width=8)
        table.add_column("Description", style="dim")

        for i, repo in enumerate(repos, 1):
            table.add_row(
                str(i),
                repo.get("name", "?"),
                str(repo.get("score", "?")),
                str(repo.get("stars", "?")),
                (repo.get("description", "?") or "")[:40],
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise


@suggest_app.command("all")
def suggest_all(
    project_name: str = typer.Argument(..., help="Project name to get suggestions for"),
    min_score: int = typer.Option(40, "--min-score", "-m", help="Minimum quality score (0-100)"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results per source"),
    max_queries: int = typer.Option(3, "--queries", "-q", help="Maximum search queries per source"),
):
    """Suggest content from all sources (YouTube and GitHub).

    Combines suggestions from YouTube videos and GitHub repositories
    based on the project's facets.

    Example:
        dz suggest all my-project
        dz suggest all my-project --min-score 50 --limit 5
    """
    try:
        # Load project
        project = db.get_project(project_name)
        if not project:
            console.print(f"[red]Project not found:[/red] {project_name}")
            console.print("[dim]Use 'dz project list' to see available projects[/dim]")
            return

        console.print(f"[yellow]Suggestions for project:[/yellow] {project_name}\n")
        _display_project_facets(project)

        # YouTube suggestions
        console.print("[dim]Searching YouTube...[/dim]\n")
        videos = _suggest_youtube(
            project,
            min_score=min_score,
            limit=limit,
            max_queries=max_queries,
        )

        if videos:
            console.print(f"[green]YouTube Videos ({len(videos)} results):[/green]")
            table = Table(show_header=True)
            table.add_column("#", style="dim", width=3)
            table.add_column("Title", style="cyan", no_wrap=False, max_width=50)
            table.add_column("Score", style="yellow", justify="right", width=6)
            table.add_column("Views", style="green", justify="right", width=8)
            table.add_column("Channel", style="dim", max_width=20)

            for i, video in enumerate(videos, 1):
                title = video.get("title", "?")
                if len(title) > 50:
                    title = title[:47] + "..."
                channel = video.get("channel", "?")
                if len(channel) > 20:
                    channel = channel[:17] + "..."

                table.add_row(
                    str(i),
                    title,
                    str(video.get("score", "?")),
                    _format_views(video.get("view_count")),
                    channel,
                )

            console.print(table)
        else:
            console.print("[dim]No YouTube videos found matching your criteria.[/dim]")

        # GitHub suggestions
        console.print()
        repos = _suggest_github(
            project,
            min_score=min_score,
            limit=limit,
            max_queries=max_queries,
        )

        if repos:
            console.print(f"\n[green]GitHub Repositories ({len(repos)} results):[/green]")
            table = Table(show_header=True)
            table.add_column("#", style="dim", width=3)
            table.add_column("Repository", style="cyan", no_wrap=False)
            table.add_column("Score", style="yellow", justify="right", width=6)
            table.add_column("Stars", style="green", justify="right", width=8)
            table.add_column("Description", style="dim")

            for i, repo in enumerate(repos, 1):
                table.add_row(
                    str(i),
                    repo.get("name", "?"),
                    str(repo.get("score", "?")),
                    str(repo.get("stars", "?")),
                    (repo.get("description", "?") or "")[:40],
                )

            console.print(table)

        # Summary
        console.print("\n[dim]To harvest content:[/dim]")
        console.print("  dz harvest <youtube_url>")
        console.print("  dz harvest <github_url>")

        if videos:
            console.print("\n[dim]Top YouTube URLs:[/dim]")
            for i, video in enumerate(videos[:3], 1):
                console.print(f"  {i}. {video.get('url', '?')}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise


if __name__ == "__main__":
    app()
