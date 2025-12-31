"""Distillyzer CLI - Personal learning accelerator."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from . import db, harvest as harv, transcribe, embed, query as q, visualize as viz, extract as ext, artifacts as art

app = typer.Typer(help="Distillyzer - Harvest knowledge, query it, use it.")
artifacts_app = typer.Typer(help="Manage and use extracted artifacts.")
app.add_typer(artifacts_app, name="artifacts")
skills_app = typer.Typer(help="Manage presentation skills.")
app.add_typer(skills_app, name="skills")
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

            # TODO: Embed code files
            console.print("[dim]Code embedding not yet implemented[/dim]")

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
                num_chunks = embed.embed_transcript_chunks(result["item_id"], timed_chunks)
                console.print(f"[green]Stored:[/green] {num_chunks} chunks")

                # Cleanup audio file
                Path(result["audio_path"]).unlink(missing_ok=True)

                # Auto-regenerate index
                console.print("[yellow]Updating index...[/yellow]")
                _regenerate_index()
                console.print("[green]Index updated[/green]")
        else:
            console.print("[red]Unknown URL type.[/red] Supported: YouTube, GitHub")
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
    """Generate HTML index file grouped by source."""
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        "<title>Distillyzer Index</title>",
        "<style>",
        "body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; line-height: 1.6; }",
        "h1 { color: #333; }",
        "h2 { color: #444; margin-top: 2em; border-bottom: 2px solid #0066cc; padding-bottom: 0.3em; }",
        "ul { list-style: none; padding: 0; }",
        "li { margin: 0.5em 0; padding: 0.5em; background: #f9f9f9; border-radius: 4px; }",
        "a { color: #0066cc; text-decoration: none; }",
        "a:hover { text-decoration: underline; }",
        "a .timestamp { font-family: monospace; color: #0066cc; margin-right: 1em; }",
        "a:hover .timestamp { text-decoration: underline; }",
        ".preview { color: #333; }",
        ".source-section { margin: 2em 0; padding: 1em; border: 1px solid #ddd; border-radius: 8px; }",
        ".source-header { font-size: 1.5em; font-weight: bold; color: #333; margin-bottom: 1em; }",
        "details { margin: 0.5em 0; }",
        "summary { cursor: pointer; font-size: 1.1em; font-weight: 600; color: #555; padding: 0.5em; background: #f0f0f0; border-radius: 4px; }",
        "summary:hover { background: #e8e8e8; }",
        "summary .chunk-count { font-size: 0.7em; font-weight: normal; color: #888; margin-left: 0.5em; }",
        "</style>",
        "</head><body>",
        "<h1>Distillyzer Knowledge Index</h1>",
    ]

    for source_name, items in grouped_items.items():
        total_chunks = sum(len(item["chunks"]) for item in items)
        html_parts.append("<div class='source-section'>")
        html_parts.append(f"<div class='source-header'>{source_name} ({len(items)} items, {total_chunks} chunks)</div>")

        for item in items:
            chunk_count = len(item["chunks"])
            html_parts.append("<details>")
            html_parts.append(f"<summary>{item['title']}<span class='chunk-count'>({chunk_count} chunks)</span></summary>")
            is_youtube = "youtube.com" in (item["url"] or "") or "youtu.be" in (item["url"] or "")
            html_parts.append("<ul>")

            for chunk in item["chunks"]:
                ts = chunk["timestamp_start"]
                ts_str = _format_timestamp(ts)
                preview = chunk["content"][:80].replace("<", "&lt;").replace(">", "&gt;")
                if len(chunk["content"]) > 80:
                    preview += "..."

                if is_youtube and item["url"]:
                    link = _make_timestamp_link(item["url"], ts)
                    html_parts.append(
                        f'<li><a href="{link}"><span class="timestamp">{ts_str}</span></a>'
                        f'<span class="preview">{preview}</span></li>'
                    )
                else:
                    link = item["url"] or "#"
                    html_parts.append(
                        f'<li><a href="{link}"><span class="timestamp">{ts_str}</span></a>'
                        f'<span class="preview">{preview}</span></li>'
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
def visualize(
    concept: str,
    output_dir: str = typer.Option("images", "--output", "-o", help="Output directory"),
    chunk_id: int = typer.Option(None, "--chunk", "-c", help="Generate from specific chunk ID"),
    no_context: bool = typer.Option(False, "--no-context", help="Skip knowledge base search"),
    num_sources: int = typer.Option(3, "--sources", "-s", help="Number of sources for context"),
):
    """Generate informational images using Gemini (Nano Banana)."""
    try:
        if chunk_id:
            console.print(f"[yellow]Generating image for chunk {chunk_id}...[/yellow]\n")
            result = viz.generate_from_chunk(chunk_id, output_dir=output_dir)
        else:
            console.print(f"[yellow]Generating image for:[/yellow] {concept}\n")
            if not no_context:
                console.print("[dim]Searching knowledge base for context...[/dim]")
            result = viz.generate_image(
                concept,
                output_dir=output_dir,
                use_context=not no_context,
                num_chunks=num_sources,
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
    output_dir: str = typer.Option("demos", "--output", "-o", help="Output directory"),
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
        output_path.mkdir(parents=True, exist_ok=True)

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


if __name__ == "__main__":
    app()
