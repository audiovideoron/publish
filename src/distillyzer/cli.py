"""Distillyzer CLI - Personal learning accelerator."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from . import db, harvest as harv, transcribe, embed, query as q, visualize as viz

app = typer.Typer(help="Distillyzer - Harvest knowledge, query it, use it.")
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


if __name__ == "__main__":
    app()
