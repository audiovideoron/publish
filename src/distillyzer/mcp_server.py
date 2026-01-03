"""
Distillyzer MCP Server

Exposes distillyzer's knowledge base to Claude Desktop.

Usage:
    # Test with inspector (use arch -arm64 on Apple Silicon Macs)
    arch -arm64 uv run mcp dev src/distillyzer/mcp_server.py

    # Install in Claude Desktop
    uv run mcp install src/distillyzer/mcp_server.py --name "Distillyzer"
"""

from mcp.server.fastmcp import FastMCP

# TODO: Replace subprocess calls with direct imports once working
# from distillyzer.db import get_session
# from distillyzer.query import semantic_search

import subprocess

mcp = FastMCP("distillyzer")


@mcp.tool()
def query(question: str, sources: int = 5) -> str:
    """
    Semantic search across your knowledge base.

    Returns relevant chunks with sources and timestamps.
    """
    result = subprocess.run(
        ["dz", "query", question, "--sources", str(sources)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout


@mcp.tool()
def stats() -> str:
    """
    Show what's in your knowledge base.

    Returns counts of items, sources, and projects.
    """
    result = subprocess.run(
        ["dz", "stats"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout


@mcp.tool()
def harvest(url: str) -> str:
    """
    Harvest a YouTube video, GitHub repo, or article.

    Downloads, transcribes (if video), chunks, and embeds the content.
    """
    result = subprocess.run(
        ["dz", "harvest", url],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout or "Harvest complete"


@mcp.tool()
def harvest_channel(channel_url: str, limit: int = 10) -> str:
    """
    List videos from a YouTube channel for selective harvesting.

    Returns video titles and URLs so you can choose which to harvest.

    Args:
        channel_url: YouTube channel URL
        limit: Maximum videos to list
    """
    result = subprocess.run(
        ["dz", "harvest-channel", channel_url, "--limit", str(limit)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout


@mcp.tool()
def embed(item_id: int = 0, all_items: bool = False) -> str:
    """
    Re-embed existing items without re-harvesting.

    Deletes old chunks and creates new embeddings using current settings.
    Useful when embedding model or chunking strategy changes.

    Args:
        item_id: Item ID to re-embed (use 0 with all_items=True for all)
        all_items: Re-embed all items
    """
    if all_items:
        cmd = ["dz", "embed", "--all"]
    elif item_id > 0:
        cmd = ["dz", "embed", str(item_id)]
    else:
        return "Error: Provide either item_id or set all_items=True"

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout


@mcp.tool()
def search_youtube(topic: str, limit: int = 10) -> str:
    """
    Search YouTube for videos on a topic.

    Returns video titles and URLs for potential harvesting.
    """
    result = subprocess.run(
        ["dz", "search", topic, "--limit", str(limit)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout


@mcp.tool()
def list_projects() -> str:
    """
    List all distillyzer projects.
    """
    result = subprocess.run(
        ["dz", "project", "list"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout


@mcp.tool()
def project_discover(name: str) -> str:
    """
    Find knowledge matching a project's facets.

    Cross-pollination: Find content that could help your project.
    """
    result = subprocess.run(
        ["dz", "project", "discover", name],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout


@mcp.tool()
def extract(topic: str) -> str:
    """
    Extract patterns, prompts, or checklists from your knowledge.
    """
    result = subprocess.run(
        ["dz", "extract", topic],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout


@mcp.tool()
def list_items() -> str:
    """
    List harvested items in the knowledge base.
    """
    result = subprocess.run(
        ["dz", "items", "list"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout


# --- Core Tools (3) ---


@mcp.tool()
def visualize(concept: str, artistic: bool = False, no_context: bool = False, sources: int = 3) -> str:
    """
    Generate an image for a concept using Gemini.

    By default generates informational diagrams. Use artistic=True for creative imagery.

    Args:
        concept: The concept to visualize
        artistic: Generate artistic imagery instead of diagrams
        no_context: Skip knowledge base search for context
        sources: Number of sources to use for context
    """
    cmd = ["dz", "visualize", concept, "--sources", str(sources)]
    if artistic:
        cmd.append("--artistic")
    if no_context:
        cmd.append("--no-context")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout or "Image generated"


@mcp.tool()
def crosspolinate(query_text: str, facet: str = "needs", limit: int = 5) -> str:
    """
    Find projects that could benefit from given content/topic.

    Reverse cross-pollination: Given a topic or chunk of content,
    find which active projects might need it.

    Args:
        query_text: Text to find matching projects for
        facet: Facet to match: about, uses, needs
        limit: Max results to return
    """
    result = subprocess.run(
        ["dz", "crosspolinate", query_text, "--facet", facet, "--limit", str(limit)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout


@mcp.tool()
def generate_index(output: str = "index.html") -> str:
    """
    Generate HTML index with timestamp links, grouped by source.

    Creates a browsable HTML file of all knowledge base content.

    Args:
        output: Output file path
    """
    result = subprocess.run(
        ["dz", "index", "--output", output],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout or f"Index generated at {output}"


# --- Artifacts Tools (5) ---


@mcp.tool()
def artifacts_list() -> str:
    """
    List all stored artifacts.

    Shows artifacts grouped by source file with type and name.
    """
    result = subprocess.run(
        ["dz", "artifacts", "list"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout


@mcp.tool()
def artifacts_show(name: str) -> str:
    """
    Show a specific artifact by name.

    Displays the artifact's content, context, and source.

    Args:
        name: Name of the artifact to show
    """
    result = subprocess.run(
        ["dz", "artifacts", "show", name],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout


@mcp.tool()
def artifacts_apply(name: str, context: str) -> str:
    """
    Apply an artifact to your specific context.

    Takes an artifact and adapts it to your project/situation.

    Args:
        name: Name of the artifact to apply
        context: Your project/situation context (e.g., "I have a FastAPI backend")
    """
    result = subprocess.run(
        ["dz", "artifacts", "apply", name, "--context", context],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout


@mcp.tool()
def artifacts_search(query: str) -> str:
    """
    Search artifacts by keyword.

    Args:
        query: Search term to find matching artifacts
    """
    result = subprocess.run(
        ["dz", "artifacts", "search", query],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout


@mcp.tool()
def artifacts_scaffold(name: str, project_name: str = "", output_dir: str = "") -> str:
    """
    Generate a working test project from an artifact.

    Creates a runnable demo that implements the technique.

    Args:
        name: Name of the artifact to scaffold from
        project_name: Project name (defaults to artifact name)
        output_dir: Output directory (default: ~/projects)
    """
    cmd = ["dz", "artifacts", "scaffold", name]
    if project_name:
        cmd.extend(["--name", project_name])
    if output_dir:
        cmd.extend(["--output", output_dir])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout


# --- Skills Tools (5) ---


@mcp.tool()
def skills_list() -> str:
    """
    List all presentation skills.

    Shows skill name, type, description, and creation date.
    """
    result = subprocess.run(
        ["dz", "skills", "list"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout


@mcp.tool()
def skills_show(name: str) -> str:
    """
    Show a skill's content.

    Args:
        name: Name of the skill to show
    """
    result = subprocess.run(
        ["dz", "skills", "show", name],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout


@mcp.tool()
def skills_create(name: str, skill_type: str, content: str, description: str = "") -> str:
    """
    Create a new skill.

    Args:
        name: Skill name
        skill_type: Skill type (diagnostic, tutorial, etc.)
        content: Skill content (markdown text)
        description: Optional skill description
    """
    cmd = ["dz", "skills", "create", "--name", name, "--type", skill_type]
    if description:
        cmd.extend(["--description", description])

    # Pass content via stdin
    result = subprocess.run(
        cmd,
        input=content,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout or f"Created skill: {name}"


@mcp.tool()
def skills_update(name: str, content: str, description: str = "") -> str:
    """
    Update a skill's content.

    Args:
        name: Name of the skill to update
        content: New skill content
        description: Optional new description
    """
    cmd = ["dz", "skills", "update", name]
    if description:
        cmd.extend(["--description", description])

    # Pass content via stdin
    result = subprocess.run(
        cmd,
        input=content,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout or f"Updated skill: {name}"


@mcp.tool()
def skills_delete(name: str) -> str:
    """
    Delete a skill.

    Args:
        name: Name of the skill to delete
    """
    # Use yes to auto-confirm
    result = subprocess.run(
        ["dz", "skills", "delete", name],
        input="y\n",
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout or f"Deleted skill: {name}"


# --- Projects Tools (6) ---


@mcp.tool()
def project_create(
    name: str,
    about: str = "",
    uses: str = "",
    needs: str = "",
    description: str = "",
    status: str = "active"
) -> str:
    """
    Create a new project with faceted organization.

    Args:
        name: Project name (unique identifier)
        about: Comma-separated 'about' facets (what it's about)
        uses: Comma-separated 'uses' facets (technologies used)
        needs: Comma-separated 'needs' facets (requirements)
        description: Project description
        status: Project status (active, completed, archived)
    """
    cmd = ["dz", "project", "create", name, "--status", status]
    if about:
        cmd.extend(["--about", about])
    if uses:
        cmd.extend(["--uses", uses])
    if needs:
        cmd.extend(["--needs", needs])
    if description:
        cmd.extend(["--description", description])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout or f"Created project: {name}"


@mcp.tool()
def project_show(name: str) -> str:
    """
    Show project details including linked items and skills.

    Args:
        name: Project name
    """
    result = subprocess.run(
        ["dz", "project", "show", name],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout


@mcp.tool()
def project_link(name: str, item: int = 0, skill: str = "") -> str:
    """
    Link an item or skill to a project.

    Args:
        name: Project name
        item: Item ID to link (use 0 to skip)
        skill: Skill name to link (use empty string to skip)
    """
    cmd = ["dz", "project", "link", name]
    if item > 0:
        cmd.extend(["--item", str(item)])
    if skill:
        cmd.extend(["--skill", skill])

    if item == 0 and not skill:
        return "Error: Provide either item ID or skill name to link"

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout or f"Linked to project: {name}"


@mcp.tool()
def project_unlink(name: str, item: int = 0, skill: str = "") -> str:
    """
    Unlink an item or skill from a project.

    Args:
        name: Project name
        item: Item ID to unlink (use 0 to skip)
        skill: Skill name to unlink (use empty string to skip)
    """
    cmd = ["dz", "project", "unlink", name]
    if item > 0:
        cmd.extend(["--item", str(item)])
    if skill:
        cmd.extend(["--skill", skill])

    if item == 0 and not skill:
        return "Error: Provide either item ID or skill name to unlink"

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout or f"Unlinked from project: {name}"


@mcp.tool()
def project_update(
    name: str,
    about: str = "",
    uses: str = "",
    needs: str = "",
    description: str = "",
    status: str = ""
) -> str:
    """
    Update a project's facets, description, or status.

    Args:
        name: Project name
        about: Replace 'about' facets (comma-separated)
        uses: Replace 'uses' facets (comma-separated)
        needs: Replace 'needs' facets (comma-separated)
        description: Update description
        status: Update status (active, completed, archived)
    """
    cmd = ["dz", "project", "update", name]
    if about:
        cmd.extend(["--about", about])
    if uses:
        cmd.extend(["--uses", uses])
    if needs:
        cmd.extend(["--needs", needs])
    if description:
        cmd.extend(["--description", description])
    if status:
        cmd.extend(["--status", status])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout or f"Updated project: {name}"


@mcp.tool()
def project_delete(name: str) -> str:
    """
    Delete a project.

    Args:
        name: Project name to delete
    """
    # Auto-confirm deletion
    result = subprocess.run(
        ["dz", "project", "delete", name],
        input="y\n",
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout or f"Deleted project: {name}"


@mcp.tool()
def project_embed(name: str = "all") -> str:
    """
    Embed project facets for cross-pollination discovery.

    Args:
        name: Project name (or 'all' for all projects)
    """
    result = subprocess.run(
        ["dz", "project", "embed", name],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout or f"Embedded project: {name}"


# --- Sources Tools (2) ---


@mcp.tool()
def sources_list() -> str:
    """
    List all sources (channels, repos).

    Shows source ID, type, name, and URL.
    """
    result = subprocess.run(
        ["dz", "sources", "list"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout


@mcp.tool()
def sources_delete(source_id: int, force: bool = True) -> str:
    """
    Delete a source by ID.

    Note: This only deletes the source record. Associated items remain.

    Args:
        source_id: Source ID to delete
        force: Skip confirmation (default True for MCP)
    """
    cmd = ["dz", "sources", "delete", str(source_id)]
    if force:
        cmd.append("--force")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout or f"Deleted source: {source_id}"


# --- Items Tools (1) ---


@mcp.tool()
def items_delete(item_id: int, force: bool = True) -> str:
    """
    Delete an item by ID.

    This will also delete all associated chunks (embeddings) for this item.

    Args:
        item_id: Item ID to delete
        force: Skip confirmation (default True for MCP)
    """
    cmd = ["dz", "items", "delete", str(item_id)]
    if force:
        cmd.append("--force")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout or f"Deleted item: {item_id}"


# --- Suggest Tools (3) ---


@mcp.tool()
def suggest_youtube(project_name: str, min_score: int = 40, limit: int = 10) -> str:
    """
    Suggest YouTube videos based on project facets.

    Searches YouTube using queries generated from the project's NEEDS, USES,
    and ABOUT facets, then scores and ranks results by educational quality.

    Args:
        project_name: Project name to get suggestions for
        min_score: Minimum quality score (0-100)
        limit: Maximum results to show
    """
    result = subprocess.run(
        ["dz", "suggest", "youtube", project_name,
         "--min-score", str(min_score),
         "--limit", str(limit)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout


@mcp.tool()
def suggest_github(project_name: str, min_score: int = 40, limit: int = 10) -> str:
    """
    Suggest GitHub repositories based on project facets.

    Searches GitHub using queries generated from the project's NEEDS, USES,
    and ABOUT facets, then scores and ranks results.

    Args:
        project_name: Project name to get suggestions for
        min_score: Minimum quality score (0-100)
        limit: Maximum results to show
    """
    result = subprocess.run(
        ["dz", "suggest", "github", project_name,
         "--min-score", str(min_score),
         "--limit", str(limit)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout


@mcp.tool()
def suggest_all(project_name: str, min_score: int = 40, limit: int = 10) -> str:
    """
    Suggest content from all sources (YouTube and GitHub).

    Combines suggestions from YouTube videos and GitHub repositories
    based on the project's facets.

    Args:
        project_name: Project name to get suggestions for
        min_score: Minimum quality score (0-100)
        limit: Maximum results per source
    """
    result = subprocess.run(
        ["dz", "suggest", "all", project_name,
         "--min-score", str(min_score),
         "--limit", str(limit)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout


def main():
    """Run the MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
