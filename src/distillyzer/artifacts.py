"""Manage and use extracted implementation artifacts."""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from . import db
from .embed import get_embedding

load_dotenv()

logger = logging.getLogger(__name__)

# Default Anthropic model - can be overridden via ANTHROPIC_MODEL environment variable
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"


def get_anthropic_model() -> str:
    """Get the Anthropic model to use, from environment or default."""
    return os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)

# Retry configuration for Anthropic API calls
ANTHROPIC_RETRY_EXCEPTIONS = (
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)
anthropic_retry = retry(
    retry=retry_if_exception_type(ANTHROPIC_RETRY_EXCEPTIONS),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

# Lazy-initialized Anthropic client
_anthropic_client = None


class MissingAPIKeyError(Exception):
    """Raised when a required API key is not configured."""
    pass


def get_anthropic_client() -> anthropic.Anthropic:
    """Get the Anthropic client, initializing it lazily with validation.

    Raises:
        MissingAPIKeyError: If ANTHROPIC_API_KEY environment variable is not set.
    """
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise MissingAPIKeyError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Please set it in your .env file or environment to use artifact features. "
                "You can get an API key from https://console.anthropic.com/settings/keys"
            )
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


@anthropic_retry
def _call_claude(system_prompt: str, user_message: str, max_tokens: int = 2000):
    """Make a Claude API call with automatic retry for transient errors.

    Includes automatic retry with exponential backoff for transient errors
    (connection errors, rate limits, timeouts, server errors).
    """
    client = get_anthropic_client()
    return client.messages.create(
        model=get_anthropic_model(),
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

# Default artifacts directory
ARTIFACTS_DIR = Path("artifacts")


def get_artifacts_dir() -> Path:
    """Get or create the artifacts directory."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


def list_artifact_files() -> list[Path]:
    """List all artifact JSON files."""
    artifacts_dir = get_artifacts_dir()
    return sorted(artifacts_dir.glob("*.json"))


def load_artifacts(file_path: Path) -> dict:
    """Load artifacts from a JSON file."""
    with open(file_path) as f:
        return json.load(f)


def list_all_artifacts() -> list[dict]:
    """List all artifacts from all files."""
    all_artifacts = []
    for file_path in list_artifact_files():
        try:
            data = load_artifacts(file_path)
            source_file = file_path.stem
            for artifact in data.get("artifacts", []):
                artifact["_source_file"] = source_file
                all_artifacts.append(artifact)
        except (json.JSONDecodeError, KeyError):
            continue
    return all_artifacts


def find_artifact(name: str) -> Optional[dict]:
    """Find an artifact by name (fuzzy match)."""
    name_lower = name.lower()
    all_artifacts = list_all_artifacts()

    # Exact match first
    for artifact in all_artifacts:
        if artifact.get("name", "").lower() == name_lower:
            return artifact

    # Partial match
    for artifact in all_artifacts:
        if name_lower in artifact.get("name", "").lower():
            return artifact

    return None


def apply_artifact(artifact: dict, context: str) -> dict:
    """
    Use Claude to help apply an artifact to a specific context.

    Args:
        artifact: The artifact dict (name, content, type, context)
        context: Description of the user's current project/situation

    Returns:
        dict with applied content and guidance
    """
    artifact_type = artifact.get("type", "pattern")
    artifact_name = artifact.get("name", "Unknown")
    artifact_content = artifact.get("content", "")
    artifact_context = artifact.get("context", "")

    system_prompt = f"""You are an expert at applying implementation patterns and artifacts to real projects.

You're helping apply this {artifact_type}: "{artifact_name}"

Original artifact:
{artifact_content}

Original context for this artifact:
{artifact_context}

Your job:
1. Understand the user's specific situation
2. Adapt the artifact to their context
3. Provide concrete, actionable implementation steps
4. Include code examples if applicable
5. Warn about any gotchas or things to watch out for"""

    user_message = f"""I want to apply this artifact to my project.

My situation:
{context}

How should I implement this? Give me specific, actionable steps."""

    # Query Claude with automatic retry for transient errors
    try:
        response = _call_claude(system_prompt, user_message, max_tokens=2000)
    except ANTHROPIC_RETRY_EXCEPTIONS as e:
        # Retries exhausted for transient errors
        return {
            "status": "error",
            "message": f"API error after retries: {e}",
        }
    except anthropic.APIStatusError as e:
        return {
            "status": "error",
            "message": f"Anthropic API error: {e.status_code} - {e.message}",
        }

    return {
        "artifact_name": artifact_name,
        "artifact_type": artifact_type,
        "applied_guidance": response.content[0].text,
        "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
    }


def search_artifacts(query: str) -> list[dict]:
    """Search artifacts by name or content."""
    query_lower = query.lower()
    results = []

    for artifact in list_all_artifacts():
        name = artifact.get("name", "").lower()
        content = artifact.get("content", "").lower()
        context = artifact.get("context", "").lower()

        if query_lower in name or query_lower in content or query_lower in context:
            results.append(artifact)

    return results


def scaffold_project(
    artifact: dict,
    project_name: str,
    output_dir: Path,
) -> dict:
    """
    Generate a working test project that demonstrates an artifact.

    Args:
        artifact: The artifact to implement
        project_name: Name for the generated project
        output_dir: Where to create the project

    Returns:
        dict with generated files and instructions
    """
    artifact_type = artifact.get("type", "pattern")
    artifact_name = artifact.get("name", "Unknown")
    artifact_content = artifact.get("content", "")
    artifact_context = artifact.get("context", "")

    system_prompt = """You are an expert at creating minimal, working code examples that demonstrate concepts.

Your job is to generate a complete, runnable test project that implements a specific pattern or technique.

Requirements:
1. Generate ACTUAL working code, not pseudocode
2. Keep it minimal - just enough to demonstrate the concept
3. Use Python unless the concept requires something else
4. Include a README.md with setup and run instructions
5. Include comments explaining what each part demonstrates
6. Make it immediately runnable (no complex setup)

Output format: Use this exact structure with ===FILE=== markers:

===META===
run_command: python main.py
description: Brief description here
===END_META===

===FILE=== main.py
# file contents here
===END_FILE===

===FILE=== README.md
# README contents here
===END_FILE===

Use ===FILE=== path and ===END_FILE=== to delimit each file."""

    user_message = f"""Create a test project called "{project_name}" that demonstrates this {artifact_type}:

Name: {artifact_name}

Content:
{artifact_content}

Context: {artifact_context}

Generate a minimal working project that I can run to see this technique in action.
The project should be self-contained and educational - I want to learn by seeing it work."""

    # Query Claude with automatic retry for transient errors
    try:
        response = _call_claude(system_prompt, user_message, max_tokens=4000)
    except ANTHROPIC_RETRY_EXCEPTIONS as e:
        # Retries exhausted for transient errors
        return {
            "status": "error",
            "message": f"API error after retries: {e}",
        }
    except anthropic.APIStatusError as e:
        return {
            "status": "error",
            "message": f"Anthropic API error: {e.status_code} - {e.message}",
        }

    response_text = response.content[0].text

    # Parse the custom format with ===FILE=== markers
    import re

    # Extract metadata
    run_command = ""
    description = ""
    meta_match = re.search(r'===META===(.*?)===END_META===', response_text, re.DOTALL)
    if meta_match:
        meta_text = meta_match.group(1)
        for line in meta_text.strip().split('\n'):
            if line.startswith('run_command:'):
                run_command = line.split(':', 1)[1].strip()
            elif line.startswith('description:'):
                description = line.split(':', 1)[1].strip()

    # Extract files
    files = {}
    file_pattern = r'===FILE===\s*(\S+)\s*\n(.*?)===END_FILE==='
    for match in re.finditer(file_pattern, response_text, re.DOTALL):
        file_path = match.group(1).strip()
        file_content = match.group(2)
        files[file_path] = file_content

    if not files:
        return {
            "status": "error",
            "message": "No files found in scaffold response",
            "raw_response": response_text[:1000],
        }

    result = {
        "files": files,
        "run_command": run_command,
        "description": description,
    }

    # Create the project directory
    project_dir = output_dir / project_name
    project_dir.mkdir(parents=True, exist_ok=True)

    # Write all files
    files_created = []
    for file_path, content in result.get("files", {}).items():
        full_path = project_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        files_created.append(str(full_path))

    return {
        "status": "success",
        "project_dir": str(project_dir),
        "files_created": files_created,
        "run_command": result.get("run_command", ""),
        "description": result.get("description", ""),
        "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
    }


def analyze_for_demo(topic: str, num_chunks: int = 10) -> dict:
    """
    Analyze knowledge base content and propose a hello world demo.

    Args:
        topic: Topic to search for
        num_chunks: Number of chunks to analyze

    Returns:
        dict with sources, core_lesson, and proposed_demo
    """
    # Search knowledge base
    query_embedding = get_embedding(topic)
    chunks = db.search_chunks(query_embedding, limit=num_chunks)

    if not chunks:
        return {
            "status": "no_content",
            "message": "No relevant content found in knowledge base",
        }

    # Build context from chunks
    context_parts = []
    sources = []
    for chunk in chunks:
        source = chunk["item_title"]
        ts = chunk.get("timestamp_start")
        ts_str = f" @ {int(ts)//60}:{int(ts)%60:02d}" if ts else ""
        context_parts.append(f"[{source}{ts_str}]\n{chunk['content']}")
        if source not in [s["title"] for s in sources]:
            sources.append({
                "title": source,
                "url": chunk.get("item_url"),
            })

    context = "\n\n---\n\n".join(context_parts)

    # Ask Claude to analyze and propose a demo
    system_prompt = """You are an expert at understanding educational content and designing minimal demonstrations.

Your job is to:
1. Understand the CORE lesson being taught
2. Propose the simplest possible "hello world" demo that proves someone understood the concept

The demo should be:
- Minimal - the absolute simplest thing that demonstrates the concept
- Runnable - actual working code, not pseudocode
- Self-contained - no complex dependencies
- Educational - clearly shows the concept in action

Output format:
CORE_LESSON: One sentence describing the essential insight
DEMO_CONCEPT: 2-3 sentences describing what the hello world demo would do
PROVES: What understanding does this demo prove?"""

    user_message = f"""Topic: {topic}

Content from knowledge base:

{context}

---

Analyze this content and propose a minimal hello world demo."""

    # Query Claude with automatic retry for transient errors
    try:
        response = _call_claude(system_prompt, user_message, max_tokens=1000)
    except ANTHROPIC_RETRY_EXCEPTIONS as e:
        # Retries exhausted for transient errors
        return {
            "status": "error",
            "message": f"API error after retries: {e}",
        }
    except anthropic.APIStatusError as e:
        return {
            "status": "error",
            "message": f"Anthropic API error: {e.status_code} - {e.message}",
        }

    response_text = response.content[0].text

    # Parse response
    core_lesson = ""
    demo_concept = ""
    proves = ""

    for line in response_text.split("\n"):
        if line.startswith("CORE_LESSON:"):
            core_lesson = line.split(":", 1)[1].strip()
        elif line.startswith("DEMO_CONCEPT:"):
            demo_concept = line.split(":", 1)[1].strip()
        elif line.startswith("PROVES:"):
            proves = line.split(":", 1)[1].strip()

    # Handle multi-line values
    if not demo_concept:
        # Try to extract from full text
        if "DEMO_CONCEPT:" in response_text:
            parts = response_text.split("DEMO_CONCEPT:")[1]
            if "PROVES:" in parts:
                demo_concept = parts.split("PROVES:")[0].strip()
            else:
                demo_concept = parts.strip()

    return {
        "status": "success",
        "topic": topic,
        "sources": sources,
        "core_lesson": core_lesson,
        "demo_concept": demo_concept,
        "proves": proves,
        "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
    }


def build_demo(
    topic: str,
    core_lesson: str,
    demo_concept: str,
    project_name: str,
    output_dir: Path,
    num_chunks: int = 10,
) -> dict:
    """
    Build a hello world demo based on analyzed lesson.

    Args:
        topic: Original topic
        core_lesson: The core lesson identified
        demo_concept: The proposed demo concept
        project_name: Name for the project
        output_dir: Where to create it

    Returns:
        dict with generated files
    """
    # Get context again for implementation details
    query_embedding = get_embedding(topic)
    chunks = db.search_chunks(query_embedding, limit=num_chunks)

    context_parts = []
    for chunk in chunks:
        context_parts.append(chunk['content'])
    context = "\n\n".join(context_parts[:5])  # Use top 5 for implementation

    system_prompt = """You are an expert at creating minimal, working code demonstrations.

Create a hello world demo that demonstrates a specific concept. The demo must be:
1. MINIMAL - absolutely the simplest thing that works
2. WORKING - actual runnable code that EXECUTES the concept, not just prints about it
3. CLEAR - obvious what it's demonstrating
4. SELF-CONTAINED - single ./run.sh command handles everything

CRITICAL RULES:
- The demo must ACTUALLY RUN AND DO THE THING, not just print descriptions
- If you need external dependencies, list them in requirements.txt
- The run.sh script handles venv creation, dependency installation, and execution
- Prefer standard library when possible, but use real dependencies if needed for the concept
- The user should see the concept IN ACTION, not read about it

Output format: Use ===FILE=== markers:

===META===
description: One line description
===END_META===

===FILE=== main.py
# The actual demo code that RUNS the concept
===END_FILE===

===FILE=== requirements.txt
# Dependencies (one per line, empty if none needed)
===END_FILE===

===FILE=== README.md
# Brief explanation
===END_FILE===

IMPORTANT: Do NOT include run.sh - it will be generated automatically."""

    user_message = f"""Create a hello world demo for this concept:

Topic: {topic}
Core Lesson: {core_lesson}
Demo Concept: {demo_concept}

Reference content from the lesson:
{context[:3000]}

Build the simplest possible demo that proves understanding of this concept."""

    # Query Claude with automatic retry for transient errors
    try:
        response = _call_claude(system_prompt, user_message, max_tokens=4000)
    except ANTHROPIC_RETRY_EXCEPTIONS as e:
        # Retries exhausted for transient errors
        return {
            "status": "error",
            "message": f"API error after retries: {e}",
        }
    except anthropic.APIStatusError as e:
        return {
            "status": "error",
            "message": f"Anthropic API error: {e.status_code} - {e.message}",
        }

    response_text = response.content[0].text

    # Parse the response (same as scaffold_project)
    import re

    run_command = ""
    description = ""
    meta_match = re.search(r'===META===(.*?)===END_META===', response_text, re.DOTALL)
    if meta_match:
        meta_text = meta_match.group(1)
        for line in meta_text.strip().split('\n'):
            if line.startswith('run_command:'):
                run_command = line.split(':', 1)[1].strip()
            elif line.startswith('description:'):
                description = line.split(':', 1)[1].strip()

    files = {}
    file_pattern = r'===FILE===\s*(\S+)\s*\n(.*?)===END_FILE==='
    for match in re.finditer(file_pattern, response_text, re.DOTALL):
        file_path = match.group(1).strip()
        file_content = match.group(2)
        files[file_path] = file_content

    if not files:
        return {
            "status": "error",
            "message": "No files generated",
            "raw_response": response_text[:1000],
        }

    # Create project
    project_dir = output_dir / project_name
    project_dir.mkdir(parents=True, exist_ok=True)

    files_created = []
    for file_path, content in files.items():
        full_path = project_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        files_created.append(str(full_path))

    # Generate run.sh that handles everything
    run_sh_content = '''#!/bin/bash
set -e

cd "$(dirname "$0")"

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies if requirements.txt exists and has content
if [ -f "requirements.txt" ] && [ -s "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install -q -r requirements.txt
fi

# Run the demo
echo ""
python main.py
'''
    run_sh_path = project_dir / "run.sh"
    run_sh_path.write_text(run_sh_content)
    run_sh_path.chmod(0o755)  # Make executable
    files_created.append(str(run_sh_path))

    return {
        "status": "success",
        "project_dir": str(project_dir),
        "files_created": files_created,
        "run_command": "./run.sh",
        "description": description,
        "core_lesson": core_lesson,
        "demo_concept": demo_concept,
        "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
    }
