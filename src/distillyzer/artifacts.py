"""Manage and use extracted implementation artifacts."""

import json
import os
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()

claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

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

    response = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

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

    response = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

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
