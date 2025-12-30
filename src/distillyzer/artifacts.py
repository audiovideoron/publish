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
