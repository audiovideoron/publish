"""Extract implementation artifacts from knowledge base content."""

import os
import json
from typing import Literal

import anthropic
from dotenv import load_dotenv

from . import db
from .embed import get_embedding

load_dotenv()

claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

ArtifactType = Literal["prompt", "pattern", "checklist", "rule", "tool", "all"]

EXTRACTION_PROMPTS = {
    "prompt": """Extract reusable PROMPT TEMPLATES from this content.

Look for:
- System prompts or instructions given to AI models
- Prompt structures that can be adapted for other use cases
- Effective prompting patterns mentioned or demonstrated

For each template found, provide:
- A descriptive name
- The template itself (with {{placeholders}} for variable parts)
- When to use it
- Example usage""",

    "pattern": """Extract reusable CODE/ARCHITECTURE PATTERNS from this content.

Look for:
- Design patterns for agentic systems
- Code structures and architectures mentioned
- Integration patterns between tools/models
- Error handling approaches
- State management strategies

For each pattern found, provide:
- Pattern name
- Problem it solves
- Implementation approach
- Example or pseudocode if available""",

    "checklist": """Extract actionable CHECKLISTS from this content.

Look for:
- Step-by-step processes mentioned
- Setup procedures
- Debugging workflows
- Deployment steps
- Review criteria

For each checklist found, provide:
- Checklist title
- Context (when to use this)
- Ordered steps with clear actions""",

    "rule": """Extract "WHEN X, DO Y" RULES from this content.

Look for:
- Conditional guidance (if this happens, do that)
- Best practices phrased as rules
- Error handling rules
- Decision criteria
- Warnings or gotchas

For each rule found, provide:
- Rule name (short, memorable)
- Condition (WHEN...)
- Action (DO...)
- Rationale (WHY...)""",

    "tool": """Extract TOOL DEFINITIONS from this content.

Look for:
- Tool schemas or definitions
- Function signatures for AI tools
- API endpoint patterns
- Tool use examples

For each tool found, provide:
- Tool name
- Description
- Input parameters (with types)
- Output format
- Example invocation""",
}


def search_context(query: str, num_chunks: int = 10) -> tuple[str, list[dict]]:
    """Search knowledge base and return context with source info."""
    query_embedding = get_embedding(query)
    chunks = db.search_chunks(query_embedding, limit=num_chunks)

    if not chunks:
        return "", []

    context_parts = []
    for chunk in chunks:
        source = chunk["item_title"]
        ts = chunk.get("timestamp_start")
        ts_str = f" @ {int(ts)//60}:{int(ts)%60:02d}" if ts else ""
        content = chunk["content"]
        context_parts.append(f"[{source}{ts_str}]\n{content}")

    return "\n\n---\n\n".join(context_parts), chunks


def extract_artifacts(
    topic: str,
    artifact_type: ArtifactType = "all",
    num_sources: int = 10,
) -> dict:
    """
    Extract implementation artifacts from knowledge base content.

    Args:
        topic: Topic or query to search for
        artifact_type: Type of artifacts to extract ("prompt", "pattern", etc.)
        num_sources: Number of source chunks to use

    Returns:
        dict with artifacts, sources, and metadata
    """
    # Get relevant context
    context, chunks = search_context(topic, num_chunks=num_sources)

    if not context:
        return {
            "artifacts": [],
            "sources": [],
            "status": "no_content",
            "message": "No relevant content found in knowledge base",
        }

    # Build extraction prompt
    if artifact_type == "all":
        type_instructions = "\n\n".join([
            f"## {t.upper()}\n{p}" for t, p in EXTRACTION_PROMPTS.items()
        ])
    else:
        type_instructions = EXTRACTION_PROMPTS.get(artifact_type, "")

    system_prompt = f"""You are an expert at extracting actionable implementation artifacts from educational content.

Your job is to find concrete, reusable artifacts that a developer can immediately apply to their work.

{type_instructions}

IMPORTANT:
- Only extract artifacts that are ACTUALLY present in the content
- Be specific and actionable - vague summaries are not useful
- Include enough context that the artifact can be used independently
- If no artifacts of a type are found, say so explicitly
- Format output as valid JSON"""

    user_message = f"""Topic: {topic}

Content from knowledge base:

{context}

---

Extract all {artifact_type if artifact_type != 'all' else ''} implementation artifacts from this content.
Return as JSON with structure:
{{
    "artifacts": [
        {{
            "type": "prompt|pattern|checklist|rule|tool",
            "name": "descriptive name",
            "content": "the actual artifact content",
            "context": "when/how to use this",
            "source": "which video/file this came from"
        }}
    ],
    "notes": "any overall observations about the content"
}}"""

    # Query Claude
    try:
        response = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.APIConnectionError as e:
        return {
            "artifacts": [],
            "sources": [],
            "status": "api_error",
            "message": f"Failed to connect to Anthropic API: {e}",
        }
    except anthropic.RateLimitError as e:
        return {
            "artifacts": [],
            "sources": [],
            "status": "rate_limit",
            "message": f"Rate limit exceeded: {e}",
        }
    except anthropic.APIStatusError as e:
        return {
            "artifacts": [],
            "sources": [],
            "status": "api_error",
            "message": f"Anthropic API error ({e.status_code}): {e.message}",
        }

    response_text = response.content[0].text

    # Parse JSON from response
    try:
        # Try to find JSON in the response
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0]
        else:
            json_str = response_text

        result = json.loads(json_str.strip())
        artifacts = result.get("artifacts", [])
        notes = result.get("notes", "")
    except json.JSONDecodeError:
        # If JSON parsing fails, return raw text
        artifacts = [{
            "type": "raw",
            "name": "Extraction Result",
            "content": response_text,
            "context": "Could not parse structured output",
            "source": "multiple",
        }]
        notes = "JSON parsing failed - raw output returned"

    return {
        "artifacts": artifacts,
        "sources": [
            {
                "title": c["item_title"],
                "url": c.get("item_url"),
                "timestamp": c.get("timestamp_start"),
                "similarity": c.get("similarity"),
            }
            for c in chunks
        ],
        "notes": notes,
        "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
        "status": "success",
    }


def extract_from_item(item_id: int, artifact_type: ArtifactType = "all") -> dict:
    """Extract artifacts from a specific item (video, file)."""
    # Get all chunks for this item
    items = db.get_items_with_chunks(item_id)
    if not items:
        return {
            "artifacts": [],
            "status": "not_found",
            "message": f"Item {item_id} not found",
        }

    item = items[0]
    if not item.get("chunks"):
        return {
            "artifacts": [],
            "status": "no_chunks",
            "message": f"Item '{item['title']}' has no indexed content",
        }

    # Combine all chunks
    content = "\n\n".join([c["content"] for c in item["chunks"]])

    # Build extraction prompt (same as above but with full content)
    if artifact_type == "all":
        type_instructions = "\n\n".join([
            f"## {t.upper()}\n{p}" for t, p in EXTRACTION_PROMPTS.items()
        ])
    else:
        type_instructions = EXTRACTION_PROMPTS.get(artifact_type, "")

    system_prompt = f"""You are an expert at extracting actionable implementation artifacts from educational content.

Your job is to find concrete, reusable artifacts that a developer can immediately apply to their work.

{type_instructions}

IMPORTANT:
- Only extract artifacts that are ACTUALLY present in the content
- Be specific and actionable - vague summaries are not useful
- Include enough context that the artifact can be used independently
- If no artifacts of a type are found, say so explicitly
- Format output as valid JSON"""

    user_message = f"""Source: {item['title']}

Full content:

{content[:15000]}  # Limit to avoid token overflow

---

Extract all {artifact_type if artifact_type != 'all' else ''} implementation artifacts from this content.
Return as JSON with structure:
{{
    "artifacts": [
        {{
            "type": "prompt|pattern|checklist|rule|tool",
            "name": "descriptive name",
            "content": "the actual artifact content",
            "context": "when/how to use this"
        }}
    ],
    "notes": "any overall observations about the content"
}}"""

    try:
        response = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.APIConnectionError as e:
        return {
            "item_id": item_id,
            "item_title": item["title"],
            "artifacts": [],
            "status": "api_error",
            "message": f"Failed to connect to Anthropic API: {e}",
        }
    except anthropic.RateLimitError as e:
        return {
            "item_id": item_id,
            "item_title": item["title"],
            "artifacts": [],
            "status": "rate_limit",
            "message": f"Rate limit exceeded: {e}",
        }
    except anthropic.APIStatusError as e:
        return {
            "item_id": item_id,
            "item_title": item["title"],
            "artifacts": [],
            "status": "api_error",
            "message": f"Anthropic API error ({e.status_code}): {e.message}",
        }

    response_text = response.content[0].text

    try:
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0]
        else:
            json_str = response_text

        result = json.loads(json_str.strip())
        artifacts = result.get("artifacts", [])
        notes = result.get("notes", "")
    except json.JSONDecodeError:
        artifacts = [{
            "type": "raw",
            "name": "Extraction Result",
            "content": response_text,
            "context": "Could not parse structured output",
        }]
        notes = "JSON parsing failed - raw output returned"

    return {
        "item_id": item_id,
        "item_title": item["title"],
        "artifacts": artifacts,
        "notes": notes,
        "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
        "status": "success",
    }
