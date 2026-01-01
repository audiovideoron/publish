"""Query the knowledge base using semantic search and Claude."""

import os

import anthropic
from dotenv import load_dotenv

from . import db
from .embed import get_embedding

load_dotenv()

claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def search(query: str, limit: int = 10) -> list[dict]:
    """
    Semantic search across the knowledge base.
    Returns chunks sorted by similarity.
    """
    query_embedding = get_embedding(query)
    return db.search_chunks(query_embedding, limit=limit)


def format_timestamp(seconds: float | None) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    if seconds is None:
        return ""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_context(chunks: list[dict]) -> str:
    """Format search results as context for Claude."""
    if not chunks:
        return "No relevant content found in the knowledge base."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source_info = f"[{chunk['item_title']}]"
        if chunk.get("timestamp_start") is not None:
            ts = format_timestamp(chunk["timestamp_start"])
            source_info += f" @ {ts}"
        if chunk.get("item_url"):
            source_info += f"\n{chunk['item_url']}"

        context_parts.append(f"--- Source {i} {source_info} ---\n{chunk['content']}")

    return "\n\n".join(context_parts)


def ask(question: str, num_sources: int = 5) -> dict:
    """
    Ask a question using RAG: search for context, then query Claude.
    Returns dict with answer and sources.
    """
    # Search for relevant chunks
    chunks = search(question, limit=num_sources)
    context = format_context(chunks)

    # Build prompt
    system_prompt = """You are a helpful assistant that answers questions based on the provided context from the user's knowledge base.

The context comes from YouTube video transcripts and GitHub code repositories that the user has collected.

Guidelines:
- Answer based primarily on the provided context
- If the context doesn't contain enough information, say so
- Reference specific sources when applicable
- For video content, mention timestamps if available
- Be concise but thorough"""

    user_message = f"""Context from knowledge base:

{context}

---

Question: {question}"""

    # Query Claude
    try:
        response = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.APIConnectionError as e:
        raise RuntimeError(f"Failed to connect to Anthropic API: {e}") from e
    except anthropic.RateLimitError as e:
        raise RuntimeError(f"Anthropic API rate limit exceeded: {e}") from e
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"Anthropic API error (status {e.status_code}): {e.message}") from e

    return {
        "answer": response.content[0].text,
        "sources": chunks,
        "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
    }


def chat_turn(
    question: str,
    history: list[dict],
    num_sources: int = 5,
) -> dict:
    """
    Single turn in a chat conversation with memory.
    history is a list of {"role": "user"|"assistant", "content": str}
    """
    # Search for relevant chunks
    chunks = search(question, limit=num_sources)
    context = format_context(chunks)

    system_prompt = """You are a helpful assistant that answers questions based on the provided context from the user's knowledge base.

The context comes from YouTube video transcripts and GitHub code repositories that the user has collected.

Guidelines:
- Answer based primarily on the provided context
- If the context doesn't contain enough information, say so
- Reference specific sources when applicable
- For video content, mention timestamps if available
- Be concise but thorough
- Remember the conversation history for follow-up questions"""

    # Build messages with history
    messages = []
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current question with context
    current_message = f"""Context from knowledge base:

{context}

---

Question: {question}"""
    messages.append({"role": "user", "content": current_message})

    # Query Claude
    try:
        response = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=system_prompt,
            messages=messages,
        )
    except anthropic.APIConnectionError as e:
        raise RuntimeError(f"Failed to connect to Anthropic API: {e}") from e
    except anthropic.RateLimitError as e:
        raise RuntimeError(f"Anthropic API rate limit exceeded: {e}") from e
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"Anthropic API error (status {e.status_code}): {e.message}") from e

    return {
        "answer": response.content[0].text,
        "sources": chunks,
        "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
    }
