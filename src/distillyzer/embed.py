"""Chunking and embedding using OpenAI API."""

import logging
import os

import tiktoken
from openai import OpenAI, OpenAIError, APIConnectionError, RateLimitError, APITimeoutError
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from . import db

logger = logging.getLogger(__name__)

# Retry configuration for OpenAI API calls
OPENAI_RETRY_EXCEPTIONS = (APIConnectionError, RateLimitError, APITimeoutError)
openai_retry = retry(
    retry=retry_if_exception_type(OPENAI_RETRY_EXCEPTIONS),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
MAX_TOKENS = 8000  # Leave buffer for safety


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken."""
    enc = tiktoken.get_encoding(model)
    return len(enc.encode(text))


def chunk_text(
    text: str,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[str]:
    """
    Split text into chunks of approximately max_tokens.
    Uses sentence boundaries when possible.
    """
    if not text.strip():
        return []

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    if len(tokens) <= max_tokens:
        return [text]

    # Split by sentences first
    sentences = text.replace("\n", " ").split(". ")
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if not sentence.endswith("."):
            sentence += "."

        sentence_tokens = len(enc.encode(sentence))

        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            # Save current chunk
            chunks.append(" ".join(current_chunk))

            # Start new chunk with overlap (last few sentences)
            overlap_text = ""
            overlap_count = 0
            for s in reversed(current_chunk):
                if overlap_count + len(enc.encode(s)) < overlap_tokens:
                    overlap_text = s + " " + overlap_text
                    overlap_count += len(enc.encode(s))
                else:
                    break

            current_chunk = [overlap_text.strip()] if overlap_text.strip() else []
            current_tokens = overlap_count

        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_code(text: str, max_tokens: int = 500) -> list[str]:
    """
    Split code into chunks, trying to preserve function/class boundaries.
    """
    if not text.strip():
        return []

    enc = tiktoken.get_encoding("cl100k_base")

    if len(enc.encode(text)) <= max_tokens:
        return [text]

    # Split by double newlines (paragraph/function boundaries)
    blocks = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_tokens = 0

    for block in blocks:
        block_tokens = len(enc.encode(block))

        if block_tokens > max_tokens:
            # Block too large, split by lines
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            lines = block.split("\n")
            line_chunk = []
            line_tokens = 0
            for line in lines:
                lt = len(enc.encode(line))
                if line_tokens + lt > max_tokens and line_chunk:
                    chunks.append("\n".join(line_chunk))
                    line_chunk = []
                    line_tokens = 0
                line_chunk.append(line)
                line_tokens += lt
            if line_chunk:
                chunks.append("\n".join(line_chunk))
        elif current_tokens + block_tokens > max_tokens:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [block]
            current_tokens = block_tokens
        else:
            current_chunk.append(block)
            current_tokens += block_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


@openai_retry
def get_embedding(text: str) -> list[float]:
    """Get embedding for a single text using OpenAI API.

    Includes automatic retry with exponential backoff for transient errors
    (connection errors, rate limits, timeouts).
    """
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding
    except OPENAI_RETRY_EXCEPTIONS:
        # Let tenacity handle these via the decorator
        raise
    except OpenAIError as e:
        raise RuntimeError(f"OpenAI embedding API error: {e}") from e


@openai_retry
def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Get embeddings for multiple texts in a single API call.

    Includes automatic retry with exponential backoff for transient errors
    (connection errors, rate limits, timeouts).
    """
    if not texts:
        return []

    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        return [item.embedding for item in response.data]
    except OPENAI_RETRY_EXCEPTIONS:
        # Let tenacity handle these via the decorator
        raise
    except OpenAIError as e:
        raise RuntimeError(f"OpenAI batch embedding API error: {e}") from e


def embed_transcript_chunks(
    item_id: int,
    timed_chunks: list[dict],
) -> int:
    """
    Embed transcript chunks and store in DB.
    timed_chunks should have: text, start, end
    Returns number of chunks stored.
    """
    if not timed_chunks:
        return 0

    texts = [c["text"] for c in timed_chunks]
    embeddings = get_embeddings_batch(texts)

    db_chunks = []
    for i, (chunk, embedding) in enumerate(zip(timed_chunks, embeddings)):
        db_chunks.append({
            "item_id": item_id,
            "content": chunk["text"],
            "chunk_index": i,
            "timestamp_start": chunk.get("start"),
            "timestamp_end": chunk.get("end"),
            "embedding": embedding,
        })

    db.create_chunks_batch(db_chunks)
    return len(db_chunks)


def embed_project_facets(project_id: int) -> dict:
    """
    Embed a project's facets (about, uses, needs) for cross-pollination.
    Returns dict with counts of embedded facets.
    """
    project = db.get_project_by_id(project_id)
    if not project:
        raise ValueError(f"Project {project_id} not found")

    embedded = {"about": False, "uses": False, "needs": False}

    # Embed each non-empty facet as a combined string
    if project.get("facet_about"):
        text = ", ".join(project["facet_about"])
        embedding = get_embedding(f"About: {text}")
        db.update_project_embeddings(project_id, embedding_about=embedding)
        embedded["about"] = True

    if project.get("facet_uses"):
        text = ", ".join(project["facet_uses"])
        embedding = get_embedding(f"Uses: {text}")
        db.update_project_embeddings(project_id, embedding_uses=embedding)
        embedded["uses"] = True

    if project.get("facet_needs"):
        text = ", ".join(project["facet_needs"])
        embedding = get_embedding(f"Needs: {text}")
        db.update_project_embeddings(project_id, embedding_needs=embedding)
        embedded["needs"] = True

    return embedded


def embed_all_projects() -> list[dict]:
    """
    Embed all projects that have facets but no embeddings.
    Returns list of projects that were embedded.
    """
    projects = db.get_projects_needing_embeddings()
    results = []

    for project in projects:
        try:
            embedded = embed_project_facets(project["id"])
            results.append({
                "id": project["id"],
                "name": project["name"],
                "embedded": embedded,
            })
        except Exception as e:
            results.append({
                "id": project["id"],
                "name": project["name"],
                "error": str(e),
            })

    return results


def embed_text_content(
    item_id: int,
    text: str,
    is_code: bool = False,
) -> int:
    """
    Chunk and embed text content (transcript or code).
    Returns number of chunks stored.
    """
    if is_code:
        chunks = chunk_code(text)
    else:
        chunks = chunk_text(text)

    if not chunks:
        return 0

    embeddings = get_embeddings_batch(chunks)

    db_chunks = []
    for i, (content, embedding) in enumerate(zip(chunks, embeddings)):
        db_chunks.append({
            "item_id": item_id,
            "content": content,
            "chunk_index": i,
            "timestamp_start": None,
            "timestamp_end": None,
            "embedding": embedding,
        })

    db.create_chunks_batch(db_chunks)
    return len(db_chunks)


def embed_repo_files(
    file_items: list[dict],
    progress_callback: callable = None,
) -> dict:
    """
    Embed code files from a harvested repo.
    file_items should have: item_id, path, content, extension

    Returns dict with total_files, total_chunks, errors.
    """
    total_chunks = 0
    errors = []

    for i, file_item in enumerate(file_items):
        try:
            # Skip non-code files from embedding (keep .md and .txt as text)
            is_code = file_item["extension"] not in {".md", ".txt"}

            num_chunks = embed_text_content(
                file_item["item_id"],
                file_item["content"],
                is_code=is_code,
            )
            total_chunks += num_chunks

            if progress_callback:
                progress_callback(i + 1, len(file_items), file_item["path"], num_chunks)

        except Exception as e:
            errors.append({"path": file_item["path"], "error": str(e)})

    return {
        "total_files": len(file_items),
        "total_chunks": total_chunks,
        "errors": errors,
    }
