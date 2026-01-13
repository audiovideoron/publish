"""Chunking and embedding using OpenAI API."""

import logging
import os
import threading
import time

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


class RateLimiter:
    """Token bucket rate limiter for API calls.

    Implements a token bucket algorithm to prevent hitting API rate limits.
    Supports both request-per-minute (RPM) and tokens-per-minute (TPM) limits.

    OpenAI embedding rate limits vary by tier but typically:
    - Tier 1: 500 RPM, 1M TPM
    - Tier 2: 500 RPM, 1M TPM
    - Tier 3+: Higher limits

    Default values are conservative to work across tiers.
    """

    def __init__(
        self,
        requests_per_minute: int = 300,
        tokens_per_minute: int = 500_000,
    ):
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum API requests per minute.
            tokens_per_minute: Maximum tokens per minute.
        """
        self.rpm_limit = requests_per_minute
        self.tpm_limit = tokens_per_minute

        # Token bucket state
        self._request_tokens = float(requests_per_minute)
        self._token_tokens = float(tokens_per_minute)
        self._last_update = time.monotonic()
        self._lock = threading.Lock()

        # Refill rates (tokens per second)
        self._rpm_refill_rate = requests_per_minute / 60.0
        self._tpm_refill_rate = tokens_per_minute / 60.0

    def _refill_buckets(self) -> None:
        """Refill token buckets based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now

        # Refill request bucket
        self._request_tokens = min(
            self.rpm_limit,
            self._request_tokens + elapsed * self._rpm_refill_rate,
        )

        # Refill token bucket
        self._token_tokens = min(
            self.tpm_limit,
            self._token_tokens + elapsed * self._tpm_refill_rate,
        )

    def acquire(self, num_tokens: int = 0) -> float:
        """Acquire permission to make an API call, blocking if necessary.

        Args:
            num_tokens: Estimated number of tokens in the request.

        Returns:
            Time spent waiting (in seconds).
        """
        wait_time = 0.0

        with self._lock:
            self._refill_buckets()

            # Calculate wait time if we need to wait
            request_wait = 0.0
            token_wait = 0.0

            if self._request_tokens < 1:
                request_wait = (1 - self._request_tokens) / self._rpm_refill_rate

            if num_tokens > 0 and self._token_tokens < num_tokens:
                token_wait = (num_tokens - self._token_tokens) / self._tpm_refill_rate

            wait_time = max(request_wait, token_wait)

            if wait_time > 0:
                logger.debug(f"Rate limiter waiting {wait_time:.2f}s")

        # Wait outside the lock
        if wait_time > 0:
            time.sleep(wait_time)
            with self._lock:
                self._refill_buckets()

        # Consume tokens
        with self._lock:
            self._request_tokens -= 1
            if num_tokens > 0:
                self._token_tokens -= num_tokens

        return wait_time


# Global rate limiter instance
# Can be configured via environment variables or replaced at runtime
_rate_limiter = None


def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        rpm = int(os.getenv("OPENAI_RPM_LIMIT", "300"))
        tpm = int(os.getenv("OPENAI_TPM_LIMIT", "500000"))
        _rate_limiter = RateLimiter(requests_per_minute=rpm, tokens_per_minute=tpm)
        logger.info(f"Rate limiter initialized: {rpm} RPM, {tpm} TPM")
    return _rate_limiter


def reset_rate_limiter(
    requests_per_minute: int = None,
    tokens_per_minute: int = None,
) -> RateLimiter:
    """Reset the global rate limiter with new limits.

    Args:
        requests_per_minute: New RPM limit (default: from env or 300).
        tokens_per_minute: New TPM limit (default: from env or 500000).

    Returns:
        The new rate limiter instance.
    """
    global _rate_limiter
    rpm = requests_per_minute or int(os.getenv("OPENAI_RPM_LIMIT", "300"))
    tpm = tokens_per_minute or int(os.getenv("OPENAI_TPM_LIMIT", "500000"))
    _rate_limiter = RateLimiter(requests_per_minute=rpm, tokens_per_minute=tpm)
    logger.info(f"Rate limiter reset: {rpm} RPM, {tpm} TPM")
    return _rate_limiter

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

# Lazy-initialized OpenAI client
_openai_client = None

# Default embedding dimensions for known OpenAI models
EMBEDDING_MODEL_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# Configurable via environment variables
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv(
    "OPENAI_EMBEDDING_DIM",
    str(EMBEDDING_MODEL_DIMS.get(EMBEDDING_MODEL, 1536))
))
MAX_TOKENS = 8000  # Leave buffer for safety

# Configurable max tokens for chunking (default: 500)
EMBED_MAX_TOKENS = int(os.getenv("EMBED_MAX_TOKENS", "500"))


def get_embedding_model_info() -> dict:
    """Get current embedding model configuration.

    Returns:
        Dict with 'model', 'dim', and 'known_models' keys.
    """
    return {
        "model": EMBEDDING_MODEL,
        "dim": EMBEDDING_DIM,
        "known_models": EMBEDDING_MODEL_DIMS,
    }


class MissingAPIKeyError(Exception):
    """Raised when a required API key is not configured."""
    pass


def get_openai_client() -> OpenAI:
    """Get the OpenAI client, initializing it lazily with validation.

    Raises:
        MissingAPIKeyError: If OPENAI_API_KEY environment variable is not set.
    """
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise MissingAPIKeyError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it in your .env file or environment to use embedding features. "
                "You can get an API key from https://platform.openai.com/api-keys"
            )
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken."""
    enc = tiktoken.get_encoding(model)
    return len(enc.encode(text))


def chunk_text(
    text: str,
    max_tokens: int = None,
    overlap_tokens: int = 50,
) -> list[str]:
    """
    Split text into chunks of approximately max_tokens.
    Uses sentence boundaries when possible.

    Args:
        text: The text to chunk.
        max_tokens: Maximum tokens per chunk. Defaults to EMBED_MAX_TOKENS (env var or 500).
        overlap_tokens: Number of tokens to overlap between chunks.

    Returns:
        List of text chunks.
    """
    if max_tokens is None:
        max_tokens = EMBED_MAX_TOKENS
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


def chunk_code(text: str, max_tokens: int = None) -> list[str]:
    """
    Split code into chunks, trying to preserve function/class boundaries.

    Args:
        text: The code text to chunk.
        max_tokens: Maximum tokens per chunk. Defaults to EMBED_MAX_TOKENS (env var or 500).

    Returns:
        List of code chunks.
    """
    if max_tokens is None:
        max_tokens = EMBED_MAX_TOKENS
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

    Includes:
    - Proactive rate limiting to prevent hitting API limits
    - Automatic retry with exponential backoff for transient errors
      (connection errors, rate limits, timeouts)
    """
    # Estimate token count for rate limiting
    token_count = count_tokens(text)
    rate_limiter = get_rate_limiter()
    rate_limiter.acquire(num_tokens=token_count)

    try:
        client = get_openai_client()
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

    Includes:
    - Proactive rate limiting to prevent hitting API limits
    - Automatic retry with exponential backoff for transient errors
      (connection errors, rate limits, timeouts)
    """
    if not texts:
        return []

    # Estimate total token count for rate limiting
    total_tokens = sum(count_tokens(text) for text in texts)
    rate_limiter = get_rate_limiter()
    rate_limiter.acquire(num_tokens=total_tokens)

    try:
        client = get_openai_client()
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


def reembed_item(item_id: int) -> dict:
    """
    Re-embed an existing item by deleting old chunks and creating new embeddings.

    This preserves the item record and metadata but regenerates all embeddings.
    Useful when embedding model or chunking strategy changes.

    Returns dict with:
    - item_id: The item ID
    - title: Item title
    - type: Item type (video, article, code_file)
    - old_chunks: Number of chunks deleted
    - new_chunks: Number of new chunks created
    - status: 'success' or 'error'
    - error: Error message (if status is 'error')
    """
    # Get item info
    item = db.get_item_by_id(item_id)
    if not item:
        return {
            "item_id": item_id,
            "title": None,
            "type": None,
            "old_chunks": 0,
            "new_chunks": 0,
            "status": "error",
            "error": f"Item {item_id} not found",
        }

    # Get existing content before deleting chunks
    content = db.get_item_content(item_id)
    if not content:
        return {
            "item_id": item_id,
            "title": item["title"],
            "type": item["type"],
            "old_chunks": 0,
            "new_chunks": 0,
            "status": "error",
            "error": "No content found for item",
        }

    # Delete old chunks
    old_chunks = db.delete_chunks_for_item(item_id)

    # Determine if code or text based on item type
    is_code = item["type"] == "code_file"

    # Re-embed with new chunks
    try:
        new_chunks = embed_text_content(item_id, content, is_code=is_code)
        return {
            "item_id": item_id,
            "title": item["title"],
            "type": item["type"],
            "old_chunks": old_chunks,
            "new_chunks": new_chunks,
            "status": "success",
        }
    except Exception as e:
        return {
            "item_id": item_id,
            "title": item["title"],
            "type": item["type"],
            "old_chunks": old_chunks,
            "new_chunks": 0,
            "status": "error",
            "error": str(e),
        }


def reembed_all_items(
    progress_callback: callable = None,
) -> dict:
    """
    Re-embed all items in the database.

    Args:
        progress_callback: Optional callback(current, total, item_title, status)

    Returns dict with:
    - total_items: Number of items processed
    - successful: Number of items successfully re-embedded
    - failed: Number of items that failed
    - total_old_chunks: Total chunks deleted
    - total_new_chunks: Total new chunks created
    - errors: List of error details
    """
    items = db.list_items()
    total = len(items)
    successful = 0
    failed = 0
    total_old_chunks = 0
    total_new_chunks = 0
    errors = []

    for i, item in enumerate(items):
        result = reembed_item(item["id"])

        if result["status"] == "success":
            successful += 1
            total_old_chunks += result["old_chunks"]
            total_new_chunks += result["new_chunks"]
        else:
            failed += 1
            errors.append({
                "item_id": item["id"],
                "title": item["title"],
                "error": result.get("error", "Unknown error"),
            })

        if progress_callback:
            progress_callback(i + 1, total, item["title"], result["status"])

    return {
        "total_items": total,
        "successful": successful,
        "failed": failed,
        "total_old_chunks": total_old_chunks,
        "total_new_chunks": total_new_chunks,
        "errors": errors,
    }


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
