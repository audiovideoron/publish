"""Transcription using OpenAI Whisper API."""

import logging
import os
from pathlib import Path

from openai import OpenAI, APIError, APIConnectionError, RateLimitError, AuthenticationError, APITimeoutError
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

load_dotenv()

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

# Lazy-initialized OpenAI client
_openai_client = None

# Default chunk duration for timed chunks (configurable via environment variable)
DEFAULT_CHUNK_DURATION = float(os.getenv("TRANSCRIBE_CHUNK_DURATION", "60.0"))


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
                "Please set it in your .env file or environment to use transcription features. "
                "You can get an API key from https://platform.openai.com/api-keys"
            )
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


@openai_retry
def transcribe_audio(audio_path: str | Path, language: str | None = None) -> dict:
    """
    Transcribe audio file using OpenAI Whisper API.
    Returns dict with text and segments (with timestamps).

    Includes automatic retry with exponential backoff for transient errors
    (connection errors, rate limits, timeouts).
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with open(audio_path, "rb") as audio_file:
        # Use verbose_json to get word-level timestamps
        try:
            client = get_openai_client()
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                language=language,
                timestamp_granularities=["segment"],
            )
        except OPENAI_RETRY_EXCEPTIONS:
            # Let tenacity handle these via the decorator
            raise
        except AuthenticationError as e:
            raise RuntimeError(f"OpenAI authentication failed. Check your API key: {e}") from e
        except APIError as e:
            raise RuntimeError(f"OpenAI API error during transcription: {e}") from e

    # Extract segments with timestamps
    segments = []
    if hasattr(response, "segments") and response.segments:
        for seg in response.segments:
            segments.append({
                "start": getattr(seg, "start", 0),
                "end": getattr(seg, "end", 0),
                "text": getattr(seg, "text", "").strip(),
            })

    return {
        "text": response.text,
        "segments": segments,
        "language": getattr(response, "language", None),
        "duration": getattr(response, "duration", None),
    }


def segments_to_timed_chunks(
    segments: list[dict],
    chunk_duration: float | None = None,
) -> list[dict]:
    """
    Combine segments into larger chunks of approximately chunk_duration seconds.
    Preserves timestamp information.

    Args:
        segments: List of segment dictionaries with 'start', 'end', and 'text' keys.
        chunk_duration: Duration in seconds for each chunk. If None, uses
            DEFAULT_CHUNK_DURATION (configurable via TRANSCRIBE_CHUNK_DURATION env var,
            defaults to 60.0 seconds).
    """
    if chunk_duration is None:
        chunk_duration = DEFAULT_CHUNK_DURATION

    if not segments:
        return []

    chunks = []
    current_chunk = {
        "text": "",
        "start": segments[0]["start"],
        "end": segments[0]["end"],
    }

    for seg in segments:
        # Check if adding this segment would exceed chunk duration
        if seg["end"] - current_chunk["start"] > chunk_duration and current_chunk["text"]:
            chunks.append(current_chunk)
            current_chunk = {
                "text": seg["text"],
                "start": seg["start"],
                "end": seg["end"],
            }
        else:
            # Add to current chunk
            if current_chunk["text"]:
                current_chunk["text"] += " " + seg["text"]
            else:
                current_chunk["text"] = seg["text"]
            current_chunk["end"] = seg["end"]

    # Don't forget the last chunk
    if current_chunk["text"]:
        chunks.append(current_chunk)

    return chunks
