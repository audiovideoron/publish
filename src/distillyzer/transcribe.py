"""Transcription using OpenAI Whisper API."""

import os
from pathlib import Path

from openai import OpenAI, APIError, APIConnectionError, RateLimitError, AuthenticationError
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def transcribe_audio(audio_path: str | Path, language: str | None = None) -> dict:
    """
    Transcribe audio file using OpenAI Whisper API.
    Returns dict with text and segments (with timestamps).
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with open(audio_path, "rb") as audio_file:
        # Use verbose_json to get word-level timestamps
        try:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                language=language,
                timestamp_granularities=["segment"],
            )
        except AuthenticationError as e:
            raise RuntimeError(f"OpenAI authentication failed. Check your API key: {e}") from e
        except RateLimitError as e:
            raise RuntimeError(f"OpenAI rate limit exceeded. Please try again later: {e}") from e
        except APIConnectionError as e:
            raise RuntimeError(f"Failed to connect to OpenAI API: {e}") from e
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
    chunk_duration: float = 60.0,
) -> list[dict]:
    """
    Combine segments into larger chunks of approximately chunk_duration seconds.
    Preserves timestamp information.
    """
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
