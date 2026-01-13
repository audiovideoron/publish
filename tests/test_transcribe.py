"""Tests for transcription functions (transcribe.py)."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from publishing import transcribe


class TestTranscribeAudio:
    """Tests for audio transcription."""

    def test_transcribe_audio_file_not_found(self, tmp_path):
        """Test transcription with non-existent file."""
        fake_path = tmp_path / "nonexistent.mp3"

        with pytest.raises(FileNotFoundError):
            transcribe.transcribe_audio(fake_path)

    def test_transcribe_audio_success(self, tmp_path):
        """Test successful audio transcription."""
        # Create a fake audio file
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"\x00" * 100)

        mock_segments = [
            MagicMock(start=0.0, end=5.0, text="Hello"),
            MagicMock(start=5.0, end=10.0, text="world"),
        ]
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_response.segments = mock_segments
        mock_response.language = "en"
        mock_response.duration = 10.0

        with patch("publishing.transcribe.get_openai_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.audio.transcriptions.create.return_value = mock_response

            result = transcribe.transcribe_audio(audio_file)

            assert result["text"] == "Hello world"
            assert len(result["segments"]) == 2
            assert result["language"] == "en"
            assert result["duration"] == 10.0

    def test_transcribe_audio_with_language(self, tmp_path):
        """Test transcription with specified language."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"\x00" * 100)

        mock_response = MagicMock()
        mock_response.text = "Bonjour"
        mock_response.segments = []
        mock_response.language = "fr"
        mock_response.duration = 5.0

        with patch("publishing.transcribe.get_openai_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.audio.transcriptions.create.return_value = mock_response

            result = transcribe.transcribe_audio(audio_file, language="fr")

            mock_client.audio.transcriptions.create.assert_called_once()
            call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
            assert call_kwargs["language"] == "fr"

    def test_transcribe_audio_auth_error(self, tmp_path):
        """Test transcription with authentication error."""
        from openai import AuthenticationError

        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"\x00" * 100)

        with patch("publishing.transcribe.get_openai_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.audio.transcriptions.create.side_effect = AuthenticationError(
                message="Invalid API key",
                response=MagicMock(status_code=401),
                body=None,
            )

            with pytest.raises(RuntimeError) as exc_info:
                transcribe.transcribe_audio(audio_file)

            assert "authentication failed" in str(exc_info.value).lower()

    def test_transcribe_audio_rate_limit(self, tmp_path):
        """Test transcription with rate limit error (retryable, will be raised after retries)."""
        from openai import RateLimitError

        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"\x00" * 100)

        with patch("publishing.transcribe.get_openai_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.audio.transcriptions.create.side_effect = RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body=None,
            )

            with pytest.raises(RateLimitError):
                transcribe.transcribe_audio(audio_file)

    def test_transcribe_audio_api_connection_error(self, tmp_path):
        """Test transcription with API connection error (retryable, will be raised after retries)."""
        from openai import APIConnectionError

        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"\x00" * 100)

        with patch("publishing.transcribe.get_openai_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.audio.transcriptions.create.side_effect = APIConnectionError(
                request=MagicMock()
            )

            with pytest.raises(APIConnectionError):
                transcribe.transcribe_audio(audio_file)

    def test_transcribe_audio_no_segments(self, tmp_path):
        """Test transcription when no segments are returned."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"\x00" * 100)

        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_response.segments = None
        mock_response.language = "en"
        mock_response.duration = 10.0

        with patch("publishing.transcribe.get_openai_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.audio.transcriptions.create.return_value = mock_response

            result = transcribe.transcribe_audio(audio_file)

            assert result["text"] == "Hello world"
            assert result["segments"] == []


class TestSegmentsToTimedChunks:
    """Tests for combining segments into timed chunks."""

    def test_segments_to_timed_chunks_empty(self):
        """Test with empty segments list."""
        result = transcribe.segments_to_timed_chunks([])
        assert result == []

    def test_segments_to_timed_chunks_single(self):
        """Test with single segment."""
        segments = [{"start": 0.0, "end": 30.0, "text": "Hello world"}]

        result = transcribe.segments_to_timed_chunks(segments, chunk_duration=60.0)

        assert len(result) == 1
        assert result[0]["text"] == "Hello world"
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 30.0

    def test_segments_to_timed_chunks_combine(self):
        """Test combining multiple segments into one chunk."""
        segments = [
            {"start": 0.0, "end": 10.0, "text": "Hello"},
            {"start": 10.0, "end": 20.0, "text": "world"},
            {"start": 20.0, "end": 30.0, "text": "today"},
        ]

        result = transcribe.segments_to_timed_chunks(segments, chunk_duration=60.0)

        assert len(result) == 1
        assert "Hello" in result[0]["text"]
        assert "world" in result[0]["text"]
        assert "today" in result[0]["text"]
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 30.0

    def test_segments_to_timed_chunks_split(self):
        """Test splitting segments into multiple chunks."""
        segments = [
            {"start": 0.0, "end": 30.0, "text": "First part"},
            {"start": 30.0, "end": 60.0, "text": "Second part"},
            {"start": 60.0, "end": 90.0, "text": "Third part"},
            {"start": 90.0, "end": 120.0, "text": "Fourth part"},
        ]

        result = transcribe.segments_to_timed_chunks(segments, chunk_duration=50.0)

        # Should create multiple chunks based on duration
        assert len(result) >= 2

        # First chunk should have correct start time
        assert result[0]["start"] == 0.0

        # Last chunk should have correct end time
        assert result[-1]["end"] == 120.0

    def test_segments_to_timed_chunks_exact_duration(self):
        """Test when segments exactly match chunk duration."""
        segments = [
            {"start": 0.0, "end": 30.0, "text": "First"},
            {"start": 30.0, "end": 60.0, "text": "Second"},
            {"start": 60.0, "end": 90.0, "text": "Third"},
        ]

        result = transcribe.segments_to_timed_chunks(segments, chunk_duration=60.0)

        assert len(result) == 2

    def test_segments_to_timed_chunks_preserves_timing(self):
        """Test that timing information is preserved correctly."""
        segments = [
            {"start": 100.0, "end": 110.0, "text": "Late start"},
            {"start": 110.0, "end": 120.0, "text": "Continues"},
        ]

        result = transcribe.segments_to_timed_chunks(segments, chunk_duration=60.0)

        assert len(result) == 1
        assert result[0]["start"] == 100.0
        assert result[0]["end"] == 120.0

    def test_segments_to_timed_chunks_concatenates_text(self):
        """Test that text is properly concatenated with spaces."""
        segments = [
            {"start": 0.0, "end": 10.0, "text": "Word1"},
            {"start": 10.0, "end": 20.0, "text": "Word2"},
            {"start": 20.0, "end": 30.0, "text": "Word3"},
        ]

        result = transcribe.segments_to_timed_chunks(segments, chunk_duration=60.0)

        assert result[0]["text"] == "Word1 Word2 Word3"

    def test_segments_to_timed_chunks_long_video(self):
        """Test with many segments simulating a long video."""
        # Create segments for a 30-minute video (1800 seconds)
        segments = [
            {"start": float(i * 10), "end": float((i + 1) * 10), "text": f"Segment {i}"}
            for i in range(180)  # 180 segments of 10 seconds each
        ]

        result = transcribe.segments_to_timed_chunks(segments, chunk_duration=60.0)

        # Should create approximately 30 chunks (1800/60)
        assert len(result) >= 25  # Allow some flexibility
        assert len(result) <= 35

        # Verify continuity
        for i in range(len(result) - 1):
            # Each chunk should end where the next one starts or overlap
            assert result[i]["end"] <= result[i + 1]["end"]
