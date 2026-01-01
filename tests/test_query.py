"""Tests for query functions (query.py)."""

import pytest
from unittest.mock import MagicMock, patch

from distillyzer import query


class TestFormatTimestamp:
    """Tests for timestamp formatting."""

    def test_format_timestamp_none(self):
        """Test formatting None timestamp."""
        result = query.format_timestamp(None)
        assert result == ""

    def test_format_timestamp_zero(self):
        """Test formatting zero seconds."""
        result = query.format_timestamp(0)
        assert result == "0:00"

    def test_format_timestamp_minutes_only(self):
        """Test formatting time with only minutes."""
        result = query.format_timestamp(125)  # 2:05
        assert result == "2:05"

    def test_format_timestamp_with_hours(self):
        """Test formatting time with hours."""
        result = query.format_timestamp(3665)  # 1:01:05
        assert result == "1:01:05"

    def test_format_timestamp_exact_hour(self):
        """Test formatting exactly one hour."""
        result = query.format_timestamp(3600)
        assert result == "1:00:00"

    def test_format_timestamp_float(self):
        """Test formatting float seconds."""
        result = query.format_timestamp(65.7)  # 1:05
        assert result == "1:05"


class TestFormatContext:
    """Tests for context formatting."""

    def test_format_context_empty(self):
        """Test formatting empty chunks."""
        result = query.format_context([])
        assert "No relevant content" in result

    def test_format_context_single_chunk(self, sample_chunk):
        """Test formatting single chunk."""
        result = query.format_context([sample_chunk])

        assert "Source 1" in result
        assert sample_chunk["item_title"] in result
        assert sample_chunk["content"] in result

    def test_format_context_multiple_chunks(self, sample_chunks):
        """Test formatting multiple chunks."""
        result = query.format_context(sample_chunks)

        assert "Source 1" in result
        assert "Source 2" in result
        for chunk in sample_chunks:
            assert chunk["content"] in result

    def test_format_context_with_timestamps(self, sample_chunk):
        """Test formatting chunk with timestamp."""
        result = query.format_context([sample_chunk])

        # Should include formatted timestamp
        assert "@ 0:00" in result

    def test_format_context_with_url(self, sample_chunk):
        """Test formatting chunk with URL."""
        result = query.format_context([sample_chunk])

        assert sample_chunk["item_url"] in result

    def test_format_context_no_timestamp(self):
        """Test formatting chunk without timestamp."""
        chunk = {
            "chunk_id": 1,
            "content": "Test content",
            "chunk_index": 0,
            "timestamp_start": None,
            "timestamp_end": None,
            "item_title": "Test",
            "item_url": "https://example.com",
            "item_type": "article",
            "similarity": 0.9,
        }
        result = query.format_context([chunk])

        # Should not include @ timestamp
        assert "@ " not in result or "@" in chunk["item_url"]


class TestSearch:
    """Tests for semantic search."""

    def test_search_success(self, sample_chunks):
        """Test successful search."""
        with patch("distillyzer.query.get_embedding") as mock_embed, \
             patch("distillyzer.query.db") as mock_db:

            mock_embed.return_value = [0.1] * 1536
            mock_db.search_chunks.return_value = sample_chunks

            result = query.search("machine learning", limit=5)

            assert len(result) == 2
            mock_embed.assert_called_once_with("machine learning")
            mock_db.search_chunks.assert_called_once()

    def test_search_empty_results(self):
        """Test search with no results."""
        with patch("distillyzer.query.get_embedding") as mock_embed, \
             patch("distillyzer.query.db") as mock_db:

            mock_embed.return_value = [0.1] * 1536
            mock_db.search_chunks.return_value = []

            result = query.search("obscure topic")

            assert result == []


class TestAsk:
    """Tests for RAG-based question answering."""

    def test_ask_success(self, sample_chunks):
        """Test successful question answering."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Machine learning is a subset of AI.")]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

        with patch("distillyzer.query.search") as mock_search, \
             patch("distillyzer.query.claude") as mock_claude:

            mock_search.return_value = sample_chunks
            mock_claude.messages.create.return_value = mock_response

            result = query.ask("What is machine learning?")

            assert "answer" in result
            assert "sources" in result
            assert "tokens_used" in result
            assert result["tokens_used"] == 150

    def test_ask_no_context(self):
        """Test asking when no context found."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="I don't have information on that.")]
        mock_response.usage = MagicMock(input_tokens=50, output_tokens=20)

        with patch("distillyzer.query.search") as mock_search, \
             patch("distillyzer.query.claude") as mock_claude:

            mock_search.return_value = []
            mock_claude.messages.create.return_value = mock_response

            result = query.ask("Unknown topic?")

            assert "answer" in result
            assert result["sources"] == []

    def test_ask_api_connection_error(self, sample_chunks):
        """Test asking when API connection fails."""
        import anthropic

        with patch("distillyzer.query.search") as mock_search, \
             patch("distillyzer.query.claude") as mock_claude:

            mock_search.return_value = sample_chunks
            mock_claude.messages.create.side_effect = anthropic.APIConnectionError(
                request=MagicMock()
            )

            with pytest.raises(RuntimeError) as exc_info:
                query.ask("What is AI?")

            assert "connect" in str(exc_info.value).lower()

    def test_ask_rate_limit_error(self, sample_chunks):
        """Test asking when rate limited."""
        import anthropic

        with patch("distillyzer.query.search") as mock_search, \
             patch("distillyzer.query.claude") as mock_claude:

            mock_search.return_value = sample_chunks
            mock_claude.messages.create.side_effect = anthropic.RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body=None,
            )

            with pytest.raises(RuntimeError) as exc_info:
                query.ask("What is AI?")

            assert "rate limit" in str(exc_info.value).lower()

    def test_ask_api_status_error(self, sample_chunks):
        """Test asking when API returns error status."""
        import anthropic

        with patch("distillyzer.query.search") as mock_search, \
             patch("distillyzer.query.claude") as mock_claude:

            mock_search.return_value = sample_chunks
            mock_claude.messages.create.side_effect = anthropic.APIStatusError(
                message="Server error",
                response=MagicMock(status_code=500),
                body=None,
            )

            with pytest.raises(RuntimeError) as exc_info:
                query.ask("What is AI?")

            assert "500" in str(exc_info.value)


class TestChatTurn:
    """Tests for chat conversation turns."""

    def test_chat_turn_first_message(self, sample_chunks):
        """Test first message in chat (no history)."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello! How can I help?")]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

        with patch("distillyzer.query.search") as mock_search, \
             patch("distillyzer.query.claude") as mock_claude:

            mock_search.return_value = sample_chunks
            mock_claude.messages.create.return_value = mock_response

            result = query.chat_turn("Hello", history=[])

            assert "answer" in result
            call_args = mock_claude.messages.create.call_args
            assert len(call_args[1]["messages"]) == 1  # Just the current message

    def test_chat_turn_with_history(self, sample_chunks):
        """Test chat turn with conversation history."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Based on our previous discussion...")]
        mock_response.usage = MagicMock(input_tokens=200, output_tokens=100)

        history = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI stands for Artificial Intelligence."},
        ]

        with patch("distillyzer.query.search") as mock_search, \
             patch("distillyzer.query.claude") as mock_claude:

            mock_search.return_value = sample_chunks
            mock_claude.messages.create.return_value = mock_response

            result = query.chat_turn("Tell me more", history=history)

            assert "answer" in result
            call_args = mock_claude.messages.create.call_args
            # Should have history + current message
            assert len(call_args[1]["messages"]) == 3

    def test_chat_turn_preserves_history_roles(self, sample_chunks):
        """Test that chat turn preserves history roles correctly."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response")]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

        history = [
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
            {"role": "assistant", "content": "Answer 2"},
        ]

        with patch("distillyzer.query.search") as mock_search, \
             patch("distillyzer.query.claude") as mock_claude:

            mock_search.return_value = sample_chunks
            mock_claude.messages.create.return_value = mock_response

            query.chat_turn("Question 3", history=history)

            call_args = mock_claude.messages.create.call_args
            messages = call_args[1]["messages"]

            # Verify role alternation
            for i, msg in enumerate(messages[:-1]):  # Exclude last (current) message
                expected_role = "user" if i % 2 == 0 else "assistant"
                assert msg["role"] == expected_role

    def test_chat_turn_api_error(self, sample_chunks):
        """Test chat turn with API error."""
        import anthropic

        with patch("distillyzer.query.search") as mock_search, \
             patch("distillyzer.query.claude") as mock_claude:

            mock_search.return_value = sample_chunks
            mock_claude.messages.create.side_effect = anthropic.APIConnectionError(
                request=MagicMock()
            )

            with pytest.raises(RuntimeError):
                query.chat_turn("Hello", history=[])

    def test_chat_turn_includes_sources(self, sample_chunks):
        """Test that chat turn includes sources in response."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response")]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

        with patch("distillyzer.query.search") as mock_search, \
             patch("distillyzer.query.claude") as mock_claude:

            mock_search.return_value = sample_chunks
            mock_claude.messages.create.return_value = mock_response

            result = query.chat_turn("Question", history=[])

            assert "sources" in result
            assert len(result["sources"]) == len(sample_chunks)
