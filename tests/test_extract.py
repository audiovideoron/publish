"""Tests for artifact extraction functions (extract.py)."""

import pytest
import json
from unittest.mock import MagicMock, patch

from distillyzer import extract


class TestSearchContext:
    """Tests for context search."""

    def test_search_context_success(self, sample_chunks):
        """Test successful context search."""
        with patch("distillyzer.extract.get_embedding") as mock_embed, \
             patch("distillyzer.extract.db") as mock_db:

            mock_embed.return_value = [0.1] * 1536
            mock_db.search_chunks.return_value = sample_chunks

            context, chunks = extract.search_context("machine learning", num_chunks=5)

            assert context != ""
            assert len(chunks) == 2
            # Context should contain source info
            assert "ML Basics" in context

    def test_search_context_empty(self):
        """Test context search with no results."""
        with patch("distillyzer.extract.get_embedding") as mock_embed, \
             patch("distillyzer.extract.db") as mock_db:

            mock_embed.return_value = [0.1] * 1536
            mock_db.search_chunks.return_value = []

            context, chunks = extract.search_context("obscure topic")

            assert context == ""
            assert chunks == []

    def test_search_context_with_timestamps(self):
        """Test context search formats timestamps correctly."""
        chunks = [{
            "chunk_id": 1,
            "content": "Test content",
            "chunk_index": 0,
            "timestamp_start": 125.0,  # 2:05
            "timestamp_end": 185.0,
            "item_title": "Test Video",
            "item_url": "https://yt.com/v",
            "item_type": "video",
            "similarity": 0.9,
        }]

        with patch("distillyzer.extract.get_embedding") as mock_embed, \
             patch("distillyzer.extract.db") as mock_db:

            mock_embed.return_value = [0.1] * 1536
            mock_db.search_chunks.return_value = chunks

            context, _ = extract.search_context("test")

            # Should format as MM:SS
            assert "2:05" in context


class TestExtractArtifacts:
    """Tests for artifact extraction from topic."""

    def test_extract_artifacts_no_content(self):
        """Test extraction when no content found."""
        with patch("distillyzer.extract.search_context") as mock_search:
            mock_search.return_value = ("", [])

            result = extract.extract_artifacts("obscure topic")

            assert result["status"] == "no_content"
            assert result["artifacts"] == []

    def test_extract_artifacts_success(self, sample_chunks):
        """Test successful artifact extraction."""
        mock_artifacts = {
            "artifacts": [
                {
                    "type": "pattern",
                    "name": "Chain of Thought",
                    "content": "Use step-by-step reasoning",
                    "context": "When solving complex problems",
                    "source": "ML Basics",
                }
            ],
            "notes": "Good patterns found",
        }

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(mock_artifacts))]
        mock_response.usage = MagicMock(input_tokens=500, output_tokens=200)

        with patch("distillyzer.extract.search_context") as mock_search, \
             patch("distillyzer.extract.claude") as mock_claude:

            mock_search.return_value = ("Context content", sample_chunks)
            mock_claude.messages.create.return_value = mock_response

            result = extract.extract_artifacts("prompting patterns")

            assert result["status"] == "success"
            assert len(result["artifacts"]) == 1
            assert result["artifacts"][0]["type"] == "pattern"
            assert result["tokens_used"] == 700

    def test_extract_artifacts_with_type_filter(self, sample_chunks):
        """Test extraction with specific artifact type."""
        mock_artifacts = {"artifacts": [], "notes": "No prompts found"}

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(mock_artifacts))]
        mock_response.usage = MagicMock(input_tokens=300, output_tokens=50)

        with patch("distillyzer.extract.search_context") as mock_search, \
             patch("distillyzer.extract.claude") as mock_claude:

            mock_search.return_value = ("Context", sample_chunks)
            mock_claude.messages.create.return_value = mock_response

            result = extract.extract_artifacts("topic", artifact_type="prompt")

            # Verify that the system prompt included only prompt instructions
            call_args = mock_claude.messages.create.call_args
            system_prompt = call_args[1]["system"]
            assert "PROMPT TEMPLATES" in system_prompt

    def test_extract_artifacts_json_in_code_block(self, sample_chunks):
        """Test extraction when JSON is in code block."""
        mock_artifacts = {"artifacts": [{"type": "rule", "name": "Test"}], "notes": ""}

        response_text = f"```json\n{json.dumps(mock_artifacts)}\n```"

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=response_text)]
        mock_response.usage = MagicMock(input_tokens=300, output_tokens=100)

        with patch("distillyzer.extract.search_context") as mock_search, \
             patch("distillyzer.extract.claude") as mock_claude:

            mock_search.return_value = ("Context", sample_chunks)
            mock_claude.messages.create.return_value = mock_response

            result = extract.extract_artifacts("topic")

            assert result["status"] == "success"
            assert len(result["artifacts"]) == 1

    def test_extract_artifacts_invalid_json(self, sample_chunks):
        """Test extraction when response is not valid JSON."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="This is not JSON")]
        mock_response.usage = MagicMock(input_tokens=300, output_tokens=50)

        with patch("distillyzer.extract.search_context") as mock_search, \
             patch("distillyzer.extract.claude") as mock_claude:

            mock_search.return_value = ("Context", sample_chunks)
            mock_claude.messages.create.return_value = mock_response

            result = extract.extract_artifacts("topic")

            # Should return raw response as artifact
            assert len(result["artifacts"]) == 1
            assert result["artifacts"][0]["type"] == "raw"

    def test_extract_artifacts_api_connection_error(self, sample_chunks):
        """Test extraction with API connection error."""
        import anthropic

        with patch("distillyzer.extract.search_context") as mock_search, \
             patch("distillyzer.extract.claude") as mock_claude:

            mock_search.return_value = ("Context", sample_chunks)
            mock_claude.messages.create.side_effect = anthropic.APIConnectionError(
                request=MagicMock()
            )

            result = extract.extract_artifacts("topic")

            assert result["status"] == "api_error"
            assert "connect" in result["message"].lower()

    def test_extract_artifacts_rate_limit(self, sample_chunks):
        """Test extraction with rate limit error."""
        import anthropic

        with patch("distillyzer.extract.search_context") as mock_search, \
             patch("distillyzer.extract.claude") as mock_claude:

            mock_search.return_value = ("Context", sample_chunks)
            mock_claude.messages.create.side_effect = anthropic.RateLimitError(
                message="Rate limit",
                response=MagicMock(status_code=429),
                body=None,
            )

            result = extract.extract_artifacts("topic")

            assert result["status"] == "rate_limit"


class TestExtractFromItem:
    """Tests for extraction from specific item."""

    def test_extract_from_item_not_found(self):
        """Test extraction from non-existent item."""
        with patch("distillyzer.extract.db") as mock_db:
            mock_db.get_items_with_chunks.return_value = []

            result = extract.extract_from_item(999)

            assert result["status"] == "not_found"

    def test_extract_from_item_no_chunks(self):
        """Test extraction from item with no chunks."""
        with patch("distillyzer.extract.db") as mock_db:
            mock_db.get_items_with_chunks.return_value = [{
                "id": 1,
                "title": "Empty Video",
                "chunks": [],
            }]

            result = extract.extract_from_item(1)

            assert result["status"] == "no_chunks"

    def test_extract_from_item_success(self):
        """Test successful extraction from item."""
        mock_item = {
            "id": 1,
            "title": "Test Video",
            "chunks": [
                {"content": "Chunk 1 content"},
                {"content": "Chunk 2 content"},
            ],
        }

        mock_artifacts = {
            "artifacts": [{"type": "pattern", "name": "Test Pattern"}],
            "notes": "",
        }

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(mock_artifacts))]
        mock_response.usage = MagicMock(input_tokens=400, output_tokens=150)

        with patch("distillyzer.extract.db") as mock_db, \
             patch("distillyzer.extract.claude") as mock_claude:

            mock_db.get_items_with_chunks.return_value = [mock_item]
            mock_claude.messages.create.return_value = mock_response

            result = extract.extract_from_item(1)

            assert result["status"] == "success"
            assert result["item_id"] == 1
            assert result["item_title"] == "Test Video"

    def test_extract_from_item_api_error(self):
        """Test extraction with API error."""
        import anthropic

        mock_item = {
            "id": 1,
            "title": "Test Video",
            "chunks": [{"content": "Content"}],
        }

        with patch("distillyzer.extract.db") as mock_db, \
             patch("distillyzer.extract.claude") as mock_claude:

            mock_db.get_items_with_chunks.return_value = [mock_item]
            mock_claude.messages.create.side_effect = anthropic.APIConnectionError(
                request=MagicMock()
            )

            result = extract.extract_from_item(1)

            assert result["status"] == "api_error"
            assert result["item_id"] == 1


class TestExtractionPrompts:
    """Tests for extraction prompts configuration."""

    def test_extraction_prompts_exist(self):
        """Test that all extraction prompts are defined."""
        expected_types = ["prompt", "pattern", "checklist", "rule", "tool"]

        for artifact_type in expected_types:
            assert artifact_type in extract.EXTRACTION_PROMPTS
            assert len(extract.EXTRACTION_PROMPTS[artifact_type]) > 0

    def test_extraction_prompts_content(self):
        """Test extraction prompts contain useful instructions."""
        for artifact_type, prompt in extract.EXTRACTION_PROMPTS.items():
            # Each prompt should mention what to look for
            assert "Look for" in prompt or "Extract" in prompt.lower()
            # Each prompt should mention what to provide
            assert "provide" in prompt.lower() or "return" in prompt.lower()


class TestArtifactType:
    """Tests for artifact type handling."""

    def test_artifact_type_literal(self):
        """Test that ArtifactType includes expected values."""
        # This tests the type definition indirectly
        valid_types = ["prompt", "pattern", "checklist", "rule", "tool", "all"]

        for t in valid_types:
            # Should not raise when used in extract_artifacts
            with patch("distillyzer.extract.search_context") as mock_search:
                mock_search.return_value = ("", [])
                result = extract.extract_artifacts("topic", artifact_type=t)
                assert result is not None
