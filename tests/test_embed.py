"""Tests for embedding and chunking functions (embed.py)."""

import time
import pytest
from unittest.mock import MagicMock, patch

from publishing import embed


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_rate_limiter_init(self):
        """Test RateLimiter initialization."""
        limiter = embed.RateLimiter(requests_per_minute=100, tokens_per_minute=50000)
        assert limiter.rpm_limit == 100
        assert limiter.tpm_limit == 50000

    def test_rate_limiter_acquire_no_wait(self):
        """Test acquiring when bucket is full (no wait needed)."""
        limiter = embed.RateLimiter(requests_per_minute=100, tokens_per_minute=50000)

        start = time.monotonic()
        wait_time = limiter.acquire(num_tokens=100)
        elapsed = time.monotonic() - start

        # Should not wait when bucket is full
        assert wait_time < 0.1
        assert elapsed < 0.1

    def test_rate_limiter_acquire_waits_on_rpm(self):
        """Test that acquire waits when request bucket is exhausted."""
        # Very low RPM to trigger waiting quickly
        limiter = embed.RateLimiter(requests_per_minute=60, tokens_per_minute=1000000)

        # Exhaust the request bucket
        for _ in range(60):
            limiter.acquire(num_tokens=0)

        # Next acquire should need to wait
        start = time.monotonic()
        wait_time = limiter.acquire(num_tokens=0)
        elapsed = time.monotonic() - start

        # Should wait approximately 1 second (refill rate is 1 per second)
        assert wait_time > 0.5
        assert elapsed > 0.5

    def test_rate_limiter_acquire_waits_on_tpm(self):
        """Test that acquire waits when token bucket is exhausted."""
        # Very low TPM to trigger waiting quickly
        limiter = embed.RateLimiter(requests_per_minute=1000, tokens_per_minute=60)

        # Exhaust the token bucket
        limiter.acquire(num_tokens=60)

        # Next acquire should need to wait
        start = time.monotonic()
        wait_time = limiter.acquire(num_tokens=10)
        elapsed = time.monotonic() - start

        # Should wait for token refill
        assert wait_time > 0.1
        assert elapsed > 0.1

    def test_rate_limiter_refills_over_time(self):
        """Test that buckets refill over time."""
        limiter = embed.RateLimiter(requests_per_minute=60, tokens_per_minute=60)

        # Exhaust both buckets
        limiter.acquire(num_tokens=60)

        # Wait for refill
        time.sleep(1.1)

        # Should be able to acquire without waiting (or minimal wait)
        start = time.monotonic()
        wait_time = limiter.acquire(num_tokens=1)
        elapsed = time.monotonic() - start

        # Should not wait significantly since buckets refilled
        assert wait_time < 0.5
        assert elapsed < 0.5


class TestRateLimiterFunctions:
    """Tests for rate limiter helper functions."""

    def test_get_rate_limiter_creates_instance(self):
        """Test that get_rate_limiter creates a rate limiter."""
        # Reset global state
        embed._rate_limiter = None

        limiter = embed.get_rate_limiter()
        assert limiter is not None
        assert isinstance(limiter, embed.RateLimiter)

    def test_get_rate_limiter_returns_same_instance(self):
        """Test that get_rate_limiter returns the same instance."""
        embed._rate_limiter = None
        limiter1 = embed.get_rate_limiter()
        limiter2 = embed.get_rate_limiter()
        assert limiter1 is limiter2

    def test_reset_rate_limiter(self):
        """Test resetting the rate limiter with new limits."""
        embed._rate_limiter = None
        old_limiter = embed.get_rate_limiter()

        new_limiter = embed.reset_rate_limiter(
            requests_per_minute=200,
            tokens_per_minute=100000,
        )

        assert new_limiter is not old_limiter
        assert new_limiter.rpm_limit == 200
        assert new_limiter.tpm_limit == 100000
        assert embed.get_rate_limiter() is new_limiter

    def test_get_rate_limiter_from_env(self):
        """Test that rate limiter respects environment variables."""
        embed._rate_limiter = None

        with patch.dict("os.environ", {"OPENAI_RPM_LIMIT": "150", "OPENAI_TPM_LIMIT": "75000"}):
            # Need to reset to pick up new env vars
            embed._rate_limiter = None
            limiter = embed.get_rate_limiter()
            assert limiter.rpm_limit == 150
            assert limiter.tpm_limit == 75000

        # Clean up
        embed._rate_limiter = None


class TestCountTokens:
    """Tests for token counting."""

    def test_count_tokens_empty(self):
        """Test counting tokens in empty string."""
        result = embed.count_tokens("")
        assert result == 0

    def test_count_tokens_simple(self):
        """Test counting tokens in simple text."""
        result = embed.count_tokens("Hello world")
        assert result > 0
        assert result < 10  # Simple phrase should be few tokens

    def test_count_tokens_longer(self):
        """Test counting tokens in longer text."""
        text = "This is a longer piece of text that should have more tokens than the simple example."
        result = embed.count_tokens(text)
        assert result > 10


class TestChunkText:
    """Tests for text chunking."""

    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        result = embed.chunk_text("")
        assert result == []

    def test_chunk_text_whitespace(self):
        """Test chunking whitespace-only text."""
        result = embed.chunk_text("   \n\t   ")
        assert result == []

    def test_chunk_text_small(self):
        """Test chunking text smaller than max_tokens."""
        text = "This is a short sentence."
        result = embed.chunk_text(text, max_tokens=500)
        assert len(result) == 1
        assert text in result[0]

    def test_chunk_text_large(self):
        """Test chunking large text into multiple chunks."""
        # Create text with multiple sentences
        sentences = ["This is sentence number {}.".format(i) for i in range(100)]
        text = " ".join(sentences)

        result = embed.chunk_text(text, max_tokens=50, overlap_tokens=10)

        assert len(result) > 1
        # Verify all chunks are non-empty
        for chunk in result:
            assert len(chunk.strip()) > 0

    def test_chunk_text_preserves_sentences(self):
        """Test that chunking tries to preserve sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = embed.chunk_text(text, max_tokens=20)

        # Each chunk should end with a period (sentence boundary)
        for chunk in result:
            assert chunk.strip().endswith(".")


class TestChunkCode:
    """Tests for code chunking."""

    def test_chunk_code_empty(self):
        """Test chunking empty code."""
        result = embed.chunk_code("")
        assert result == []

    def test_chunk_code_small(self):
        """Test chunking small code."""
        code = "def hello():\n    print('Hello')"
        result = embed.chunk_code(code, max_tokens=500)
        assert len(result) == 1

    def test_chunk_code_large(self):
        """Test chunking large code into multiple chunks."""
        # Create code with multiple functions
        functions = [f"def function_{i}():\n    return {i}\n\n" for i in range(50)]
        code = "".join(functions)

        result = embed.chunk_code(code, max_tokens=50)

        assert len(result) > 1

    def test_chunk_code_preserves_blocks(self):
        """Test that code chunking tries to preserve block boundaries."""
        code = """def first():
    return 1

def second():
    return 2

def third():
    return 3"""

        result = embed.chunk_code(code, max_tokens=100)

        # Verify chunks contain complete functions where possible
        for chunk in result:
            assert chunk.strip()  # Non-empty


class TestGetEmbedding:
    """Tests for single embedding generation."""

    def test_get_embedding_success(self):
        """Test successful embedding generation."""
        mock_embedding = [0.1] * 1536

        with patch("publishing.embed.get_openai_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=mock_embedding)]
            mock_client.embeddings.create.return_value = mock_response

            result = embed.get_embedding("Test text")

            assert result == mock_embedding
            mock_client.embeddings.create.assert_called_once()

    def test_get_embedding_api_error(self):
        """Test embedding generation with API error."""
        from openai import OpenAIError

        with patch("publishing.embed.get_openai_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.embeddings.create.side_effect = OpenAIError("API Error")

            with pytest.raises(RuntimeError) as exc_info:
                embed.get_embedding("Test text")

            assert "OpenAI embedding API error" in str(exc_info.value)


class TestGetEmbeddingsBatch:
    """Tests for batch embedding generation."""

    def test_get_embeddings_batch_empty(self):
        """Test batch embedding with empty list."""
        result = embed.get_embeddings_batch([])
        assert result == []

    def test_get_embeddings_batch_success(self):
        """Test successful batch embedding generation."""
        texts = ["Text 1", "Text 2", "Text 3"]
        mock_embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]

        with patch("publishing.embed.get_openai_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=e) for e in mock_embeddings]
            mock_client.embeddings.create.return_value = mock_response

            result = embed.get_embeddings_batch(texts)

            assert len(result) == 3
            assert result == mock_embeddings

    def test_get_embeddings_batch_api_error(self):
        """Test batch embedding with API error."""
        from openai import OpenAIError

        with patch("publishing.embed.get_openai_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.embeddings.create.side_effect = OpenAIError("Batch API Error")

            with pytest.raises(RuntimeError) as exc_info:
                embed.get_embeddings_batch(["Text 1", "Text 2"])

            assert "batch embedding API error" in str(exc_info.value)


class TestEmbedTranscriptChunks:
    """Tests for embedding transcript chunks."""

    def test_embed_transcript_chunks_empty(self):
        """Test embedding empty transcript chunks."""
        result = embed.embed_transcript_chunks(item_id=1, timed_chunks=[])
        assert result == 0

    def test_embed_transcript_chunks_success(self):
        """Test successful transcript chunk embedding."""
        timed_chunks = [
            {"text": "Hello world", "start": 0.0, "end": 10.0},
            {"text": "This is a test", "start": 10.0, "end": 20.0},
        ]
        mock_embeddings = [[0.1] * 1536, [0.2] * 1536]

        with patch("publishing.embed.get_embeddings_batch") as mock_batch, \
             patch("publishing.embed.db") as mock_db:
            mock_batch.return_value = mock_embeddings
            mock_db.create_chunks_batch.return_value = [1, 2]

            result = embed.embed_transcript_chunks(item_id=1, timed_chunks=timed_chunks)

            assert result == 2
            mock_db.create_chunks_batch.assert_called_once()


class TestEmbedTextContent:
    """Tests for embedding text content."""

    def test_embed_text_content_empty(self):
        """Test embedding empty text."""
        with patch("publishing.embed.chunk_text") as mock_chunk:
            mock_chunk.return_value = []

            result = embed.embed_text_content(item_id=1, text="")

            assert result == 0

    def test_embed_text_content_text(self):
        """Test embedding regular text content."""
        with patch("publishing.embed.chunk_text") as mock_chunk_text, \
             patch("publishing.embed.get_embeddings_batch") as mock_batch, \
             patch("publishing.embed.db") as mock_db:
            mock_chunk_text.return_value = ["Chunk 1", "Chunk 2"]
            mock_batch.return_value = [[0.1] * 1536, [0.2] * 1536]
            mock_db.create_chunks_batch.return_value = [1, 2]

            result = embed.embed_text_content(item_id=1, text="Some text", is_code=False)

            assert result == 2
            mock_chunk_text.assert_called_once()

    def test_embed_text_content_code(self):
        """Test embedding code content."""
        with patch("publishing.embed.chunk_code") as mock_chunk_code, \
             patch("publishing.embed.get_embeddings_batch") as mock_batch, \
             patch("publishing.embed.db") as mock_db:
            mock_chunk_code.return_value = ["def foo(): pass"]
            mock_batch.return_value = [[0.1] * 1536]
            mock_db.create_chunks_batch.return_value = [1]

            result = embed.embed_text_content(item_id=1, text="def foo(): pass", is_code=True)

            assert result == 1
            mock_chunk_code.assert_called_once()


class TestEmbedProjectFacets:
    """Tests for embedding project facets."""

    def test_embed_project_facets_not_found(self):
        """Test embedding facets for non-existent project."""
        with patch("publishing.embed.db") as mock_db:
            mock_db.get_project_by_id.return_value = None

            with pytest.raises(ValueError) as exc_info:
                embed.embed_project_facets(project_id=999)

            assert "not found" in str(exc_info.value)

    def test_embed_project_facets_success(self):
        """Test successful project facet embedding."""
        project = {
            "id": 1,
            "name": "test",
            "facet_about": ["AI", "ML"],
            "facet_uses": ["Python"],
            "facet_needs": ["GPU"],
        }

        with patch("publishing.embed.db") as mock_db, \
             patch("publishing.embed.get_embedding") as mock_embed:
            mock_db.get_project_by_id.return_value = project
            mock_embed.return_value = [0.1] * 1536
            mock_db.update_project_embeddings.return_value = True

            result = embed.embed_project_facets(project_id=1)

            assert result["about"] is True
            assert result["uses"] is True
            assert result["needs"] is True
            assert mock_embed.call_count == 3

    def test_embed_project_facets_partial(self):
        """Test embedding only some facets when others are empty."""
        project = {
            "id": 1,
            "name": "test",
            "facet_about": ["AI"],
            "facet_uses": [],  # Empty
            "facet_needs": None,  # None
        }

        with patch("publishing.embed.db") as mock_db, \
             patch("publishing.embed.get_embedding") as mock_embed:
            mock_db.get_project_by_id.return_value = project
            mock_embed.return_value = [0.1] * 1536
            mock_db.update_project_embeddings.return_value = True

            result = embed.embed_project_facets(project_id=1)

            assert result["about"] is True
            assert result["uses"] is False
            assert result["needs"] is False
            assert mock_embed.call_count == 1


class TestEmbedRepoFiles:
    """Tests for embedding repository files."""

    def test_embed_repo_files_empty(self):
        """Test embedding empty file list."""
        result = embed.embed_repo_files(file_items=[])
        assert result["total_files"] == 0
        assert result["total_chunks"] == 0
        assert result["errors"] == []

    def test_embed_repo_files_success(self):
        """Test successful file embedding."""
        file_items = [
            {"item_id": 1, "path": "main.py", "content": "print('hello')", "extension": ".py"},
            {"item_id": 2, "path": "README.md", "content": "# Readme", "extension": ".md"},
        ]

        with patch("publishing.embed.embed_text_content") as mock_embed:
            mock_embed.return_value = 2  # 2 chunks per file

            result = embed.embed_repo_files(file_items=file_items)

            assert result["total_files"] == 2
            assert result["total_chunks"] == 4
            assert result["errors"] == []

    def test_embed_repo_files_with_errors(self):
        """Test file embedding with some errors."""
        file_items = [
            {"item_id": 1, "path": "good.py", "content": "print('hello')", "extension": ".py"},
            {"item_id": 2, "path": "bad.py", "content": "error", "extension": ".py"},
        ]

        with patch("publishing.embed.embed_text_content") as mock_embed:
            mock_embed.side_effect = [2, Exception("Embedding failed")]

            result = embed.embed_repo_files(file_items=file_items)

            assert result["total_files"] == 2
            assert result["total_chunks"] == 2
            assert len(result["errors"]) == 1
            assert result["errors"][0]["path"] == "bad.py"

    def test_embed_repo_files_with_callback(self):
        """Test file embedding with progress callback."""
        file_items = [
            {"item_id": 1, "path": "file1.py", "content": "code", "extension": ".py"},
            {"item_id": 2, "path": "file2.py", "content": "more code", "extension": ".py"},
        ]
        callback_calls = []

        def callback(current, total, path, chunks):
            callback_calls.append((current, total, path, chunks))

        with patch("publishing.embed.embed_text_content") as mock_embed:
            mock_embed.return_value = 1

            embed.embed_repo_files(file_items=file_items, progress_callback=callback)

            assert len(callback_calls) == 2
            assert callback_calls[0] == (1, 2, "file1.py", 1)
            assert callback_calls[1] == (2, 2, "file2.py", 1)
