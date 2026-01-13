"""Shared pytest fixtures and configuration."""

import pytest
from unittest.mock import MagicMock, patch


# --- Mock Database Connection ---

@pytest.fixture
def mock_db_connection():
    """Mock database connection context manager."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    with patch("publishing.db.get_connection") as mock_get_conn:
        mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
        yield mock_conn, mock_cursor


# --- Mock OpenAI Client ---

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for embeddings and transcription."""
    with patch("publishing.embed.client") as mock_embed_client, \
         patch("publishing.transcribe.client") as mock_transcribe_client:
        yield {
            "embed": mock_embed_client,
            "transcribe": mock_transcribe_client,
        }


@pytest.fixture
def mock_embedding_response():
    """Create a mock embedding response."""
    def _create(embedding=None):
        if embedding is None:
            embedding = [0.1] * 1536  # Default embedding dimension
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=embedding)]
        return mock_response
    return _create


@pytest.fixture
def mock_transcription_response():
    """Create a mock transcription response."""
    def _create(text="Hello world", segments=None, language="en", duration=60.0):
        if segments is None:
            segments = [
                MagicMock(start=0.0, end=5.0, text="Hello"),
                MagicMock(start=5.0, end=10.0, text="world"),
            ]
        mock_response = MagicMock()
        mock_response.text = text
        mock_response.segments = segments
        mock_response.language = language
        mock_response.duration = duration
        return mock_response
    return _create


# --- Mock Anthropic Client ---

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for Claude API (via get_anthropic_client)."""
    with patch("publishing.query.get_anthropic_client") as mock_query_get_client, \
         patch("publishing.extract.get_anthropic_client") as mock_extract_get_client, \
         patch("publishing.artifacts.get_anthropic_client") as mock_artifacts_get_client:
        mock_query_client = MagicMock()
        mock_extract_client = MagicMock()
        mock_artifacts_client = MagicMock()
        mock_query_get_client.return_value = mock_query_client
        mock_extract_get_client.return_value = mock_extract_client
        mock_artifacts_get_client.return_value = mock_artifacts_client
        yield {
            "query": mock_query_client,
            "extract": mock_extract_client,
            "artifacts": mock_artifacts_client,
        }


@pytest.fixture
def mock_claude_response():
    """Create a mock Claude response."""
    def _create(text="This is a response", input_tokens=100, output_tokens=50):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=text)]
        mock_response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
        return mock_response
    return _create


# --- Sample Data Fixtures ---

@pytest.fixture
def sample_source():
    """Sample source data."""
    return {
        "id": 1,
        "type": "youtube_channel",
        "name": "Test Channel",
        "url": "https://youtube.com/@testchannel",
        "metadata": {"description": "A test channel"},
    }


@pytest.fixture
def sample_item():
    """Sample item data."""
    return {
        "id": 1,
        "source_id": 1,
        "type": "video",
        "title": "Test Video",
        "url": "https://youtube.com/watch?v=abc123",
        "metadata": {"channel": "Test Channel", "duration": 600},
    }


@pytest.fixture
def sample_chunk():
    """Sample chunk data."""
    return {
        "chunk_id": 1,
        "content": "This is test content about machine learning.",
        "chunk_index": 0,
        "timestamp_start": 0.0,
        "timestamp_end": 60.0,
        "item_title": "Test Video",
        "item_url": "https://youtube.com/watch?v=abc123",
        "item_type": "video",
        "similarity": 0.85,
    }


@pytest.fixture
def sample_chunks():
    """List of sample chunks for search results."""
    return [
        {
            "chunk_id": 1,
            "content": "Machine learning is a subset of artificial intelligence.",
            "chunk_index": 0,
            "timestamp_start": 0.0,
            "timestamp_end": 60.0,
            "item_title": "ML Basics",
            "item_url": "https://youtube.com/watch?v=ml1",
            "item_type": "video",
            "similarity": 0.92,
        },
        {
            "chunk_id": 2,
            "content": "Deep learning uses neural networks with many layers.",
            "chunk_index": 1,
            "timestamp_start": 60.0,
            "timestamp_end": 120.0,
            "item_title": "ML Basics",
            "item_url": "https://youtube.com/watch?v=ml1",
            "item_type": "video",
            "similarity": 0.87,
        },
    ]


@pytest.fixture
def sample_project():
    """Sample project data."""
    return {
        "id": 1,
        "name": "test-project",
        "description": "A test project",
        "status": "active",
        "facet_about": ["AI", "machine learning"],
        "facet_uses": ["Python", "TensorFlow"],
        "facet_needs": ["GPU training", "data pipeline"],
        "beads_epic_id": None,
        "metadata": {},
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
    }


@pytest.fixture
def sample_skill():
    """Sample skill data."""
    return {
        "id": 1,
        "name": "test-skill",
        "type": "prompt",
        "description": "A test skill",
        "content": "This is the skill content.",
        "metadata": {},
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
    }


@pytest.fixture
def sample_segments():
    """Sample transcription segments."""
    return [
        {"start": 0.0, "end": 10.0, "text": "Hello and welcome."},
        {"start": 10.0, "end": 25.0, "text": "Today we'll talk about AI."},
        {"start": 25.0, "end": 40.0, "text": "Machine learning is important."},
        {"start": 40.0, "end": 60.0, "text": "Let's dive into the details."},
    ]


# --- File System Fixtures ---

@pytest.fixture
def temp_audio_file(tmp_path):
    """Create a temporary audio file for testing."""
    audio_path = tmp_path / "test_audio.mp3"
    # Create a minimal file (just for path testing)
    audio_path.write_bytes(b"\x00" * 100)
    return audio_path


@pytest.fixture
def temp_repo_dir(tmp_path):
    """Create a temporary directory structure simulating a repo."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()

    # Create some Python files
    (repo_dir / "main.py").write_text('print("Hello")')
    (repo_dir / "utils.py").write_text('def helper(): pass')

    # Create a subdirectory with files
    src_dir = repo_dir / "src"
    src_dir.mkdir()
    (src_dir / "module.py").write_text('class MyClass: pass')

    # Create a README
    (repo_dir / "README.md").write_text("# Test Repo\n\nThis is a test.")

    return repo_dir
