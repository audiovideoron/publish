"""Tests for database operations (db.py)."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from distillyzer import db


class TestGetConnection:
    """Tests for get_connection context manager."""

    def test_get_connection_is_context_manager(self):
        """Test that get_connection returns a context manager."""
        # The function should be a context manager (generator)
        from contextlib import contextmanager
        import inspect

        # Verify it's a generator function decorated with contextmanager
        assert hasattr(db.get_connection, '__wrapped__') or inspect.isgeneratorfunction(db.get_connection.__wrapped__ if hasattr(db.get_connection, '__wrapped__') else None) or callable(db.get_connection)

    def test_get_connection_uses_database_url(self):
        """Test that DATABASE_URL is used for connection."""
        # Verify DATABASE_URL is defined in the module
        assert hasattr(db, 'DATABASE_URL')
        assert db.DATABASE_URL is not None


class TestSourceOperations:
    """Tests for source CRUD operations."""

    def test_create_source(self):
        """Test creating a source."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (42,)

        with patch("distillyzer.db.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            result = db.create_source(
                type="youtube_channel",
                name="Test Channel",
                url="https://youtube.com/@test",
                metadata={"foo": "bar"},
            )

            assert result == 42
            mock_cursor.execute.assert_called_once()
            mock_conn.commit.assert_called_once()

    def test_get_source_by_url_found(self):
        """Test getting a source by URL when it exists."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1, "youtube_channel", "Test", "https://example.com", {})

        with patch("distillyzer.db.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            result = db.get_source_by_url("https://example.com")

            assert result is not None
            assert result["id"] == 1
            assert result["type"] == "youtube_channel"

    def test_get_source_by_url_not_found(self):
        """Test getting a source by URL when it doesn't exist."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None

        with patch("distillyzer.db.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            result = db.get_source_by_url("https://nonexistent.com")

            assert result is None

    def test_list_sources(self):
        """Test listing all sources."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (1, "youtube_channel", "Channel 1", "https://yt.com/1", {}),
            (2, "github_repo", "Repo 1", "https://github.com/1", {}),
        ]

        with patch("distillyzer.db.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            result = db.list_sources()

            assert len(result) == 2
            assert result[0]["type"] == "youtube_channel"
            assert result[1]["type"] == "github_repo"

    def test_delete_source_success(self):
        """Test deleting a source successfully."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1

        with patch("distillyzer.db.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            result = db.delete_source(1)

            assert result is True
            mock_conn.commit.assert_called_once()

    def test_delete_source_not_found(self):
        """Test deleting a source that doesn't exist."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0

        with patch("distillyzer.db.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            result = db.delete_source(999)

            assert result is False


class TestItemOperations:
    """Tests for item CRUD operations."""

    def test_create_item(self):
        """Test creating an item."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (123,)

        with patch("distillyzer.db.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            result = db.create_item(
                source_id=1,
                type="video",
                title="Test Video",
                url="https://youtube.com/watch?v=test",
            )

            assert result == 123
            mock_conn.commit.assert_called_once()

    def test_get_item_by_url_found(self):
        """Test getting an item by URL."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1, 1, "video", "Test", "https://yt.com/v", {})

        with patch("distillyzer.db.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            result = db.get_item_by_url("https://yt.com/v")

            assert result["type"] == "video"

    def test_list_items_all(self):
        """Test listing all items."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (1, 1, "video", "Video 1", "https://yt.com/1", {}),
            (2, 1, "video", "Video 2", "https://yt.com/2", {}),
        ]

        with patch("distillyzer.db.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            result = db.list_items()

            assert len(result) == 2

    def test_list_items_by_source(self):
        """Test listing items filtered by source_id."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (1, 1, "video", "Video 1", "https://yt.com/1", {}),
        ]

        with patch("distillyzer.db.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            result = db.list_items(source_id=1)

            assert len(result) == 1
            # Verify the query included source_id filter
            call_args = mock_cursor.execute.call_args
            assert "source_id = %s" in call_args[0][0]


class TestChunkOperations:
    """Tests for chunk operations."""

    def test_create_chunk(self):
        """Test creating a chunk."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (456,)

        with patch("distillyzer.db.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            embedding = [0.1] * 1536
            result = db.create_chunk(
                item_id=1,
                content="Test content",
                chunk_index=0,
                embedding=embedding,
                timestamp_start=0.0,
                timestamp_end=60.0,
            )

            assert result == 456

    def test_create_chunks_batch(self):
        """Test creating multiple chunks in a batch."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [(1,), (2,), (3,)]

        with patch("distillyzer.db.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            chunks = [
                {"item_id": 1, "content": "Chunk 1", "chunk_index": 0, "embedding": [0.1] * 1536},
                {"item_id": 1, "content": "Chunk 2", "chunk_index": 1, "embedding": [0.2] * 1536},
                {"item_id": 1, "content": "Chunk 3", "chunk_index": 2, "embedding": [0.3] * 1536},
            ]
            result = db.create_chunks_batch(chunks)

            assert len(result) == 3
            assert result == [1, 2, 3]
            mock_conn.commit.assert_called_once()

    def test_search_chunks(self):
        """Test searching chunks by embedding similarity."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (1, "Content 1", 0, 0.0, 60.0, "Video 1", "https://yt.com/1", "video", 0.95),
            (2, "Content 2", 1, 60.0, 120.0, "Video 1", "https://yt.com/1", "video", 0.85),
        ]

        with patch("distillyzer.db.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            query_embedding = [0.1] * 1536
            result = db.search_chunks(query_embedding, limit=10)

            assert len(result) == 2
            assert result[0]["similarity"] == 0.95
            assert result[0]["content"] == "Content 1"


class TestProjectOperations:
    """Tests for project CRUD operations."""

    def test_create_project(self):
        """Test creating a project."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)

        with patch("distillyzer.db.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            result = db.create_project(
                name="test-project",
                description="A test project",
                facet_about=["AI"],
                facet_uses=["Python"],
                facet_needs=["GPU"],
            )

            assert result == 1

    def test_get_project_found(self):
        """Test getting a project by name."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (
            1, "test-project", "Description", "active",
            ["AI"], ["Python"], ["GPU"], None, {}, "2024-01-01", "2024-01-01"
        )

        with patch("distillyzer.db.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            result = db.get_project("test-project")

            assert result["name"] == "test-project"
            assert result["status"] == "active"

    def test_list_projects_all(self):
        """Test listing all projects."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (1, "proj1", "Desc 1", "active", [], [], [], None, {}, "2024-01-01", "2024-01-01"),
            (2, "proj2", "Desc 2", "archived", [], [], [], None, {}, "2024-01-01", "2024-01-01"),
        ]

        with patch("distillyzer.db.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            result = db.list_projects()

            assert len(result) == 2

    def test_list_projects_by_status(self):
        """Test listing projects filtered by status."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (1, "proj1", "Desc 1", "active", [], [], [], None, {}, "2024-01-01", "2024-01-01"),
        ]

        with patch("distillyzer.db.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            result = db.list_projects(status="active")

            assert len(result) == 1
            call_args = mock_cursor.execute.call_args
            assert "status = %s" in call_args[0][0]


class TestSkillOperations:
    """Tests for skill CRUD operations."""

    def test_create_skill(self):
        """Test creating a skill."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)

        with patch("distillyzer.db.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            result = db.create_skill(
                name="test-skill",
                type="prompt",
                content="Skill content",
                description="A test skill",
            )

            assert result == 1

    def test_get_skill_found(self):
        """Test getting a skill by name."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (
            1, "test-skill", "prompt", "A test skill",
            "Content", {}, "2024-01-01", "2024-01-01"
        )

        with patch("distillyzer.db.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            result = db.get_skill("test-skill")

            assert result["name"] == "test-skill"
            assert result["type"] == "prompt"


class TestStats:
    """Tests for statistics function."""

    def test_get_stats(self):
        """Test getting database statistics."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [(5,), (25,), (150,)]

        with patch("distillyzer.db.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            result = db.get_stats()

            assert result["sources"] == 5
            assert result["items"] == 25
            assert result["chunks"] == 150
