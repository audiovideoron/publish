"""Tests for the 'pub suggest' CLI command."""

import pytest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from publishing.cli import app


runner = CliRunner()


class TestFormatViews:
    """Tests for the _format_views helper function."""

    def test_format_views_none(self):
        """Test formatting None view count."""
        from publishing.cli import _format_views
        assert _format_views(None) == "?"

    def test_format_views_small(self):
        """Test formatting small view count."""
        from publishing.cli import _format_views
        assert _format_views(500) == "500"

    def test_format_views_thousands(self):
        """Test formatting thousands."""
        from publishing.cli import _format_views
        assert _format_views(1500) == "1.5K"
        assert _format_views(10000) == "10.0K"

    def test_format_views_millions(self):
        """Test formatting millions."""
        from publishing.cli import _format_views
        assert _format_views(1_500_000) == "1.5M"
        assert _format_views(10_000_000) == "10.0M"


class TestDisplayProjectFacets:
    """Tests for the _display_project_facets helper function."""

    def test_display_with_all_facets(self, capsys):
        """Test display with all facets present."""
        from publishing.cli import _display_project_facets, console

        project = {
            "facet_needs": ["API auth", "testing"],
            "facet_uses": ["Python", "FastAPI"],
            "facet_about": ["microservices"],
        }

        # Use a fresh console capture
        with patch.object(console, 'print') as mock_print:
            _display_project_facets(project)
            # Check that print was called with the facet info
            call_args = str(mock_print.call_args_list)
            assert "NEEDS" in call_args
            assert "USES" in call_args
            assert "ABOUT" in call_args

    def test_display_with_no_facets(self):
        """Test display warning when no facets defined."""
        from publishing.cli import _display_project_facets, console

        project = {
            "facet_needs": [],
            "facet_uses": [],
            "facet_about": [],
        }

        with patch.object(console, 'print') as mock_print:
            _display_project_facets(project)
            # Check that warning was printed
            call_args = str(mock_print.call_args_list)
            assert "Warning" in call_args or "no facets" in call_args.lower()


class TestSuggestYoutubeHelper:
    """Tests for the _suggest_youtube helper function."""

    @pytest.fixture
    def sample_project(self):
        """Sample project with facets."""
        return {
            "name": "test-project",
            "facet_needs": ["API authentication"],
            "facet_uses": ["Python", "FastAPI"],
            "facet_about": ["web development"],
        }

    def test_suggest_youtube_no_facets(self):
        """Test with project having no facets."""
        from publishing.cli import _suggest_youtube

        project = {"name": "empty", "facet_needs": [], "facet_uses": [], "facet_about": []}
        result = _suggest_youtube(project)
        assert result == []

    def test_suggest_youtube_with_facets(self, sample_project):
        """Test with project having facets."""
        from publishing.cli import _suggest_youtube

        mock_video = {
            "id": "abc123",
            "title": "FastAPI Tutorial - Full Course",
            "url": "https://youtube.com/watch?v=abc123",
            "channel": "TechChannel",
            "duration": 3600,
        }

        mock_info = {
            "view_count": 100000,
            "like_count": 5000,
            "duration": 3600,
            "description": "Learn FastAPI step by step",
        }

        with patch("publishing.cli.harv.search_youtube") as mock_search, \
             patch("publishing.cli.harv.get_video_info") as mock_get_info:
            mock_search.return_value = [mock_video]
            mock_get_info.return_value = mock_info

            result = _suggest_youtube(sample_project, min_score=0, limit=10)

            assert len(result) >= 1
            # Should have called search with generated queries
            mock_search.assert_called()
            # Videos should have scores
            for video in result:
                assert "score" in video

    def test_suggest_youtube_filters_by_score(self, sample_project):
        """Test that videos are filtered by min_score."""
        from publishing.cli import _suggest_youtube

        mock_video = {
            "id": "abc123",
            "title": "Random Video",  # Not educational
            "url": "https://youtube.com/watch?v=abc123",
            "channel": "SomeChannel",
            "duration": 60,  # Short video
        }

        mock_info = {
            "view_count": 100,
            "like_count": 1,
            "duration": 60,
            "description": "",
        }

        with patch("publishing.cli.harv.search_youtube") as mock_search, \
             patch("publishing.cli.harv.get_video_info") as mock_get_info:
            mock_search.return_value = [mock_video]
            mock_get_info.return_value = mock_info

            # High min_score should filter out low quality videos
            result = _suggest_youtube(sample_project, min_score=80, limit=10)

            # Should be empty or have fewer results
            assert len(result) == 0 or all(v.get("score", 0) >= 80 for v in result)

    def test_suggest_youtube_deduplicates(self, sample_project):
        """Test that duplicate videos are deduplicated."""
        from publishing.cli import _suggest_youtube

        mock_video = {
            "id": "abc123",
            "title": "FastAPI Tutorial",
            "url": "https://youtube.com/watch?v=abc123",
            "channel": "TechChannel",
            "duration": 3600,
        }

        mock_info = {
            "view_count": 100000,
            "like_count": 5000,
            "duration": 3600,
            "description": "Tutorial content",
        }

        with patch("publishing.cli.harv.search_youtube") as mock_search, \
             patch("publishing.cli.harv.get_video_info") as mock_get_info:
            # Return same video for multiple queries
            mock_search.return_value = [mock_video]
            mock_get_info.return_value = mock_info

            result = _suggest_youtube(sample_project, min_score=0, limit=10, max_queries=3)

            # Should only have one instance of the video despite multiple queries
            ids = [v.get("id") for v in result]
            assert len(ids) == len(set(ids))  # No duplicates


class TestSuggestGithubHelper:
    """Tests for the _suggest_github helper function."""

    @pytest.fixture
    def sample_project(self):
        """Sample project with facets."""
        return {
            "name": "test-project",
            "facet_needs": ["API authentication"],
            "facet_uses": ["Python", "FastAPI"],
            "facet_about": ["web development"],
        }

    def test_suggest_github_no_facets(self):
        """Test with project having no facets."""
        from publishing.cli import _suggest_github

        project = {"name": "empty", "facet_needs": [], "facet_uses": [], "facet_about": []}

        with patch("publishing.cli.console.print"):
            result = _suggest_github(project)
            # Currently returns empty since GitHub search is not implemented
            assert result == []

    def test_suggest_github_shows_queries(self, sample_project):
        """Test that GitHub queries are shown (placeholder behavior)."""
        from publishing.cli import _suggest_github, console

        with patch.object(console, 'print') as mock_print:
            result = _suggest_github(sample_project)

            # Should show generated queries
            call_args = str(mock_print.call_args_list)
            assert "queries" in call_args.lower() or "GitHub search is not yet implemented" in call_args


class TestSuggestYoutubeCommand:
    """Tests for the 'pub suggest youtube' CLI command."""

    @pytest.fixture
    def sample_project(self):
        """Sample project data."""
        return {
            "id": 1,
            "name": "test-project",
            "description": "A test project",
            "status": "active",
            "facet_about": ["AI", "machine learning"],
            "facet_uses": ["Python", "TensorFlow"],
            "facet_needs": ["GPU training", "data pipeline"],
        }

    def test_project_not_found(self):
        """Test error when project not found."""
        with patch("publishing.cli.db.get_project") as mock_get:
            mock_get.return_value = None

            result = runner.invoke(app, ["suggest", "youtube", "nonexistent"])

            assert result.exit_code == 0  # Command completes but shows error
            assert "not found" in result.output.lower()

    def test_youtube_suggestions_displayed(self, sample_project):
        """Test that YouTube suggestions are displayed."""
        mock_video = {
            "id": "vid1",
            "title": "Machine Learning Tutorial",
            "url": "https://youtube.com/watch?v=vid1",
            "channel": "LearnML",
            "duration": 3600,
        }

        mock_info = {
            "view_count": 500000,
            "like_count": 25000,
            "duration": 3600,
            "description": "Complete machine learning tutorial",
        }

        with patch("publishing.cli.db.get_project") as mock_get_project, \
             patch("publishing.cli.harv.search_youtube") as mock_search, \
             patch("publishing.cli.harv.get_video_info") as mock_get_info:
            mock_get_project.return_value = sample_project
            mock_search.return_value = [mock_video]
            mock_get_info.return_value = mock_info

            result = runner.invoke(app, ["suggest", "youtube", "test-project"])

            assert result.exit_code == 0
            # Should show project name
            assert "test-project" in result.output
            # Should show video title or table
            assert "Machine Learning" in result.output or "YouTube Videos" in result.output

    def test_no_results_message(self, sample_project):
        """Test message when no videos match criteria."""
        with patch("publishing.cli.db.get_project") as mock_get_project, \
             patch("publishing.cli.harv.search_youtube") as mock_search:
            mock_get_project.return_value = sample_project
            mock_search.return_value = []

            result = runner.invoke(app, ["suggest", "youtube", "test-project"])

            assert result.exit_code == 0
            assert "no videos" in result.output.lower()

    def test_min_score_option(self, sample_project):
        """Test --min-score option."""
        with patch("publishing.cli.db.get_project") as mock_get_project, \
             patch("publishing.cli.harv.search_youtube") as mock_search:
            mock_get_project.return_value = sample_project
            mock_search.return_value = []

            result = runner.invoke(app, ["suggest", "youtube", "test-project", "--min-score", "80"])

            assert result.exit_code == 0

    def test_limit_option(self, sample_project):
        """Test --limit option."""
        with patch("publishing.cli.db.get_project") as mock_get_project, \
             patch("publishing.cli.harv.search_youtube") as mock_search:
            mock_get_project.return_value = sample_project
            mock_search.return_value = []

            result = runner.invoke(app, ["suggest", "youtube", "test-project", "--limit", "5"])

            assert result.exit_code == 0


class TestSuggestGithubCommand:
    """Tests for the 'pub suggest github' CLI command."""

    @pytest.fixture
    def sample_project(self):
        """Sample project data."""
        return {
            "id": 1,
            "name": "test-project",
            "description": "A test project",
            "status": "active",
            "facet_about": ["AI", "machine learning"],
            "facet_uses": ["Python", "TensorFlow"],
            "facet_needs": ["GPU training", "data pipeline"],
        }

    def test_project_not_found(self):
        """Test error when project not found."""
        with patch("publishing.cli.db.get_project") as mock_get:
            mock_get.return_value = None

            result = runner.invoke(app, ["suggest", "github", "nonexistent"])

            assert result.exit_code == 0
            assert "not found" in result.output.lower()

    def test_github_shows_placeholder(self, sample_project):
        """Test that GitHub shows placeholder message."""
        with patch("publishing.cli.db.get_project") as mock_get_project:
            mock_get_project.return_value = sample_project

            result = runner.invoke(app, ["suggest", "github", "test-project"])

            assert result.exit_code == 0
            # Should show project name
            assert "test-project" in result.output
            # Should indicate GitHub is not yet implemented or show queries
            assert "not yet implemented" in result.output.lower() or "queries" in result.output.lower()


class TestSuggestAllCommand:
    """Tests for the 'pub suggest all' CLI command."""

    @pytest.fixture
    def sample_project(self):
        """Sample project data."""
        return {
            "id": 1,
            "name": "test-project",
            "description": "A test project",
            "status": "active",
            "facet_about": ["AI", "machine learning"],
            "facet_uses": ["Python", "TensorFlow"],
            "facet_needs": ["GPU training", "data pipeline"],
        }

    def test_project_not_found(self):
        """Test error when project not found."""
        with patch("publishing.cli.db.get_project") as mock_get:
            mock_get.return_value = None

            result = runner.invoke(app, ["suggest", "all", "nonexistent"])

            assert result.exit_code == 0
            assert "not found" in result.output.lower()

    def test_all_searches_both(self, sample_project):
        """Test that 'all' searches both YouTube and GitHub."""
        mock_video = {
            "id": "vid1",
            "title": "Python Tutorial",
            "url": "https://youtube.com/watch?v=vid1",
            "channel": "PyChannel",
            "duration": 3600,
        }

        mock_info = {
            "view_count": 100000,
            "like_count": 5000,
            "duration": 3600,
            "description": "Python tutorial",
        }

        with patch("publishing.cli.db.get_project") as mock_get_project, \
             patch("publishing.cli.harv.search_youtube") as mock_search, \
             patch("publishing.cli.harv.get_video_info") as mock_get_info:
            mock_get_project.return_value = sample_project
            mock_search.return_value = [mock_video]
            mock_get_info.return_value = mock_info

            result = runner.invoke(app, ["suggest", "all", "test-project"])

            assert result.exit_code == 0
            # Should show both YouTube and GitHub sections
            assert "YouTube" in result.output
            # GitHub section should be present (even if showing placeholder)
            assert "GitHub" in result.output or "queries" in result.output.lower()


class TestSuggestCommandHelp:
    """Tests for suggest command help text."""

    def test_suggest_help(self):
        """Test that suggest command has help text."""
        result = runner.invoke(app, ["suggest", "--help"])

        assert result.exit_code == 0
        assert "suggest" in result.output.lower()
        assert "youtube" in result.output.lower()
        assert "github" in result.output.lower()

    def test_suggest_youtube_help(self):
        """Test that suggest youtube has help text."""
        result = runner.invoke(app, ["suggest", "youtube", "--help"])

        assert result.exit_code == 0
        assert "youtube" in result.output.lower()
        assert "project" in result.output.lower()

    def test_suggest_github_help(self):
        """Test that suggest github has help text."""
        result = runner.invoke(app, ["suggest", "github", "--help"])

        assert result.exit_code == 0
        assert "github" in result.output.lower()

    def test_suggest_all_help(self):
        """Test that suggest all has help text."""
        result = runner.invoke(app, ["suggest", "all", "--help"])

        assert result.exit_code == 0
        assert "all" in result.output.lower() or "both" in result.output.lower()
