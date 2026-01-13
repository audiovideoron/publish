"""Tests for facet-based search query generation (search_queries.py)."""

import pytest
from unittest.mock import patch, MagicMock

from publishing import search_queries


class TestSlugify:
    """Tests for the _slugify helper function."""

    def test_basic_slugify(self):
        """Test basic text to slug conversion."""
        assert search_queries._slugify("Hello World") == "hello-world"

    def test_slugify_with_special_chars(self):
        """Test slugify removes special characters."""
        assert search_queries._slugify("Hello, World!") == "hello-world"

    def test_slugify_multiple_spaces(self):
        """Test slugify handles multiple spaces."""
        assert search_queries._slugify("Hello   World") == "hello-world"

    def test_slugify_leading_trailing_spaces(self):
        """Test slugify handles leading/trailing spaces."""
        assert search_queries._slugify("  Hello World  ") == "hello-world"

    def test_slugify_already_slug(self):
        """Test slugify handles already slugified text."""
        assert search_queries._slugify("hello-world") == "hello-world"

    def test_slugify_empty_string(self):
        """Test slugify handles empty string."""
        assert search_queries._slugify("") == ""

    def test_slugify_numbers(self):
        """Test slugify preserves numbers."""
        assert search_queries._slugify("Python 3.11") == "python-311"


class TestSearchQuery:
    """Tests for the SearchQuery dataclass."""

    def test_search_query_creation(self):
        """Test creating a SearchQuery."""
        query = search_queries.SearchQuery(
            query="python tutorial",
            platform="youtube",
            facet_type="needs",
            facet_value="Python",
            priority=1,
        )

        assert query.query == "python tutorial"
        assert query.platform == "youtube"
        assert query.facet_type == "needs"
        assert query.facet_value == "Python"
        assert query.priority == 1

    def test_search_query_default_priority(self):
        """Test SearchQuery has default priority of 1."""
        query = search_queries.SearchQuery(
            query="test",
            platform="github",
            facet_type="uses",
            facet_value="test",
        )

        assert query.priority == 1


class TestSearchQuerySet:
    """Tests for the SearchQuerySet dataclass."""

    @pytest.fixture
    def sample_query_set(self):
        """Create a sample query set for testing."""
        return search_queries.SearchQuerySet(
            project_name="test-project",
            queries=[
                search_queries.SearchQuery(
                    query="api auth tutorial",
                    platform="youtube",
                    facet_type="needs",
                    facet_value="API authentication",
                    priority=1,
                ),
                search_queries.SearchQuery(
                    query="api auth example",
                    platform="github",
                    facet_type="needs",
                    facet_value="API authentication",
                    priority=1,
                ),
                search_queries.SearchQuery(
                    query="fastapi tutorial",
                    platform="youtube",
                    facet_type="uses",
                    facet_value="FastAPI",
                    priority=2,
                ),
                search_queries.SearchQuery(
                    query="fastapi template",
                    platform="github",
                    facet_type="uses",
                    facet_value="FastAPI",
                    priority=2,
                ),
            ],
        )

    def test_by_platform_youtube(self, sample_query_set):
        """Test filtering by YouTube platform."""
        youtube_queries = sample_query_set.by_platform("youtube")

        assert len(youtube_queries) == 2
        assert all(q.platform == "youtube" for q in youtube_queries)

    def test_by_platform_github(self, sample_query_set):
        """Test filtering by GitHub platform."""
        github_queries = sample_query_set.by_platform("github")

        assert len(github_queries) == 2
        assert all(q.platform == "github" for q in github_queries)

    def test_by_facet_needs(self, sample_query_set):
        """Test filtering by needs facet."""
        needs_queries = sample_query_set.by_facet("needs")

        assert len(needs_queries) == 2
        assert all(q.facet_type == "needs" for q in needs_queries)

    def test_by_facet_uses(self, sample_query_set):
        """Test filtering by uses facet."""
        uses_queries = sample_query_set.by_facet("uses")

        assert len(uses_queries) == 2
        assert all(q.facet_type == "uses" for q in uses_queries)

    def test_youtube_queries(self, sample_query_set):
        """Test getting YouTube query strings."""
        queries = sample_query_set.youtube_queries()

        assert len(queries) == 2
        assert "api auth tutorial" in queries
        assert "fastapi tutorial" in queries

    def test_github_queries(self, sample_query_set):
        """Test getting GitHub query strings."""
        queries = sample_query_set.github_queries()

        assert len(queries) == 2
        assert "api auth example" in queries
        assert "fastapi template" in queries

    def test_queries_sorted_by_priority(self):
        """Test that queries are sorted by priority."""
        query_set = search_queries.SearchQuerySet(
            project_name="test",
            queries=[
                search_queries.SearchQuery(
                    query="low priority",
                    platform="youtube",
                    facet_type="about",
                    facet_value="test",
                    priority=10,
                ),
                search_queries.SearchQuery(
                    query="high priority",
                    platform="youtube",
                    facet_type="needs",
                    facet_value="test",
                    priority=1,
                ),
            ],
        )

        youtube = query_set.youtube_queries()

        assert youtube[0] == "high priority"
        assert youtube[1] == "low priority"


class TestGenerateQueriesForFacet:
    """Tests for _generate_queries_for_facet function."""

    def test_generate_needs_youtube_queries(self):
        """Test generating YouTube queries for NEEDS facet."""
        queries = search_queries._generate_queries_for_facet(
            facet_value="API authentication",
            facet_type="needs",
            platform="youtube",
        )

        query_strings = [q.query for q in queries]

        assert "API authentication tutorial" in query_strings
        assert "how to API authentication" in query_strings
        assert "API authentication explained" in query_strings

    def test_generate_uses_github_queries(self):
        """Test generating GitHub queries for USES facet."""
        queries = search_queries._generate_queries_for_facet(
            facet_value="FastAPI",
            facet_type="uses",
            platform="github",
        )

        query_strings = [q.query for q in queries]

        assert "FastAPI" in query_strings
        assert "FastAPI examples" in query_strings
        assert "FastAPI template" in query_strings

    def test_generate_with_context_terms(self):
        """Test generating queries with context from ABOUT facets."""
        queries = search_queries._generate_queries_for_facet(
            facet_value="authentication",
            facet_type="needs",
            platform="youtube",
            context_terms=["microservices", "Python"],
        )

        query_strings = [q.query for q in queries]

        # Should include context-enhanced queries
        assert "authentication microservices" in query_strings
        assert "authentication Python" in query_strings

    def test_queries_have_correct_metadata(self):
        """Test that generated queries have correct metadata."""
        queries = search_queries._generate_queries_for_facet(
            facet_value="Docker",
            facet_type="uses",
            platform="github",
        )

        for query in queries:
            assert query.platform == "github"
            assert query.facet_type == "uses"
            assert query.facet_value == "Docker"
            assert query.priority >= 1

    def test_queries_have_incrementing_priority(self):
        """Test that queries have incrementing priority based on template order."""
        queries = search_queries._generate_queries_for_facet(
            facet_value="test",
            facet_type="needs",
            platform="youtube",
        )

        priorities = [q.priority for q in queries]

        # First templates should have lower (higher priority) numbers
        assert priorities == sorted(priorities)


class TestGenerateFromProject:
    """Tests for generate_from_project function."""

    def test_generate_from_empty_project(self):
        """Test generating queries from project with no facets."""
        project = {"name": "empty-project"}

        query_set = search_queries.generate_from_project(project)

        assert query_set.project_name == "empty-project"
        assert len(query_set.queries) == 0

    def test_generate_from_project_with_needs(self):
        """Test generating queries from project with NEEDS facets."""
        project = {
            "name": "test-project",
            "facet_needs": ["API authentication", "caching"],
            "facet_uses": [],
            "facet_about": [],
        }

        query_set = search_queries.generate_from_project(project)

        # Should have queries for both platforms, for each need
        assert len(query_set.queries) > 0

        needs_queries = query_set.by_facet("needs")
        assert len(needs_queries) > 0

        # Check both platforms have queries
        youtube = query_set.by_platform("youtube")
        github = query_set.by_platform("github")
        assert len(youtube) > 0
        assert len(github) > 0

    def test_generate_from_project_with_uses(self):
        """Test generating queries from project with USES facets."""
        project = {
            "name": "test-project",
            "facet_needs": [],
            "facet_uses": ["Python", "PostgreSQL"],
            "facet_about": [],
        }

        query_set = search_queries.generate_from_project(project)

        uses_queries = query_set.by_facet("uses")
        assert len(uses_queries) > 0

        # USES queries should have adjusted priority (after NEEDS)
        for q in uses_queries:
            assert q.priority > 10

    def test_generate_from_project_with_about(self):
        """Test generating queries from project with ABOUT facets."""
        project = {
            "name": "test-project",
            "facet_needs": [],
            "facet_uses": [],
            "facet_about": ["machine learning", "data science"],
        }

        query_set = search_queries.generate_from_project(project)

        about_queries = query_set.by_facet("about")
        assert len(about_queries) > 0

        # ABOUT queries should have lowest priority
        for q in about_queries:
            assert q.priority > 20

    def test_generate_from_full_project(self, sample_project):
        """Test generating queries from full project."""
        query_set = search_queries.generate_from_project(sample_project)

        assert query_set.project_name == sample_project["name"]

        # Should have queries from all facet types
        needs = query_set.by_facet("needs")
        uses = query_set.by_facet("uses")
        about = query_set.by_facet("about")

        assert len(needs) > 0
        assert len(uses) > 0
        assert len(about) > 0

    def test_needs_have_highest_priority(self, sample_project):
        """Test that NEEDS queries have highest priority."""
        query_set = search_queries.generate_from_project(sample_project)

        needs = query_set.by_facet("needs")
        uses = query_set.by_facet("uses")
        about = query_set.by_facet("about")

        # Get min priorities for each facet type
        min_needs_priority = min(q.priority for q in needs)
        min_uses_priority = min(q.priority for q in uses)
        min_about_priority = min(q.priority for q in about)

        assert min_needs_priority < min_uses_priority
        assert min_uses_priority < min_about_priority

    def test_about_provides_context_for_needs(self):
        """Test that ABOUT facets provide context for NEEDS queries."""
        project = {
            "name": "context-test",
            "facet_needs": ["authentication"],
            "facet_uses": [],
            "facet_about": ["microservices"],
        }

        query_set = search_queries.generate_from_project(project)

        query_strings = [q.query for q in query_set.queries]

        # Should include context-enhanced query
        assert "authentication microservices" in query_strings


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_generate_youtube_queries_needs_only(self):
        """Test generating YouTube queries from NEEDS only."""
        queries = search_queries.generate_youtube_queries(
            needs=["API design", "testing"],
        )

        assert len(queries) > 0
        # Should have tutorial-style queries
        assert any("tutorial" in q for q in queries)
        assert any("how to" in q for q in queries)

    def test_generate_youtube_queries_uses_only(self):
        """Test generating YouTube queries from USES only."""
        queries = search_queries.generate_youtube_queries(
            uses=["FastAPI", "Docker"],
        )

        assert len(queries) > 0
        assert any("FastAPI" in q for q in queries)
        assert any("Docker" in q for q in queries)

    def test_generate_youtube_queries_max_limit(self):
        """Test that max_queries limits results."""
        queries = search_queries.generate_youtube_queries(
            needs=["a", "b", "c", "d", "e"],
            uses=["x", "y", "z"],
            max_queries=5,
        )

        assert len(queries) <= 5

    def test_generate_github_queries_needs_only(self):
        """Test generating GitHub queries from NEEDS only."""
        queries = search_queries.generate_github_queries(
            needs=["API design"],
        )

        assert len(queries) > 0
        # Should have example/template queries
        assert any("example" in q for q in queries)
        assert any("starter" in q or "boilerplate" in q for q in queries)

    def test_generate_github_queries_uses_only(self):
        """Test generating GitHub queries from USES only."""
        queries = search_queries.generate_github_queries(
            uses=["React", "TypeScript"],
        )

        assert len(queries) > 0
        assert any("React" in q for q in queries)

    def test_generate_github_queries_with_slugs(self):
        """Test that GitHub queries include slugified versions."""
        queries = search_queries.generate_github_queries(
            needs=["machine learning"],
        )

        # Should include awesome-list style query with slug
        assert any("awesome-machine-learning" in q for q in queries)


class TestDatabaseIntegration:
    """Tests for functions that integrate with the database."""

    def test_generate_for_project_id_not_found(self):
        """Test error when project ID not found."""
        with patch("publishing.db.get_project_by_id") as mock_get:
            mock_get.return_value = None

            with pytest.raises(ValueError, match="not found"):
                search_queries.generate_for_project_id(999)

    def test_generate_for_project_id_success(self, sample_project):
        """Test generating queries for project by ID."""
        with patch("publishing.db.get_project_by_id") as mock_get:
            mock_get.return_value = sample_project

            query_set = search_queries.generate_for_project_id(1)

            assert query_set.project_name == sample_project["name"]
            assert len(query_set.queries) > 0
            mock_get.assert_called_once_with(1)

    def test_generate_for_project_name_not_found(self):
        """Test error when project name not found."""
        with patch("publishing.db.get_project") as mock_get:
            mock_get.return_value = None

            with pytest.raises(ValueError, match="not found"):
                search_queries.generate_for_project_name("nonexistent")

    def test_generate_for_project_name_success(self, sample_project):
        """Test generating queries for project by name."""
        with patch("publishing.db.get_project") as mock_get:
            mock_get.return_value = sample_project

            query_set = search_queries.generate_for_project_name("test-project")

            assert query_set.project_name == sample_project["name"]
            assert len(query_set.queries) > 0
            mock_get.assert_called_once_with("test-project")


class TestQueryTemplates:
    """Tests for query template constants."""

    def test_youtube_templates_exist(self):
        """Test that YouTube templates are defined for all facet types."""
        assert "needs" in search_queries.YOUTUBE_TEMPLATES
        assert "uses" in search_queries.YOUTUBE_TEMPLATES
        assert "about" in search_queries.YOUTUBE_TEMPLATES

    def test_github_templates_exist(self):
        """Test that GitHub templates are defined for all facet types."""
        assert "needs" in search_queries.GITHUB_TEMPLATES
        assert "uses" in search_queries.GITHUB_TEMPLATES
        assert "about" in search_queries.GITHUB_TEMPLATES

    def test_youtube_needs_templates_are_tutorial_focused(self):
        """Test that NEEDS YouTube templates focus on learning."""
        templates = search_queries.YOUTUBE_TEMPLATES["needs"]

        # At least one should be tutorial focused
        tutorial_keywords = ["tutorial", "how to", "explained", "beginners"]
        assert any(
            any(keyword in t for keyword in tutorial_keywords) for t in templates
        )

    def test_github_uses_templates_include_examples(self):
        """Test that USES GitHub templates include examples."""
        templates = search_queries.GITHUB_TEMPLATES["uses"]

        # Should include example-style templates
        example_keywords = ["example", "template", "starter"]
        assert any(
            any(keyword in t for keyword in example_keywords) for t in templates
        )
