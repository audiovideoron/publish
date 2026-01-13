"""Generate search queries from project facets for content discovery.

Transforms project facets (NEEDS, USES, ABOUT) into targeted search queries
for YouTube and GitHub to find relevant learning content.
"""

from dataclasses import dataclass, field
from typing import Literal

Platform = Literal["youtube", "github"]


@dataclass
class SearchQuery:
    """A search query with metadata."""

    query: str
    platform: Platform
    facet_type: Literal["needs", "uses", "about"]
    facet_value: str
    priority: int = 1  # 1 = highest priority


@dataclass
class SearchQuerySet:
    """Collection of search queries for a project."""

    project_name: str
    queries: list[SearchQuery] = field(default_factory=list)

    def by_platform(self, platform: Platform) -> list[SearchQuery]:
        """Get queries for a specific platform."""
        return [q for q in self.queries if q.platform == platform]

    def by_facet(self, facet_type: str) -> list[SearchQuery]:
        """Get queries for a specific facet type."""
        return [q for q in self.queries if q.facet_type == facet_type]

    def youtube_queries(self) -> list[str]:
        """Get YouTube query strings, sorted by priority."""
        queries = self.by_platform("youtube")
        queries.sort(key=lambda q: q.priority)
        return [q.query for q in queries]

    def github_queries(self) -> list[str]:
        """Get GitHub query strings, sorted by priority."""
        queries = self.by_platform("github")
        queries.sort(key=lambda q: q.priority)
        return [q.query for q in queries]


# YouTube query templates for different facet types
YOUTUBE_TEMPLATES = {
    "needs": [
        "{facet} tutorial",
        "how to {facet}",
        "{facet} explained",
        "{facet} for beginners",
        "{facet} step by step",
    ],
    "uses": [
        "{facet} tutorial",
        "{facet} guide",
        "{facet} crash course",
        "{facet} best practices",
        "getting started with {facet}",
    ],
    "about": [
        "{facet}",
        "{facet} overview",
    ],
}

# GitHub query templates for different facet types
GITHUB_TEMPLATES = {
    "needs": [
        "{facet} example",
        "{facet} tutorial",
        "awesome-{facet_slug}",
        "{facet} starter",
        "{facet} boilerplate",
    ],
    "uses": [
        "{facet}",
        "{facet} examples",
        "{facet} template",
        "{facet_slug}-starter",
        "{facet_slug}-tutorial",
    ],
    "about": [
        "{facet}",
        "awesome-{facet_slug}",
    ],
}


def _slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    # Convert to lowercase, replace spaces with hyphens
    slug = text.lower().strip()
    slug = slug.replace(" ", "-")
    # Remove special characters, keep alphanumeric and hyphens
    slug = "".join(c for c in slug if c.isalnum() or c == "-")
    # Remove multiple consecutive hyphens
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def _generate_queries_for_facet(
    facet_value: str,
    facet_type: Literal["needs", "uses", "about"],
    platform: Platform,
    context_terms: list[str] | None = None,
) -> list[SearchQuery]:
    """Generate search queries for a single facet value.

    Args:
        facet_value: The facet value (e.g., "API authentication")
        facet_type: Type of facet (needs, uses, about)
        platform: Target platform (youtube, github)
        context_terms: Optional terms from ABOUT facets to add context

    Returns:
        List of SearchQuery objects
    """
    templates = YOUTUBE_TEMPLATES if platform == "youtube" else GITHUB_TEMPLATES
    facet_templates = templates.get(facet_type, [])

    queries = []
    facet_slug = _slugify(facet_value)

    for priority, template in enumerate(facet_templates, start=1):
        query_text = template.format(facet=facet_value, facet_slug=facet_slug)

        queries.append(
            SearchQuery(
                query=query_text,
                platform=platform,
                facet_type=facet_type,
                facet_value=facet_value,
                priority=priority,
            )
        )

    # Add context-enhanced queries for NEEDS facets (most important)
    if facet_type == "needs" and context_terms:
        for context in context_terms[:2]:  # Limit to top 2 context terms
            context_query = f"{facet_value} {context}"
            queries.append(
                SearchQuery(
                    query=context_query,
                    platform=platform,
                    facet_type=facet_type,
                    facet_value=facet_value,
                    priority=len(facet_templates) + 1,  # Lower priority
                )
            )

    return queries


def generate_from_project(project: dict) -> SearchQuerySet:
    """Generate search queries from a project's facets.

    The project dict should have:
    - name: Project name
    - facet_needs: List of things the project needs to learn
    - facet_uses: List of technologies/tools the project uses
    - facet_about: List of topics the project is about

    Priority ordering:
    1. NEEDS facets (highest priority - these are learning gaps)
    2. USES facets (medium priority - deepen tool knowledge)
    3. ABOUT facets (lowest priority - context/background)

    Args:
        project: Project dictionary with facet fields

    Returns:
        SearchQuerySet containing all generated queries
    """
    query_set = SearchQuerySet(project_name=project.get("name", "unknown"))

    # Get facets
    needs = project.get("facet_needs", []) or []
    uses = project.get("facet_uses", []) or []
    about = project.get("facet_about", []) or []

    # ABOUT terms provide context for other queries
    context_terms = about[:3] if about else []

    # Generate NEEDS queries (highest priority)
    for facet in needs:
        for platform in ["youtube", "github"]:
            queries = _generate_queries_for_facet(
                facet, "needs", platform, context_terms
            )
            # Boost NEEDS priority
            for q in queries:
                q.priority = q.priority  # Keep relative ordering
            query_set.queries.extend(queries)

    # Generate USES queries (medium priority)
    for facet in uses:
        for platform in ["youtube", "github"]:
            queries = _generate_queries_for_facet(facet, "uses", platform)
            # Adjust priority to be after NEEDS
            for q in queries:
                q.priority = q.priority + 10
            query_set.queries.extend(queries)

    # Generate ABOUT queries (lowest priority, limited)
    for facet in about[:2]:  # Only top 2 about topics
        for platform in ["youtube", "github"]:
            queries = _generate_queries_for_facet(facet, "about", platform)
            # Lowest priority
            for q in queries:
                q.priority = q.priority + 20
            query_set.queries.extend(queries)

    return query_set


def generate_youtube_queries(
    needs: list[str] | None = None,
    uses: list[str] | None = None,
    about: list[str] | None = None,
    max_queries: int = 20,
) -> list[str]:
    """Generate YouTube search queries from facets.

    Convenience function that returns just the query strings.

    Args:
        needs: List of skills/knowledge gaps to fill
        uses: List of tools/frameworks being used
        about: List of topics for context
        max_queries: Maximum number of queries to return

    Returns:
        List of YouTube search query strings, ordered by priority
    """
    project = {
        "name": "adhoc",
        "facet_needs": needs or [],
        "facet_uses": uses or [],
        "facet_about": about or [],
    }

    query_set = generate_from_project(project)
    return query_set.youtube_queries()[:max_queries]


def generate_github_queries(
    needs: list[str] | None = None,
    uses: list[str] | None = None,
    about: list[str] | None = None,
    max_queries: int = 20,
) -> list[str]:
    """Generate GitHub search queries from facets.

    Convenience function that returns just the query strings.

    Args:
        needs: List of skills/knowledge gaps to fill
        uses: List of tools/frameworks being used
        about: List of topics for context
        max_queries: Maximum number of queries to return

    Returns:
        List of GitHub search query strings, ordered by priority
    """
    project = {
        "name": "adhoc",
        "facet_needs": needs or [],
        "facet_uses": uses or [],
        "facet_about": about or [],
    }

    query_set = generate_from_project(project)
    return query_set.github_queries()[:max_queries]


def generate_for_project_id(project_id: int) -> SearchQuerySet:
    """Generate search queries for a project by ID.

    Loads the project from the database and generates queries.

    Args:
        project_id: Database ID of the project

    Returns:
        SearchQuerySet for the project

    Raises:
        ValueError: If project not found
    """
    from . import db

    project = db.get_project_by_id(project_id)
    if not project:
        raise ValueError(f"Project with ID {project_id} not found")

    return generate_from_project(project)


def generate_for_project_name(project_name: str) -> SearchQuerySet:
    """Generate search queries for a project by name.

    Loads the project from the database and generates queries.

    Args:
        project_name: Name of the project

    Returns:
        SearchQuerySet for the project

    Raises:
        ValueError: If project not found
    """
    from . import db

    project = db.get_project(project_name)
    if not project:
        raise ValueError(f"Project '{project_name}' not found")

    return generate_from_project(project)
