"""Distillyzer - Personal learning accelerator.

Harvest knowledge from YouTube, GitHub, and articles. Query it with semantic search.
Extract implementation artifacts. Build hello world demos.
"""

__version__ = "0.1.0"

# Database operations
from .db import (
    # Connection
    get_connection,
    # Sources CRUD
    create_source,
    get_source_by_url,
    get_or_create_source,
    list_sources,
    update_source,
    delete_source,
    # Items CRUD
    create_item,
    get_item_by_url,
    list_items,
    update_item,
    delete_item,
    # Chunks
    create_chunk,
    create_chunks_batch,
    delete_chunk,
    search_chunks,
    # Index helpers
    get_items_with_chunks,
    get_items_grouped_by_source,
    # Skills CRUD
    create_skill,
    get_skill,
    list_skills,
    update_skill,
    delete_skill,
    # Projects CRUD
    create_project,
    get_project,
    get_project_by_id,
    list_projects,
    update_project,
    delete_project,
    # Project linking
    link_project_item,
    unlink_project_item,
    get_project_items,
    get_item_projects,
    link_project_skill,
    unlink_project_skill,
    get_project_skills,
    # Project discovery
    update_project_embeddings,
    discover_for_project,
    discover_projects_for_content,
    # Project outputs
    create_project_output,
    get_project_output,
    list_project_outputs,
    update_project_output,
    delete_project_output,
    # Stats
    get_stats,
)

# Embedding operations
from .embed import (
    count_tokens,
    chunk_text,
    chunk_code,
    get_embedding,
    get_embeddings_batch,
    embed_transcript_chunks,
    embed_project_facets,
    embed_all_projects,
    embed_text_content,
    embed_repo_files,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_DIMS,
    get_embedding_model_info,
    MissingAPIKeyError,
)

# Query operations
from .query import (
    search,
    ask,
    chat_turn,
    format_timestamp,
    format_context,
)

# Harvest operations
from .harvest import (
    # Exceptions
    HarvestError,
    YtDlpError,
    GitCloneError,
    # YouTube
    parse_youtube_url,
    get_video_info,
    download_audio,
    search_youtube,
    harvest_video,
    harvest_channel,
    # GitHub
    parse_github_url,
    harvest_repo,
    # Articles
    harvest_article,
)

# Transcription
from .transcribe import (
    transcribe_audio,
    segments_to_timed_chunks,
)

# Extraction
from .extract import (
    extract_artifacts,
    extract_from_item,
    search_context,
    ArtifactType,
)

# Artifacts management
from .artifacts import (
    get_artifacts_dir,
    list_artifact_files,
    load_artifacts,
    list_all_artifacts,
    find_artifact,
    apply_artifact,
    search_artifacts,
    scaffold_project,
    analyze_for_demo,
    build_demo,
)

# Visualization
from .visualize import (
    generate_image,
    generate_from_chunk,
)

# Scoring - Video
from .scoring import (
    # Video scoring
    score_video,
    filter_videos_by_score,
    is_educational,
    VideoScoreBreakdown,
    EDUCATIONAL_KEYWORDS,
    NEGATIVE_VIDEO_KEYWORDS,
    # GitHub repo scoring
    score_github_repo,
    score_readme,
    score_documentation,
    score_repo_engagement,
    score_repo_from_path,
    filter_repos_by_score,
    RepoScoreBreakdown,
    # Backwards compat aliases
    ScoreBreakdown,
    score_engagement,
    calculate_negative_penalty,
)

# Search query generation
from .search_queries import (
    SearchQuery,
    SearchQuerySet,
    generate_from_project,
    generate_youtube_queries,
    generate_github_queries,
    generate_for_project_id,
    generate_for_project_name,
    YOUTUBE_TEMPLATES,
    GITHUB_TEMPLATES,
)

# CLI app
from .cli import app

__all__ = [
    # Version
    "__version__",
    # CLI
    "app",
    # Database - Connection
    "get_connection",
    # Database - Sources
    "create_source",
    "get_source_by_url",
    "get_or_create_source",
    "list_sources",
    "update_source",
    "delete_source",
    # Database - Items
    "create_item",
    "get_item_by_url",
    "list_items",
    "update_item",
    "delete_item",
    # Database - Chunks
    "create_chunk",
    "create_chunks_batch",
    "delete_chunk",
    "search_chunks",
    # Database - Index helpers
    "get_items_with_chunks",
    "get_items_grouped_by_source",
    # Database - Skills
    "create_skill",
    "get_skill",
    "list_skills",
    "update_skill",
    "delete_skill",
    # Database - Projects
    "create_project",
    "get_project",
    "get_project_by_id",
    "list_projects",
    "update_project",
    "delete_project",
    # Database - Project linking
    "link_project_item",
    "unlink_project_item",
    "get_project_items",
    "get_item_projects",
    "link_project_skill",
    "unlink_project_skill",
    "get_project_skills",
    # Database - Project discovery
    "update_project_embeddings",
    "discover_for_project",
    "discover_projects_for_content",
    # Database - Project outputs
    "create_project_output",
    "get_project_output",
    "list_project_outputs",
    "update_project_output",
    "delete_project_output",
    # Database - Stats
    "get_stats",
    # Embedding
    "count_tokens",
    "chunk_text",
    "chunk_code",
    "get_embedding",
    "get_embeddings_batch",
    "embed_transcript_chunks",
    "embed_project_facets",
    "embed_all_projects",
    "embed_text_content",
    "embed_repo_files",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL_DIMS",
    "get_embedding_model_info",
    "MissingAPIKeyError",
    # Query
    "search",
    "ask",
    "chat_turn",
    "format_timestamp",
    "format_context",
    # Harvest - Exceptions
    "HarvestError",
    "YtDlpError",
    "GitCloneError",
    # Harvest - YouTube
    "parse_youtube_url",
    "get_video_info",
    "download_audio",
    "search_youtube",
    "harvest_video",
    "harvest_channel",
    # Harvest - GitHub
    "parse_github_url",
    "harvest_repo",
    # Harvest - Articles
    "harvest_article",
    # Transcribe
    "transcribe_audio",
    "segments_to_timed_chunks",
    # Extract
    "extract_artifacts",
    "extract_from_item",
    "search_context",
    "ArtifactType",
    # Artifacts
    "get_artifacts_dir",
    "list_artifact_files",
    "load_artifacts",
    "list_all_artifacts",
    "find_artifact",
    "apply_artifact",
    "search_artifacts",
    "scaffold_project",
    "analyze_for_demo",
    "build_demo",
    # Visualize
    "generate_image",
    "generate_from_chunk",
    # Scoring - Video
    "score_video",
    "filter_videos_by_score",
    "is_educational",
    "VideoScoreBreakdown",
    "EDUCATIONAL_KEYWORDS",
    "NEGATIVE_VIDEO_KEYWORDS",
    # Scoring - GitHub repo
    "score_github_repo",
    "score_readme",
    "score_documentation",
    "score_repo_engagement",
    "score_repo_from_path",
    "filter_repos_by_score",
    "RepoScoreBreakdown",
    # Scoring - Backwards compat
    "ScoreBreakdown",
    "score_engagement",
    "calculate_negative_penalty",
    # Search query generation
    "SearchQuery",
    "SearchQuerySet",
    "generate_from_project",
    "generate_youtube_queries",
    "generate_github_queries",
    "generate_for_project_id",
    "generate_for_project_name",
    "YOUTUBE_TEMPLATES",
    "GITHUB_TEMPLATES",
]
