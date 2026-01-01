"""Database operations for Distillyzer."""

import atexit
import os
from contextlib import contextmanager
from typing import Generator
from urllib.parse import urlparse

import numpy as np
import psycopg
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

load_dotenv()


class DatabaseConfigError(Exception):
    """Raised when DATABASE_URL is missing or malformed."""

    pass


def _validate_database_url(url: str | None) -> str:
    """Validate DATABASE_URL and return it if valid.

    Raises DatabaseConfigError with a clear message if:
    - DATABASE_URL is not set
    - DATABASE_URL is empty
    - DATABASE_URL has an invalid scheme (must be postgresql:// or postgres://)
    - DATABASE_URL is missing required components (host or database name)
    """
    if url is None or url.strip() == "":
        raise DatabaseConfigError(
            "DATABASE_URL environment variable is not set.\n"
            "Please set it to a valid PostgreSQL connection string, e.g.:\n"
            "  export DATABASE_URL='postgresql://user:password@localhost:5432/distillyzer'\n"
            "Or add it to your .env file."
        )

    url = url.strip()

    # Check scheme
    if not url.startswith(("postgresql://", "postgres://")):
        raise DatabaseConfigError(
            f"DATABASE_URL has invalid scheme.\n"
            f"Expected 'postgresql://' or 'postgres://', got: {url.split('://')[0] if '://' in url else 'no scheme'}\n"
            f"Example: postgresql://user:password@localhost:5432/distillyzer"
        )

    try:
        parsed = urlparse(url)
    except Exception as e:
        raise DatabaseConfigError(
            f"DATABASE_URL could not be parsed: {e}\n"
            f"Please provide a valid PostgreSQL connection string.\n"
            f"Example: postgresql://user:password@localhost:5432/distillyzer"
        )

    # Check for host
    if not parsed.hostname:
        raise DatabaseConfigError(
            f"DATABASE_URL is missing the host.\n"
            f"Got: {url}\n"
            f"Example: postgresql://user:password@localhost:5432/distillyzer"
        )

    # Check for database name (path should be /dbname)
    db_name = parsed.path.lstrip("/") if parsed.path else ""
    if not db_name:
        raise DatabaseConfigError(
            f"DATABASE_URL is missing the database name.\n"
            f"Got: {url}\n"
            f"Example: postgresql://user:password@localhost:5432/distillyzer"
        )

    return url


# Validate DATABASE_URL at module load time for early failure
DATABASE_URL = _validate_database_url(os.getenv("DATABASE_URL"))

# Connection pool configuration
# min_size: minimum connections to keep open
# max_size: maximum connections allowed
# max_idle: close idle connections after this many seconds
_pool: ConnectionPool | None = None


def _configure_connection(conn: psycopg.Connection) -> None:
    """Configure a connection after it's created (register pgvector)."""
    register_vector(conn)


def _get_pool() -> ConnectionPool:
    """Get or create the connection pool (lazy initialization)."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            DATABASE_URL,
            min_size=1,
            max_size=10,
            max_idle=300,  # Close idle connections after 5 minutes
            configure=_configure_connection,
        )
    return _pool


def close_pool() -> None:
    """Close the connection pool. Call this on application shutdown."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


# Register cleanup on interpreter exit
atexit.register(close_pool)


@contextmanager
def get_connection() -> Generator[psycopg.Connection, None, None]:
    """Get a database connection from the pool with pgvector support.

    The connection is automatically returned to the pool when the context exits.
    """
    pool = _get_pool()
    with pool.connection() as conn:
        yield conn


# --- Sources ---

def create_source(type: str, name: str, url: str, metadata: dict | None = None) -> int:
    """Create a source (channel, repo) and return its ID."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sources (type, name, url, metadata)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (type, name, url, Jsonb(metadata) if metadata else None),
            )
            result = cur.fetchone()
            conn.commit()
            return result[0]


def get_source_by_url(url: str) -> dict | None:
    """Get a source by URL."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, type, name, url, metadata FROM sources WHERE url = %s",
                (url,),
            )
            row = cur.fetchone()
            if row:
                return {
                    "id": row[0],
                    "type": row[1],
                    "name": row[2],
                    "url": row[3],
                    "metadata": row[4],
                }
            return None


def get_or_create_source(type: str, name: str, url: str) -> int:
    """Get existing source by name or create new one. Returns source ID."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Try to find by name first
            cur.execute(
                "SELECT id FROM sources WHERE name = %s AND type = %s",
                (name, type),
            )
            row = cur.fetchone()
            if row:
                return row[0]
            # Create new source
            cur.execute(
                """
                INSERT INTO sources (type, name, url)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (type, name, url),
            )
            result = cur.fetchone()
            conn.commit()
            return result[0]


def list_sources() -> list[dict]:
    """Return all sources."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, type, name, url, metadata FROM sources ORDER BY id"
            )
            return [
                {
                    "id": row[0],
                    "type": row[1],
                    "name": row[2],
                    "url": row[3],
                    "metadata": row[4],
                }
                for row in cur.fetchall()
            ]


def update_source(
    id: int,
    type: str | None = None,
    name: str | None = None,
    url: str | None = None,
    metadata: dict | None = None,
) -> dict | None:
    """Update a source and return the updated source."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE sources
                SET type = COALESCE(%s, type),
                    name = COALESCE(%s, name),
                    url = COALESCE(%s, url),
                    metadata = COALESCE(%s, metadata)
                WHERE id = %s
                RETURNING id, type, name, url, metadata
                """,
                (type, name, url, Jsonb(metadata) if metadata else None, id),
            )
            row = cur.fetchone()
            conn.commit()
            if row:
                return {
                    "id": row[0],
                    "type": row[1],
                    "name": row[2],
                    "url": row[3],
                    "metadata": row[4],
                }
            return None


def delete_source(id: int) -> bool:
    """Delete a source by ID."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM sources WHERE id = %s", (id,))
            conn.commit()
            return cur.rowcount > 0


# --- Items ---

def create_item(
    source_id: int | None,
    type: str,
    title: str,
    url: str | None = None,
    metadata: dict | None = None,
) -> int:
    """Create an item (video, file) and return its ID."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO items (source_id, type, title, url, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (source_id, type, title, url, Jsonb(metadata) if metadata else None),
            )
            result = cur.fetchone()
            conn.commit()
            return result[0]


def get_item_by_url(url: str) -> dict | None:
    """Get an item by URL."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, source_id, type, title, url, metadata FROM items WHERE url = %s",
                (url,),
            )
            row = cur.fetchone()
            if row:
                return {
                    "id": row[0],
                    "source_id": row[1],
                    "type": row[2],
                    "title": row[3],
                    "url": row[4],
                    "metadata": row[5],
                }
            return None


def get_item_by_id(id: int) -> dict | None:
    """Get an item by ID."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, source_id, type, title, url, metadata FROM items WHERE id = %s",
                (id,),
            )
            row = cur.fetchone()
            if row:
                return {
                    "id": row[0],
                    "source_id": row[1],
                    "type": row[2],
                    "title": row[3],
                    "url": row[4],
                    "metadata": row[5],
                }
            return None


def list_items(source_id: int | None = None) -> list[dict]:
    """Return all items, optionally filtered by source_id."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            if source_id is not None:
                cur.execute(
                    "SELECT id, source_id, type, title, url, metadata FROM items WHERE source_id = %s ORDER BY id",
                    (source_id,),
                )
            else:
                cur.execute(
                    "SELECT id, source_id, type, title, url, metadata FROM items ORDER BY id"
                )
            return [
                {
                    "id": row[0],
                    "source_id": row[1],
                    "type": row[2],
                    "title": row[3],
                    "url": row[4],
                    "metadata": row[5],
                }
                for row in cur.fetchall()
            ]


def update_item(
    id: int,
    source_id: int | None = None,
    type: str | None = None,
    title: str | None = None,
    url: str | None = None,
    metadata: dict | None = None,
) -> dict | None:
    """Update an item and return the updated item."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE items
                SET source_id = COALESCE(%s, source_id),
                    type = COALESCE(%s, type),
                    title = COALESCE(%s, title),
                    url = COALESCE(%s, url),
                    metadata = COALESCE(%s, metadata)
                WHERE id = %s
                RETURNING id, source_id, type, title, url, metadata
                """,
                (source_id, type, title, url, Jsonb(metadata) if metadata else None, id),
            )
            row = cur.fetchone()
            conn.commit()
            if row:
                return {
                    "id": row[0],
                    "source_id": row[1],
                    "type": row[2],
                    "title": row[3],
                    "url": row[4],
                    "metadata": row[5],
                }
            return None


def delete_item(id: int) -> bool:
    """Delete an item by ID. Chunks are deleted via CASCADE."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM items WHERE id = %s", (id,))
            conn.commit()
            return cur.rowcount > 0


# --- Chunks ---

def create_chunk(
    item_id: int,
    content: str,
    chunk_index: int,
    embedding: list[float],
    timestamp_start: float | None = None,
    timestamp_end: float | None = None,
) -> int:
    """Create a chunk with embedding and return its ID."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chunks (item_id, content, chunk_index, timestamp_start, timestamp_end, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (item_id, content, chunk_index, timestamp_start, timestamp_end, embedding),
            )
            result = cur.fetchone()
            conn.commit()
            return result[0]


def create_chunks_batch(chunks: list[dict]) -> list[int]:
    """Create multiple chunks in a single transaction."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            ids = []
            for chunk in chunks:
                cur.execute(
                    """
                    INSERT INTO chunks (item_id, content, chunk_index, timestamp_start, timestamp_end, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        chunk["item_id"],
                        chunk["content"],
                        chunk["chunk_index"],
                        chunk.get("timestamp_start"),
                        chunk.get("timestamp_end"),
                        chunk["embedding"],
                    ),
                )
                ids.append(cur.fetchone()[0])
            conn.commit()
            return ids


def delete_chunk(id: int) -> bool:
    """Delete a chunk by ID."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE id = %s", (id,))
            conn.commit()
            return cur.rowcount > 0


def delete_chunks_for_item(item_id: int) -> int:
    """Delete all chunks for an item. Returns number of chunks deleted."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE item_id = %s", (item_id,))
            conn.commit()
            return cur.rowcount


def get_item_content(item_id: int) -> str | None:
    """Get the concatenated content of all chunks for an item.

    Returns the full text content by joining all chunks in order.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT content FROM chunks
                WHERE item_id = %s
                ORDER BY chunk_index
                """,
                (item_id,),
            )
            rows = cur.fetchall()
            if not rows:
                return None
            return "\n\n".join(row[0] for row in rows)


def search_chunks(query_embedding: list[float], limit: int = 10) -> list[dict]:
    """Search for similar chunks using cosine similarity."""
    embedding_array = np.array(query_embedding)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    c.id, c.content, c.chunk_index, c.timestamp_start, c.timestamp_end,
                    i.title, i.url, i.type,
                    1 - (c.embedding <=> %s) AS similarity
                FROM chunks c
                JOIN items i ON c.item_id = i.id
                ORDER BY c.embedding <=> %s
                LIMIT %s
                """,
                (embedding_array, embedding_array, limit),
            )
            rows = cur.fetchall()
            return [
                {
                    "chunk_id": row[0],
                    "content": row[1],
                    "chunk_index": row[2],
                    "timestamp_start": row[3],
                    "timestamp_end": row[4],
                    "item_title": row[5],
                    "item_url": row[6],
                    "item_type": row[7],
                    "similarity": row[8],
                }
                for row in rows
            ]


# --- Index ---

def get_items_with_chunks(item_id: int | None = None) -> list[dict]:
    """Get items with their chunks for index generation."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            if item_id:
                cur.execute(
                    "SELECT id, type, title, url, metadata FROM items WHERE id = %s",
                    (item_id,),
                )
            else:
                cur.execute("SELECT id, type, title, url, metadata FROM items ORDER BY id")
            items = []
            for row in cur.fetchall():
                item = {
                    "id": row[0],
                    "type": row[1],
                    "title": row[2],
                    "url": row[3],
                    "metadata": row[4] or {},
                }
                # Get chunks for this item
                cur.execute(
                    """
                    SELECT chunk_index, content, timestamp_start, timestamp_end
                    FROM chunks
                    WHERE item_id = %s
                    ORDER BY chunk_index
                    """,
                    (item["id"],),
                )
                item["chunks"] = [
                    {
                        "chunk_index": c[0],
                        "content": c[1],
                        "timestamp_start": c[2],
                        "timestamp_end": c[3],
                    }
                    for c in cur.fetchall()
                ]
                items.append(item)
            return items


def get_items_grouped_by_source() -> dict[str, list[dict]]:
    """Get items grouped by source/channel for index generation."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Get all items with source info
            cur.execute(
                """
                SELECT i.id, i.type, i.title, i.url, i.metadata,
                       COALESCE(s.name, i.metadata->>'channel', 'Unknown') as source_name
                FROM items i
                LEFT JOIN sources s ON i.source_id = s.id
                ORDER BY source_name, i.id
                """
            )

            grouped = {}
            for row in cur.fetchall():
                item = {
                    "id": row[0],
                    "type": row[1],
                    "title": row[2],
                    "url": row[3],
                    "metadata": row[4] or {},
                }
                source_name = row[5]

                # Get chunks for this item
                cur.execute(
                    """
                    SELECT chunk_index, content, timestamp_start, timestamp_end
                    FROM chunks
                    WHERE item_id = %s
                    ORDER BY chunk_index
                    """,
                    (item["id"],),
                )
                item["chunks"] = [
                    {
                        "chunk_index": c[0],
                        "content": c[1],
                        "timestamp_start": c[2],
                        "timestamp_end": c[3],
                    }
                    for c in cur.fetchall()
                ]

                if source_name not in grouped:
                    grouped[source_name] = []
                grouped[source_name].append(item)

            return grouped


# --- Skills ---

def create_skill(
    name: str,
    type: str,
    content: str,
    description: str | None = None,
    metadata: dict | None = None,
) -> int:
    """Create a skill and return its ID."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO skills (name, type, content, description, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (name, type, content, description, Jsonb(metadata) if metadata else None),
            )
            result = cur.fetchone()
            conn.commit()
            return result[0]


def get_skill(name: str) -> dict | None:
    """Get a skill by name."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, name, type, description, content, metadata, created_at, updated_at FROM skills WHERE name = %s",
                (name,),
            )
            row = cur.fetchone()
            if row:
                return {
                    "id": row[0],
                    "name": row[1],
                    "type": row[2],
                    "description": row[3],
                    "content": row[4],
                    "metadata": row[5],
                    "created_at": row[6],
                    "updated_at": row[7],
                }
            return None


def list_skills() -> list[dict]:
    """List all skills."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, name, type, description, created_at FROM skills ORDER BY name"
            )
            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "type": row[2],
                    "description": row[3],
                    "created_at": row[4],
                }
                for row in cur.fetchall()
            ]


def update_skill(name: str, content: str, description: str | None = None) -> bool:
    """Update a skill's content."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE skills
                SET content = %s, description = COALESCE(%s, description), updated_at = NOW()
                WHERE name = %s
                """,
                (content, description, name),
            )
            conn.commit()
            return cur.rowcount > 0


def delete_skill(name: str) -> bool:
    """Delete a skill by name."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM skills WHERE name = %s", (name,))
            conn.commit()
            return cur.rowcount > 0


# --- Projects ---

def create_project(
    name: str,
    description: str | None = None,
    status: str = "active",
    facet_about: list | None = None,
    facet_uses: list | None = None,
    facet_needs: list | None = None,
    metadata: dict | None = None,
) -> int:
    """Create a project and return its ID."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO projects (name, description, status, facet_about, facet_uses, facet_needs, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    name,
                    description,
                    status,
                    Jsonb(facet_about or []),
                    Jsonb(facet_uses or []),
                    Jsonb(facet_needs or []),
                    Jsonb(metadata) if metadata else None,
                ),
            )
            result = cur.fetchone()
            conn.commit()
            return result[0]


def get_project(name: str) -> dict | None:
    """Get a project by name."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, description, status, facet_about, facet_uses, facet_needs,
                       beads_epic_id, metadata, created_at, updated_at
                FROM projects WHERE name = %s
                """,
                (name,),
            )
            row = cur.fetchone()
            if row:
                return {
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "status": row[3],
                    "facet_about": row[4] or [],
                    "facet_uses": row[5] or [],
                    "facet_needs": row[6] or [],
                    "beads_epic_id": row[7],
                    "metadata": row[8],
                    "created_at": row[9],
                    "updated_at": row[10],
                }
            return None


def get_project_by_id(id: int) -> dict | None:
    """Get a project by ID."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, description, status, facet_about, facet_uses, facet_needs,
                       beads_epic_id, metadata, created_at, updated_at
                FROM projects WHERE id = %s
                """,
                (id,),
            )
            row = cur.fetchone()
            if row:
                return {
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "status": row[3],
                    "facet_about": row[4] or [],
                    "facet_uses": row[5] or [],
                    "facet_needs": row[6] or [],
                    "beads_epic_id": row[7],
                    "metadata": row[8],
                    "created_at": row[9],
                    "updated_at": row[10],
                }
            return None


def list_projects(status: str | None = None) -> list[dict]:
    """List all projects, optionally filtered by status."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            if status:
                cur.execute(
                    """
                    SELECT id, name, description, status, facet_about, facet_uses, facet_needs,
                           beads_epic_id, metadata, created_at, updated_at
                    FROM projects WHERE status = %s ORDER BY name
                    """,
                    (status,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, name, description, status, facet_about, facet_uses, facet_needs,
                           beads_epic_id, metadata, created_at, updated_at
                    FROM projects ORDER BY name
                    """
                )
            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "status": row[3],
                    "facet_about": row[4] or [],
                    "facet_uses": row[5] or [],
                    "facet_needs": row[6] or [],
                    "beads_epic_id": row[7],
                    "metadata": row[8],
                    "created_at": row[9],
                    "updated_at": row[10],
                }
                for row in cur.fetchall()
            ]


def update_project(
    name: str,
    description: str | None = None,
    status: str | None = None,
    facet_about: list | None = None,
    facet_uses: list | None = None,
    facet_needs: list | None = None,
    metadata: dict | None = None,
) -> bool:
    """Update a project's fields. Only non-None values are updated."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE projects
                SET description = COALESCE(%s, description),
                    status = COALESCE(%s, status),
                    facet_about = COALESCE(%s, facet_about),
                    facet_uses = COALESCE(%s, facet_uses),
                    facet_needs = COALESCE(%s, facet_needs),
                    metadata = COALESCE(%s, metadata),
                    updated_at = NOW()
                WHERE name = %s
                """,
                (
                    description,
                    status,
                    Jsonb(facet_about) if facet_about is not None else None,
                    Jsonb(facet_uses) if facet_uses is not None else None,
                    Jsonb(facet_needs) if facet_needs is not None else None,
                    Jsonb(metadata) if metadata is not None else None,
                    name,
                ),
            )
            conn.commit()
            return cur.rowcount > 0


def delete_project(name: str) -> bool:
    """Delete a project by name. Links are deleted via CASCADE."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM projects WHERE name = %s", (name,))
            conn.commit()
            return cur.rowcount > 0


# --- Project Linking ---

def link_project_item(project_id: int, item_id: int) -> bool:
    """Link an item to a project."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    INSERT INTO project_items (project_id, item_id)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (project_id, item_id),
                )
                conn.commit()
                return True
            except Exception:
                return False


def unlink_project_item(project_id: int, item_id: int) -> bool:
    """Unlink an item from a project."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM project_items WHERE project_id = %s AND item_id = %s",
                (project_id, item_id),
            )
            conn.commit()
            return cur.rowcount > 0


def get_project_items(project_id: int) -> list[dict]:
    """Get all items linked to a project."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT i.id, i.source_id, i.type, i.title, i.url, i.metadata
                FROM items i
                JOIN project_items pi ON i.id = pi.item_id
                WHERE pi.project_id = %s
                ORDER BY i.id
                """,
                (project_id,),
            )
            return [
                {
                    "id": row[0],
                    "source_id": row[1],
                    "type": row[2],
                    "title": row[3],
                    "url": row[4],
                    "metadata": row[5],
                }
                for row in cur.fetchall()
            ]


def get_item_projects(item_id: int) -> list[dict]:
    """Get all projects that an item is linked to."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.id, p.name, p.description, p.status, p.facet_about,
                       p.facet_uses, p.facet_needs, p.beads_epic_id, p.metadata,
                       p.created_at, p.updated_at
                FROM projects p
                JOIN project_items pi ON p.id = pi.project_id
                WHERE pi.item_id = %s
                ORDER BY p.name
                """,
                (item_id,),
            )
            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "status": row[3],
                    "facet_about": row[4] or [],
                    "facet_uses": row[5] or [],
                    "facet_needs": row[6] or [],
                    "beads_epic_id": row[7],
                    "metadata": row[8],
                    "created_at": row[9],
                    "updated_at": row[10],
                }
                for row in cur.fetchall()
            ]


def is_item_linked_to_project(project_id: int, item_id: int) -> bool:
    """Check if an item is linked to a project."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM project_items WHERE project_id = %s AND item_id = %s",
                (project_id, item_id),
            )
            return cur.fetchone() is not None


def list_all_project_items() -> list[dict]:
    """List all project-item associations."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT pi.project_id, pi.item_id, p.name as project_name, i.title as item_title
                FROM project_items pi
                JOIN projects p ON pi.project_id = p.id
                JOIN items i ON pi.item_id = i.id
                ORDER BY p.name, i.title
                """
            )
            return [
                {
                    "project_id": row[0],
                    "item_id": row[1],
                    "project_name": row[2],
                    "item_title": row[3],
                }
                for row in cur.fetchall()
            ]


def link_project_skill(project_id: int, skill_id: int) -> bool:
    """Link a skill to a project."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    INSERT INTO project_skills (project_id, skill_id)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (project_id, skill_id),
                )
                conn.commit()
                return True
            except Exception:
                return False


def unlink_project_skill(project_id: int, skill_id: int) -> bool:
    """Unlink a skill from a project."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM project_skills WHERE project_id = %s AND skill_id = %s",
                (project_id, skill_id),
            )
            conn.commit()
            return cur.rowcount > 0


def get_project_skills(project_id: int) -> list[dict]:
    """Get all skills linked to a project."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.id, s.name, s.type, s.description, s.created_at
                FROM skills s
                JOIN project_skills ps ON s.id = ps.skill_id
                WHERE ps.project_id = %s
                ORDER BY s.name
                """,
                (project_id,),
            )
            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "type": row[2],
                    "description": row[3],
                    "created_at": row[4],
                }
                for row in cur.fetchall()
            ]


def get_skill_projects(skill_id: int) -> list[dict]:
    """Get all projects linked to a skill."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.id, p.name, p.description, p.status, p.created_at
                FROM projects p
                JOIN project_skills ps ON p.id = ps.project_id
                WHERE ps.skill_id = %s
                ORDER BY p.name
                """,
                (skill_id,),
            )
            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "status": row[3],
                    "created_at": row[4],
                }
                for row in cur.fetchall()
            ]


def list_project_skill_links() -> list[dict]:
    """List all project-skill links."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ps.project_id, ps.skill_id, p.name as project_name, s.name as skill_name
                FROM project_skills ps
                JOIN projects p ON ps.project_id = p.id
                JOIN skills s ON ps.skill_id = s.id
                ORDER BY p.name, s.name
                """
            )
            return [
                {
                    "project_id": row[0],
                    "skill_id": row[1],
                    "project_name": row[2],
                    "skill_name": row[3],
                }
                for row in cur.fetchall()
            ]


# --- Project Embeddings ---

def update_project_embeddings(
    project_id: int,
    embedding_about: list[float] | None = None,
    embedding_uses: list[float] | None = None,
    embedding_needs: list[float] | None = None,
) -> bool:
    """Update a project's facet embeddings."""
    # Convert to numpy arrays for pgvector compatibility
    emb_about = np.array(embedding_about) if embedding_about else None
    emb_uses = np.array(embedding_uses) if embedding_uses else None
    emb_needs = np.array(embedding_needs) if embedding_needs else None

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Build dynamic update to only set non-None values
            updates = []
            params = []

            if emb_about is not None:
                updates.append("embedding_about = %s")
                params.append(emb_about)
            if emb_uses is not None:
                updates.append("embedding_uses = %s")
                params.append(emb_uses)
            if emb_needs is not None:
                updates.append("embedding_needs = %s")
                params.append(emb_needs)

            if not updates:
                return False

            updates.append("updated_at = NOW()")
            params.append(project_id)

            cur.execute(
                f"""
                UPDATE projects
                SET {', '.join(updates)}
                WHERE id = %s
                """,
                tuple(params),
            )
            conn.commit()
            return cur.rowcount > 0


def discover_for_project(
    project_id: int,
    facet: str = "needs",
    limit: int = 10,
    min_similarity: float = 0.3,
) -> list[dict]:
    """
    Discover chunks that match a project's facet embedding.
    Default: find chunks that match what the project NEEDS.

    facet can be: 'about', 'uses', or 'needs'
    """
    facet_col = f"embedding_{facet}"
    if facet not in ("about", "uses", "needs"):
        raise ValueError(f"Invalid facet: {facet}")

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Get project embedding
            cur.execute(
                f"SELECT {facet_col} FROM projects WHERE id = %s",
                (project_id,),
            )
            row = cur.fetchone()
            if not row or row[0] is None:
                return []

            embedding = np.array(row[0])

            # Find similar chunks
            cur.execute(
                """
                SELECT
                    c.id, c.content, c.chunk_index, c.timestamp_start, c.timestamp_end,
                    i.id as item_id, i.title, i.url, i.type,
                    1 - (c.embedding <=> %s) AS similarity
                FROM chunks c
                JOIN items i ON c.item_id = i.id
                WHERE c.embedding IS NOT NULL
                  AND 1 - (c.embedding <=> %s) > %s
                ORDER BY c.embedding <=> %s
                LIMIT %s
                """,
                (embedding, embedding, min_similarity, embedding, limit),
            )

            return [
                {
                    "chunk_id": row[0],
                    "content": row[1],
                    "chunk_index": row[2],
                    "timestamp_start": row[3],
                    "timestamp_end": row[4],
                    "item_id": row[5],
                    "item_title": row[6],
                    "item_url": row[7],
                    "item_type": row[8],
                    "similarity": row[9],
                }
                for row in cur.fetchall()
            ]


def discover_projects_for_content(
    query_embedding: list[float],
    facet: str = "needs",
    limit: int = 5,
    min_similarity: float = 0.3,
) -> list[dict]:
    """
    Find projects whose facet matches the given embedding.
    Use case: when harvesting new content, find which projects could benefit.

    facet can be: 'about', 'uses', or 'needs'
    """
    facet_col = f"embedding_{facet}"
    if facet not in ("about", "uses", "needs"):
        raise ValueError(f"Invalid facet: {facet}")

    embedding_array = np.array(query_embedding)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    p.id, p.name, p.description, p.status,
                    p.facet_about, p.facet_uses, p.facet_needs,
                    1 - (p.{facet_col} <=> %s) AS similarity
                FROM projects p
                WHERE p.{facet_col} IS NOT NULL
                  AND p.status = 'active'
                  AND 1 - (p.{facet_col} <=> %s) > %s
                ORDER BY p.{facet_col} <=> %s
                LIMIT %s
                """,
                (embedding_array, embedding_array, min_similarity, embedding_array, limit),
            )

            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "status": row[3],
                    "facet_about": row[4] or [],
                    "facet_uses": row[5] or [],
                    "facet_needs": row[6] or [],
                    "similarity": row[7],
                }
                for row in cur.fetchall()
            ]


def get_projects_needing_embeddings() -> list[dict]:
    """Get projects that have facets but no embeddings (need to be embedded)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, facet_about, facet_uses, facet_needs
                FROM projects
                WHERE (
                    (facet_about IS NOT NULL AND jsonb_array_length(facet_about) > 0 AND embedding_about IS NULL)
                    OR (facet_uses IS NOT NULL AND jsonb_array_length(facet_uses) > 0 AND embedding_uses IS NULL)
                    OR (facet_needs IS NOT NULL AND jsonb_array_length(facet_needs) > 0 AND embedding_needs IS NULL)
                )
                ORDER BY id
                """
            )
            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "facet_about": row[2] or [],
                    "facet_uses": row[3] or [],
                    "facet_needs": row[4] or [],
                }
                for row in cur.fetchall()
            ]


# --- Project Outputs ---

def create_project_output(
    project_id: int,
    name: str,
    type: str | None = None,
    path: str | None = None,
    metadata: dict | None = None,
) -> int:
    """Create a project output and return its ID."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO project_outputs (project_id, name, type, path, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (project_id, name, type, path, Jsonb(metadata) if metadata else None),
            )
            result = cur.fetchone()
            conn.commit()
            return result[0]


def get_project_output(id: int) -> dict | None:
    """Get a project output by ID."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, project_id, name, type, path, metadata, created_at
                FROM project_outputs WHERE id = %s
                """,
                (id,),
            )
            row = cur.fetchone()
            if row:
                return {
                    "id": row[0],
                    "project_id": row[1],
                    "name": row[2],
                    "type": row[3],
                    "path": row[4],
                    "metadata": row[5],
                    "created_at": row[6],
                }
            return None


def list_project_outputs(project_id: int | None = None) -> list[dict]:
    """List all project outputs, optionally filtered by project_id."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            if project_id is not None:
                cur.execute(
                    """
                    SELECT id, project_id, name, type, path, metadata, created_at
                    FROM project_outputs WHERE project_id = %s ORDER BY id
                    """,
                    (project_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, project_id, name, type, path, metadata, created_at
                    FROM project_outputs ORDER BY id
                    """
                )
            return [
                {
                    "id": row[0],
                    "project_id": row[1],
                    "name": row[2],
                    "type": row[3],
                    "path": row[4],
                    "metadata": row[5],
                    "created_at": row[6],
                }
                for row in cur.fetchall()
            ]


def update_project_output(
    id: int,
    name: str | None = None,
    type: str | None = None,
    path: str | None = None,
    metadata: dict | None = None,
) -> dict | None:
    """Update a project output and return the updated output."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE project_outputs
                SET name = COALESCE(%s, name),
                    type = COALESCE(%s, type),
                    path = COALESCE(%s, path),
                    metadata = COALESCE(%s, metadata)
                WHERE id = %s
                RETURNING id, project_id, name, type, path, metadata, created_at
                """,
                (name, type, path, Jsonb(metadata) if metadata else None, id),
            )
            row = cur.fetchone()
            conn.commit()
            if row:
                return {
                    "id": row[0],
                    "project_id": row[1],
                    "name": row[2],
                    "type": row[3],
                    "path": row[4],
                    "metadata": row[5],
                    "created_at": row[6],
                }
            return None


def delete_project_output(id: int) -> bool:
    """Delete a project output by ID."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM project_outputs WHERE id = %s", (id,))
            conn.commit()
            return cur.rowcount > 0


# --- Stats ---

def get_stats() -> dict:
    """Get statistics about the knowledge base."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM sources")
            sources = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM items")
            items = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM chunks")
            chunks = cur.fetchone()[0]
            return {
                "sources": sources,
                "items": items,
                "chunks": chunks,
            }
