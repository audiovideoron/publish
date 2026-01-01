"""Database operations for Distillyzer."""

import os
from contextlib import contextmanager
from typing import Generator

import numpy as np
import psycopg
from psycopg.types.json import Jsonb
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/distillyzer")


@contextmanager
def get_connection() -> Generator[psycopg.Connection, None, None]:
    """Get a database connection with pgvector support."""
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    try:
        yield conn
    finally:
        conn.close()


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
