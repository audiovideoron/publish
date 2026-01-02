# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install dependencies
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run all tests
pytest

# Run single test file
pytest tests/test_harvest.py

# Run specific test
pytest tests/test_harvest.py::test_function_name -v

# Lint
ruff check src/

# Lint with auto-fix
ruff check src/ --fix
```

## CLI Usage

The CLI is invoked via `dz`:
```bash
dz harvest <url>      # Ingest YouTube video, GitHub repo, or article
dz query "question"   # Semantic search with Claude answer
dz chat               # Interactive conversation mode
dz stats              # Knowledge base statistics
dz demo "topic"       # Generate hello-world project from lessons
dz visualize "concept"  # Generate diagram via Gemini
```

## Architecture

### Data Flow
```
Content Source → Harvest → Transcribe/Parse → Chunk → Embed → PostgreSQL/pgvector
                                                                      ↓
User Query → Embed → Similarity Search → Top Chunks → Claude → Answer with Sources
```

### Core Modules (src/distillyzer/)

- **cli.py** - Typer CLI entry point. Sub-apps: `artifacts`, `skills`, `project`, `sources`, `items`, `suggest`
- **harvest.py** - Downloads content (yt-dlp for video, GitPython for repos, trafilatura for articles)
- **transcribe.py** - Whisper API transcription with timestamp preservation
- **embed.py** - OpenAI text-embedding-3-small (1536 dimensions), chunking logic
- **db.py** - PostgreSQL + pgvector operations, connection pooling
- **query.py** - Semantic search + Claude answer synthesis
- **artifacts.py** - Extract patterns from knowledge, scaffold demo projects
- **visualize.py** - Gemini image generation for concepts

### Database Schema

Primary tables: `items` (harvested content), `chunks` (embedded segments with vectors), `projects` (faceted organization), `artifacts` (extracted patterns)

Vectors stored as 1536-dimensional embeddings using pgvector extension.

## Key Conventions

### Project Output
- Commands that produce project output (`demo`, `artifacts scaffold`) default to `~/projects/<project_name>`
- Directories are created lazily - only when output is actually produced, not before

### Issue Tracking (bd/beads)
This project uses `bd` for issue tracking:
```bash
bd ready                              # Find available work
bd update <id> --status in_progress   # Claim work
bd close <id>                         # Complete work
bd sync                               # Sync with git
```

### Session Completion
Work is not complete until pushed:
```bash
git add <files>
bd sync
git commit -m "..."
git push
```

## Environment Variables

Required in `.env`:
- `DATABASE_URL` - PostgreSQL connection string
- `OPENAI_API_KEY` - For Whisper transcription and embeddings
- `ANTHROPIC_API_KEY` - For Claude query answers
- `GOOGLE_API_KEY` - For Gemini image generation (optional)
