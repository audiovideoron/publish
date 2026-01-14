# Publishing

Build a personal knowledge base from YouTube videos and GitHub repos. Transcribe, embed, and semantically search everything you care about.

## Why

YouTube's search is optimized for engagement, not learning. You watch a great 2-hour tutorial, forget where they explained that one thing, and spend 20 minutes scrubbing. GitHub repos have answers buried in code you'll never find with grep.

Publishing extracts knowledge from video and code, chunks it with timestamps, embeds it in a vector database, and lets you query it conversationally.

## How It Works

```
YouTube URL → yt-dlp → Whisper API → Chunks with timestamps → Embeddings → PostgreSQL/pgvector
GitHub URL  → git clone → Parse files → Chunks → Embeddings → PostgreSQL/pgvector
                                                                      ↓
                                              Your question → Embedding → Similarity search → Claude → Answer with sources
```

## Quick Start

```bash
# Install
git clone https://github.com/audiovideoron/publishing
cd publishing
uv venv && source .venv/bin/activate
uv pip install -e .

# Set up PostgreSQL with pgvector
createdb publishing
psql publishing < schema.sql

# Configure
cp .env.example .env
# Edit .env with your API keys:
#   OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY (optional)

# Harvest a video
pub harvest https://youtube.com/watch?v=...

# Query your knowledge
pub query "how does the auth system work?"

# Or chat interactively
pub chat
```

## Commands

### Core

| Command | Description |
|---------|-------------|
| `pub harvest <url>` | Harvest a YouTube video, GitHub repo, or article |
| `pub query "question"` | Semantic search with Claude-generated answers |
| `pub chat` | Interactive conversation with your knowledge base |
| `pub stats` | Show what's in your knowledge base |
| `pub search "topic"` | Search YouTube for videos on a topic |
| `pub harvest-channel <url>` | List videos from a channel for selective harvesting |

### Knowledge Extraction

| Command | Description |
|---------|-------------|
| `pub extract "topic"` | Extract patterns, prompts, checklists from your knowledge |
| `pub demo "topic"` | Build a hello-world project from a lesson topic |
| `pub visualize "concept"` | Generate diagrams or artistic imagery via Gemini |
| `pub index` | Generate HTML index with timestamp links |
| `pub embed [item_id]` | Re-embed items (after model/chunking changes) |

### Project Management

| Command | Description |
|---------|-------------|
| `pub project create` | Create a project with faceted organization |
| `pub project discover` | Find knowledge matching project facets |
| `pub crosspolinate "topic"` | Find projects that could benefit from a topic |
| `pub suggest youtube` | Suggest videos based on project facets |

### Data Management

| Command | Description |
|---------|-------------|
| `pub artifacts list/show/apply` | Manage extracted artifacts |
| `pub skills list/create/update` | Manage presentation skills |
| `pub sources list/delete` | Manage channels and repos |
| `pub items list/delete` | Manage harvested content |

## Example

```bash
$ pub query "What are the core principles of agentic coding?"

╭─────────────────────── Answer ───────────────────────╮
│ Based on the sources, agentic coding centers on      │
│ "The Core Four": Context, Model, Prompt, and Tools.  │
│ The key shift is from writing code to orchestrating  │
│ systems that write code on your behalf...            │
╰──────────────────────────────────────────────────────╯

Sources:
  1. TAC: Hello Agentic Coding @ 13:17 (sim: 0.63)
  2. TAC: Hello Agentic Coding @ 20:05 (sim: 0.61)
```

## Tech Stack

- **yt-dlp** - YouTube download
- **OpenAI Whisper API** - Transcription with timestamps
- **OpenAI text-embedding-3-small** - 1536-dim embeddings
- **PostgreSQL + pgvector** - Vector similarity search
- **Claude API** - Answer generation and artifact extraction
- **Gemini API** - Image generation for visualizations
- **Typer + Rich** - CLI

## Requirements

- Python 3.12+
- PostgreSQL with pgvector extension
- yt-dlp (`brew install yt-dlp`)
- ffmpeg (`brew install ffmpeg`)
- OpenAI API key (for Whisper + embeddings)
- Anthropic API key (for Claude queries)
- Google API key (optional, for Gemini visualizations)

## Privacy

All data stays local. Videos are transcribed via API but the transcripts and embeddings live in your PostgreSQL database. Nothing is stored remotely or shared.

## License

MIT
