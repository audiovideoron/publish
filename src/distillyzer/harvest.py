"""Harvest content from YouTube and GitHub."""

import json
import re
import subprocess
import tempfile
from pathlib import Path

from git import Repo

from . import db


def parse_youtube_url(url: str) -> dict:
    """Parse YouTube URL to extract video ID and type."""
    # Video patterns
    video_patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"youtube\.com/embed/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in video_patterns:
        match = re.search(pattern, url)
        if match:
            return {"type": "video", "id": match.group(1)}

    # Channel patterns
    channel_patterns = [
        r"youtube\.com/@([a-zA-Z0-9_-]+)",
        r"youtube\.com/channel/([a-zA-Z0-9_-]+)",
        r"youtube\.com/c/([a-zA-Z0-9_-]+)",
    ]
    for pattern in channel_patterns:
        match = re.search(pattern, url)
        if match:
            return {"type": "channel", "id": match.group(1)}

    return {"type": "unknown", "id": None}


def get_video_info(url: str) -> dict:
    """Get video metadata using yt-dlp."""
    result = subprocess.run(
        ["yt-dlp", "--dump-json", "--no-download", url],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")
    return json.loads(result.stdout)


def download_audio(url: str, output_dir: Path) -> Path:
    """Download audio from YouTube video using yt-dlp. Uses 64k bitrate to stay under Whisper's 25MB limit."""
    output_template = str(output_dir / "%(id)s.%(ext)s")
    result = subprocess.run(
        [
            "yt-dlp",
            "-x",  # Extract audio
            "--audio-format", "mp3",
            "--audio-quality", "9",  # Lower quality = smaller file (64kbps)
            "--postprocessor-args", "ffmpeg:-ac 1 -ar 16000",  # Mono, 16kHz (fine for speech)
            "-o", output_template,
            url,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")

    # Find the downloaded file
    for f in output_dir.iterdir():
        if f.suffix == ".mp3":
            return f
    raise RuntimeError("Audio file not found after download")


def search_youtube(query: str, limit: int = 10) -> list[dict]:
    """Search YouTube using yt-dlp."""
    result = subprocess.run(
        [
            "yt-dlp",
            "--dump-json",
            "--flat-playlist",
            "--no-download",
            f"ytsearch{limit}:{query}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp search failed: {result.stderr}")

    videos = []
    for line in result.stdout.strip().split("\n"):
        if line:
            data = json.loads(line)
            videos.append({
                "id": data.get("id"),
                "title": data.get("title"),
                "url": f"https://youtube.com/watch?v={data.get('id')}",
                "channel": data.get("channel"),
                "duration": data.get("duration"),
            })
    return videos


def harvest_video(url: str) -> dict:
    """
    Harvest a YouTube video: download audio, get metadata, save to DB.
    Returns dict with item_id and audio_path.
    """
    # Check if already harvested
    existing = db.get_item_by_url(url)
    if existing:
        return {"item_id": existing["id"], "status": "already_exists", "title": existing["title"]}

    # Get video info
    info = get_video_info(url)
    title = info.get("title", "Unknown")
    channel = info.get("channel", "Unknown")
    duration = info.get("duration", 0)

    # Download audio to temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="distillyzer_"))
    audio_path = download_audio(url, temp_dir)

    # Get or create source for channel
    channel_url = info.get("channel_url", f"https://youtube.com/@{channel}")
    source_id = db.get_or_create_source(
        type="youtube_channel",
        name=channel,
        url=channel_url,
    )

    # Create item in DB
    item_id = db.create_item(
        source_id=source_id,
        type="video",
        title=title,
        url=url,
        metadata={
            "channel": channel,
            "duration": duration,
            "video_id": info.get("id"),
        },
    )

    return {
        "item_id": item_id,
        "audio_path": str(audio_path),
        "title": title,
        "channel": channel,
        "duration": duration,
        "status": "downloaded",
    }


def harvest_channel(channel_url: str, limit: int = 50) -> list[dict]:
    """Harvest videos from a YouTube channel."""
    # Get channel videos
    result = subprocess.run(
        [
            "yt-dlp",
            "--dump-json",
            "--flat-playlist",
            "--no-download",
            "--playlist-end", str(limit),
            channel_url,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")

    videos = []
    for line in result.stdout.strip().split("\n"):
        if line:
            data = json.loads(line)
            video_url = f"https://youtube.com/watch?v={data.get('id')}"
            videos.append({
                "id": data.get("id"),
                "title": data.get("title"),
                "url": video_url,
            })
    return videos


# --- GitHub ---

def parse_github_url(url: str) -> dict:
    """Parse GitHub URL to extract owner and repo."""
    pattern = r"github\.com/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_.-]+)"
    match = re.search(pattern, url)
    if match:
        return {"owner": match.group(1), "repo": match.group(2).rstrip(".git")}
    return {"owner": None, "repo": None}


def harvest_repo(url: str, clone_dir: Path | None = None) -> dict:
    """
    Clone a GitHub repo and index its files.
    Returns dict with source_id and file count.
    """
    parsed = parse_github_url(url)
    if not parsed["owner"]:
        raise ValueError(f"Invalid GitHub URL: {url}")

    repo_name = f"{parsed['owner']}/{parsed['repo']}"

    # Check if already harvested
    existing = db.get_source_by_url(url)
    if existing:
        return {"source_id": existing["id"], "status": "already_exists", "name": repo_name}

    # Clone directory
    if clone_dir is None:
        clone_dir = Path(tempfile.mkdtemp(prefix="distillyzer_repo_"))
    repo_path = clone_dir / parsed["repo"]

    # Clone repo
    Repo.clone_from(url, repo_path, depth=1)

    # Create source in DB
    source_id = db.create_source(
        type="github_repo",
        name=repo_name,
        url=url,
        metadata={"owner": parsed["owner"], "repo": parsed["repo"]},
    )

    # Index code files
    code_extensions = {".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c", ".h", ".md", ".txt"}
    files_indexed = 0
    file_items = []  # Store file data for embedding

    for file_path in repo_path.rglob("*"):
        if file_path.is_file() and file_path.suffix in code_extensions:
            # Skip hidden dirs, node_modules, etc.
            if any(part.startswith(".") or part == "node_modules" for part in file_path.parts):
                continue

            rel_path = file_path.relative_to(repo_path)
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                if len(content) > 100:  # Skip tiny files
                    item_id = db.create_item(
                        source_id=source_id,
                        type="code_file",
                        title=str(rel_path),
                        url=f"{url}/blob/main/{rel_path}",
                        metadata={"extension": file_path.suffix, "size": len(content)},
                    )
                    files_indexed += 1
                    # Store for embedding
                    file_items.append({
                        "item_id": item_id,
                        "path": str(rel_path),
                        "content": content,
                        "extension": file_path.suffix,
                    })
            except Exception:
                pass

    return {
        "source_id": source_id,
        "name": repo_name,
        "files_indexed": files_indexed,
        "file_items": file_items,  # Include file data for embedding
        "repo_path": str(repo_path),
        "status": "cloned",
    }
