"""Content quality scoring for YouTube videos and GitHub repositories.

Provides scoring functions to evaluate educational quality and filter content
for Distillyzer's recommendation system.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# =============================================================================
# YouTube Video Scoring
# =============================================================================

# Educational keywords that increase score
EDUCATIONAL_KEYWORDS = [
    "tutorial",
    "explained",
    "how to",
    "guide",
    "learn",
    "course",
    "lesson",
    "introduction",
    "intro to",
    "getting started",
    "basics",
    "fundamentals",
    "deep dive",
    "walkthrough",
    "step by step",
    "masterclass",
    "workshop",
    "crash course",
]

# Negative keywords that decrease score (non-educational content)
NEGATIVE_VIDEO_KEYWORDS = [
    "gameplay",
    "let's play",
    "lets play",
    "playthrough",
    "vlog",
    "meme",
    "memes",
    "shorts",
    "tiktok",
    "funny",
    "prank",
    "reaction",
    "unboxing",
    "haul",
    "asmr",
    "mukbang",
    "challenge",
    "compilation",
    "montage",
    "stream highlights",
    "best moments",
    "fails",
    "try not to",
]


@dataclass
class VideoScoreBreakdown:
    """Detailed breakdown of a YouTube video's quality score."""
    total: int
    educational_score: int
    engagement_score: int
    negative_penalty: int
    educational_keywords_found: list[str]
    negative_keywords_found: list[str]
    is_short: bool
    view_count: int | None
    like_ratio: float | None

    def to_dict(self) -> dict:
        """Convert breakdown to dictionary."""
        return {
            "total": self.total,
            "educational_score": self.educational_score,
            "engagement_score": self.engagement_score,
            "negative_penalty": self.negative_penalty,
            "educational_keywords_found": self.educational_keywords_found,
            "negative_keywords_found": self.negative_keywords_found,
            "is_short": self.is_short,
            "view_count": self.view_count,
            "like_ratio": self.like_ratio,
        }


def score_video(
    title: str,
    view_count: int | None = None,
    like_count: int | None = None,
    duration: int | None = None,
    description: str | None = None,
) -> VideoScoreBreakdown:
    """
    Score a YouTube video for educational quality.

    Args:
        title: Video title
        view_count: Number of views (optional)
        like_count: Number of likes (optional)
        duration: Video duration in seconds (optional)
        description: Video description (optional)

    Returns:
        VideoScoreBreakdown with total score (0-100) and component breakdown
    """
    text_to_check = title.lower()
    if description:
        text_to_check += " " + description.lower()

    # Calculate educational score (0-40 points)
    educational_score, educational_found = _calculate_educational_score(text_to_check)

    # Calculate engagement score (0-40 points)
    engagement_score, like_ratio = _calculate_video_engagement_score(view_count, like_count)

    # Calculate negative penalty (0-50 points deduction)
    negative_penalty, negative_found = _calculate_video_negative_penalty(text_to_check)

    # Check if video is a "short" (under 60 seconds)
    is_short = duration is not None and duration < 60
    short_penalty = 20 if is_short else 0

    # Base score starts at 20 (neutral content gets some points)
    base_score = 20
    total = base_score + educational_score + engagement_score - negative_penalty - short_penalty
    total = max(0, min(100, total))

    return VideoScoreBreakdown(
        total=total,
        educational_score=educational_score,
        engagement_score=engagement_score,
        negative_penalty=negative_penalty + short_penalty,
        educational_keywords_found=educational_found,
        negative_keywords_found=negative_found,
        is_short=is_short,
        view_count=view_count,
        like_ratio=like_ratio,
    )


def _calculate_educational_score(text: str) -> tuple[int, list[str]]:
    """Calculate educational score based on keyword presence."""
    found_keywords = []
    for keyword in EDUCATIONAL_KEYWORDS:
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            found_keywords.append(keyword)
    score = min(40, len(found_keywords) * 10)
    return score, found_keywords


def _calculate_video_engagement_score(
    view_count: int | None,
    like_count: int | None,
) -> tuple[int, float | None]:
    """Calculate engagement score based on views and likes."""
    score = 0
    like_ratio = None

    if view_count is not None:
        if view_count >= 1_000_000:
            score += 20
        elif view_count >= 100_000:
            score += 15
        elif view_count >= 10_000:
            score += 10
        elif view_count >= 1_000:
            score += 5

    if view_count and like_count and view_count > 0:
        like_ratio = like_count / view_count
        if like_ratio >= 0.05:
            score += 20
        elif like_ratio >= 0.03:
            score += 15
        elif like_ratio >= 0.02:
            score += 10
        elif like_ratio >= 0.01:
            score += 5

    return score, like_ratio


def _calculate_video_negative_penalty(text: str) -> tuple[int, list[str]]:
    """Calculate penalty for non-educational content indicators."""
    found_keywords = []
    for keyword in NEGATIVE_VIDEO_KEYWORDS:
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            found_keywords.append(keyword)
    penalty = min(50, len(found_keywords) * 15)
    return penalty, found_keywords


def filter_videos_by_score(
    videos: list[dict],
    min_score: int = 30,
    score_field: str = "score",
) -> list[dict]:
    """Filter and sort videos by quality score."""
    scored_videos = []
    for video in videos:
        breakdown = score_video(
            title=video.get("title", ""),
            view_count=video.get("view_count"),
            like_count=video.get("like_count"),
            duration=video.get("duration"),
            description=video.get("description"),
        )
        if breakdown.total >= min_score:
            video_with_score = video.copy()
            video_with_score[score_field] = breakdown.total
            video_with_score["score_breakdown"] = breakdown.to_dict()
            scored_videos.append(video_with_score)
    scored_videos.sort(key=lambda v: v[score_field], reverse=True)
    return scored_videos


def is_educational(title: str, description: str | None = None) -> bool:
    """Quick check if a video appears to be educational."""
    breakdown = score_video(title=title, description=description)
    return breakdown.total >= 40


# =============================================================================
# GitHub Repository Scoring
# =============================================================================

# Joke/non-serious repo indicators
JOKE_INDICATORS = [
    "awesome-list",
    "interview-questions",
    "cheatsheet",
    "dotfiles",
    "my-resume",
    "hello-world",
    "test-repo",
    "learning-",
    "-tutorial",
    "homework",
    "assignment",
    "coursework",
    "lol",
    "meme",
    "joke",
    "troll",
    "for-fun",
    "playground",
]

# Common README section headings that indicate quality
QUALITY_SECTIONS = [
    r"##?\s*installation",
    r"##?\s*usage",
    r"##?\s*getting\s*started",
    r"##?\s*features",
    r"##?\s*api",
    r"##?\s*documentation",
    r"##?\s*examples?",
    r"##?\s*configuration",
    r"##?\s*contributing",
    r"##?\s*license",
    r"##?\s*requirements",
    r"##?\s*dependencies",
    r"##?\s*changelog",
    r"##?\s*roadmap",
]

# Badge patterns in markdown
BADGE_PATTERNS = [
    r"\[!\[.*?\]\(.*?badge.*?\)\]",
    r"!\[.*?\]\(.*?shields\.io.*?\)",
    r"!\[.*?\]\(.*?badge.*?\)",
    r"!\[build\s*status\]",
    r"!\[coverage\]",
    r"!\[npm\s*version\]",
    r"!\[pypi\s*version\]",
    r"!\[license\]",
    r"!\[downloads\]",
]


@dataclass
class RepoScoreBreakdown:
    """Detailed breakdown of a GitHub repository's quality score."""
    readme_score: int = 0
    docs_score: int = 0
    engagement_score: int = 0
    negative_penalty: int = 0

    readme_length: int = 0
    readme_has_badges: bool = False
    readme_sections_count: int = 0
    readme_has_code_examples: bool = False

    has_docs_folder: bool = False
    has_examples_folder: bool = False
    has_wiki: bool = False

    star_count: int = 0
    recent_commits: int = 0
    days_since_last_commit: Optional[int] = None

    is_fork: bool = False
    is_empty_readme: bool = False
    is_joke_repo: bool = False
    joke_indicators_found: list = field(default_factory=list)

    @property
    def total_score(self) -> int:
        """Calculate total score (0-100)."""
        raw_score = (
            self.readme_score * 0.35 +
            self.docs_score * 0.20 +
            self.engagement_score * 0.45 +
            self.negative_penalty
        )
        return max(0, min(100, int(raw_score)))

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_score": self.total_score,
            "components": {
                "readme_score": self.readme_score,
                "docs_score": self.docs_score,
                "engagement_score": self.engagement_score,
                "negative_penalty": self.negative_penalty,
            },
            "readme_details": {
                "length": self.readme_length,
                "has_badges": self.readme_has_badges,
                "sections_count": self.readme_sections_count,
                "has_code_examples": self.readme_has_code_examples,
            },
            "docs_details": {
                "has_docs_folder": self.has_docs_folder,
                "has_examples_folder": self.has_examples_folder,
                "has_wiki": self.has_wiki,
            },
            "engagement_details": {
                "star_count": self.star_count,
                "recent_commits": self.recent_commits,
                "days_since_last_commit": self.days_since_last_commit,
            },
            "negative_signals": {
                "is_fork": self.is_fork,
                "is_empty_readme": self.is_empty_readme,
                "is_joke_repo": self.is_joke_repo,
                "joke_indicators_found": self.joke_indicators_found,
            },
        }


def score_readme(readme_content: str) -> tuple[int, dict]:
    """Score README quality on a 0-100 scale."""
    if not readme_content or not readme_content.strip():
        return 0, {"length": 0, "has_badges": False, "sections_count": 0, "has_code_examples": False}

    score = 0
    content_lower = readme_content.lower()
    length = len(readme_content)

    # Length scoring (0-30 points)
    if length >= 5000:
        score += 30
    elif length >= 2000:
        score += 25
    elif length >= 1000:
        score += 20
    elif length >= 500:
        score += 15
    elif length >= 200:
        score += 10
    elif length >= 50:
        score += 5

    # Badge detection (0-15 points)
    has_badges = any(re.search(p, readme_content, re.IGNORECASE) for p in BADGE_PATTERNS)
    if has_badges:
        score += 15

    # Section detection (0-35 points)
    sections_found = sum(1 for p in QUALITY_SECTIONS if re.search(p, content_lower))
    if sections_found >= 6:
        score += 35
    elif sections_found >= 4:
        score += 25
    elif sections_found >= 2:
        score += 15
    elif sections_found >= 1:
        score += 8

    # Code examples (0-20 points)
    code_blocks = re.findall(r"```[\w]*\n[\s\S]*?```", readme_content)
    has_code_examples = len(code_blocks) >= 1
    if len(code_blocks) >= 3:
        score += 20
    elif len(code_blocks) >= 1:
        score += 12

    return score, {
        "length": length,
        "has_badges": has_badges,
        "sections_count": sections_found,
        "has_code_examples": has_code_examples,
    }


def score_documentation(
    has_docs_folder: bool = False,
    has_examples_folder: bool = False,
    has_wiki: bool = False,
) -> int:
    """Score documentation presence on a 0-100 scale."""
    score = 0
    if has_docs_folder:
        score += 50
    if has_examples_folder:
        score += 30
    if has_wiki:
        score += 20
    return score


def score_repo_engagement(
    star_count: int = 0,
    recent_commits: int = 0,
    days_since_last_commit: Optional[int] = None,
) -> int:
    """Score repository engagement on a 0-100 scale."""
    score = 0

    # Star scoring (0-50 points)
    if star_count >= 10000:
        score += 50
    elif star_count >= 5000:
        score += 45
    elif star_count >= 1000:
        score += 40
    elif star_count >= 500:
        score += 35
    elif star_count >= 100:
        score += 30
    elif star_count >= 50:
        score += 25
    elif star_count >= 20:
        score += 20
    elif star_count >= 10:
        score += 15
    elif star_count >= 5:
        score += 10
    elif star_count >= 1:
        score += 5

    # Activity scoring (0-50 points)
    if recent_commits >= 50:
        score += 30
    elif recent_commits >= 20:
        score += 25
    elif recent_commits >= 10:
        score += 20
    elif recent_commits >= 5:
        score += 15
    elif recent_commits >= 1:
        score += 10

    if days_since_last_commit is not None:
        if days_since_last_commit <= 7:
            score += 20
        elif days_since_last_commit <= 30:
            score += 15
        elif days_since_last_commit <= 90:
            score += 10
        elif days_since_last_commit <= 180:
            score += 5

    return score


def calculate_repo_negative_penalty(
    is_fork: bool = False,
    is_empty_readme: bool = False,
    repo_name: str = "",
    description: str = "",
) -> tuple[int, list]:
    """Calculate penalty for negative signals."""
    penalty = 0
    joke_indicators_found = []

    if is_fork:
        penalty -= 30
    if is_empty_readme:
        penalty -= 40

    combined = f"{repo_name.lower()} {(description or '').lower()}"
    for indicator in JOKE_INDICATORS:
        if indicator in combined:
            joke_indicators_found.append(indicator)

    if joke_indicators_found:
        penalty -= min(30, len(joke_indicators_found) * 10)

    return penalty, joke_indicators_found


def score_github_repo(
    readme_content: str = "",
    has_docs_folder: bool = False,
    has_examples_folder: bool = False,
    has_wiki: bool = False,
    star_count: int = 0,
    recent_commits: int = 0,
    days_since_last_commit: Optional[int] = None,
    is_fork: bool = False,
    repo_name: str = "",
    description: str = "",
) -> RepoScoreBreakdown:
    """Calculate comprehensive quality score for a GitHub repository."""
    breakdown = RepoScoreBreakdown()

    readme_score, readme_details = score_readme(readme_content)
    breakdown.readme_score = readme_score
    breakdown.readme_length = readme_details["length"]
    breakdown.readme_has_badges = readme_details["has_badges"]
    breakdown.readme_sections_count = readme_details["sections_count"]
    breakdown.readme_has_code_examples = readme_details["has_code_examples"]

    breakdown.docs_score = score_documentation(has_docs_folder, has_examples_folder, has_wiki)
    breakdown.has_docs_folder = has_docs_folder
    breakdown.has_examples_folder = has_examples_folder
    breakdown.has_wiki = has_wiki

    breakdown.engagement_score = score_repo_engagement(star_count, recent_commits, days_since_last_commit)
    breakdown.star_count = star_count
    breakdown.recent_commits = recent_commits
    breakdown.days_since_last_commit = days_since_last_commit

    is_empty = len((readme_content or "").strip()) < 50
    penalty, joke_indicators = calculate_repo_negative_penalty(is_fork, is_empty, repo_name, description)
    breakdown.negative_penalty = penalty
    breakdown.is_fork = is_fork
    breakdown.is_empty_readme = is_empty
    breakdown.is_joke_repo = bool(joke_indicators)
    breakdown.joke_indicators_found = joke_indicators

    return breakdown


def score_repo_from_path(
    repo_path: Path,
    star_count: int = 0,
    recent_commits: int = 0,
    days_since_last_commit: Optional[int] = None,
    is_fork: bool = False,
    has_wiki: bool = False,
    description: str = "",
) -> RepoScoreBreakdown:
    """Score a repository from a local path."""
    repo_path = Path(repo_path)
    repo_name = repo_path.name

    readme_content = ""
    for readme_name in ["README.md", "README.rst", "README.txt", "README"]:
        readme_path = repo_path / readme_name
        if readme_path.exists():
            try:
                readme_content = readme_path.read_text(encoding="utf-8", errors="ignore")
                break
            except Exception:
                pass

    has_docs_folder = (repo_path / "docs").is_dir() or (repo_path / "doc").is_dir()
    has_examples_folder = (repo_path / "examples").is_dir() or (repo_path / "example").is_dir()

    return score_github_repo(
        readme_content=readme_content,
        has_docs_folder=has_docs_folder,
        has_examples_folder=has_examples_folder,
        has_wiki=has_wiki,
        star_count=star_count,
        recent_commits=recent_commits,
        days_since_last_commit=days_since_last_commit,
        is_fork=is_fork,
        repo_name=repo_name,
        description=description,
    )


# Aliases for backwards compatibility
ScoreBreakdown = RepoScoreBreakdown
score_engagement = score_repo_engagement
calculate_negative_penalty = calculate_repo_negative_penalty


def filter_repos_by_score(
    repos: list[dict],
    min_score: int = 30,
    exclude_forks: bool = True,
    exclude_joke_repos: bool = True,
) -> list[dict]:
    """Filter a list of repositories by quality score."""
    filtered = []
    for repo in repos:
        breakdown = repo.get("score_breakdown")
        if not breakdown:
            continue

        if isinstance(breakdown, dict):
            total_score = breakdown.get("total_score", 0)
            is_fork = breakdown.get("negative_signals", {}).get("is_fork", False)
            is_joke = breakdown.get("negative_signals", {}).get("is_joke_repo", False)
        else:
            total_score = breakdown.total_score
            is_fork = breakdown.is_fork
            is_joke = breakdown.is_joke_repo

        if total_score < min_score:
            continue
        if exclude_forks and is_fork:
            continue
        if exclude_joke_repos and is_joke:
            continue

        filtered.append(repo)

    filtered.sort(
        key=lambda r: (
            r["score_breakdown"]["total_score"]
            if isinstance(r["score_breakdown"], dict)
            else r["score_breakdown"].total_score
        ),
        reverse=True,
    )
    return filtered
