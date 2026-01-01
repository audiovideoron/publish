"""YouTube video quality scoring for content recommendations.

Score videos based on:
- Educational value: keywords in title like "tutorial", "explained", "how to", "guide", "learn"
- Engagement: view count, like ratio
- Negative signals: filter out gameplay, vlogs, memes, shorts
"""

import re
from dataclasses import dataclass


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
NEGATIVE_KEYWORDS = [
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
class ScoreBreakdown:
    """Detailed breakdown of the video quality score."""
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
) -> ScoreBreakdown:
    """
    Score a YouTube video for educational quality.

    Args:
        title: Video title
        view_count: Number of views (optional)
        like_count: Number of likes (optional)
        duration: Video duration in seconds (optional)
        description: Video description (optional, used for additional keyword detection)

    Returns:
        ScoreBreakdown with total score (0-100) and component breakdown
    """
    # Combine title and description for keyword matching
    text_to_check = title.lower()
    if description:
        text_to_check += " " + description.lower()

    # Calculate educational score (0-40 points)
    educational_score, educational_found = _calculate_educational_score(text_to_check)

    # Calculate engagement score (0-40 points)
    engagement_score, like_ratio = _calculate_engagement_score(view_count, like_count)

    # Calculate negative penalty (0-50 points deduction)
    negative_penalty, negative_found = _calculate_negative_penalty(text_to_check)

    # Check if video is a "short" (under 60 seconds)
    is_short = duration is not None and duration < 60

    # Apply short penalty
    short_penalty = 20 if is_short else 0

    # Base score starts at 20 (neutral content gets some points)
    base_score = 20

    # Calculate total score
    total = base_score + educational_score + engagement_score - negative_penalty - short_penalty

    # Clamp to 0-100 range
    total = max(0, min(100, total))

    return ScoreBreakdown(
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
    """
    Calculate educational score based on keyword presence.

    Returns:
        Tuple of (score, list of found keywords)
    """
    found_keywords = []
    for keyword in EDUCATIONAL_KEYWORDS:
        # Use word boundary matching for more accurate detection
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            found_keywords.append(keyword)

    # Each keyword adds 10 points, max 40
    score = min(40, len(found_keywords) * 10)
    return score, found_keywords


def _calculate_engagement_score(
    view_count: int | None,
    like_count: int | None,
) -> tuple[int, float | None]:
    """
    Calculate engagement score based on views and likes.

    Returns:
        Tuple of (score, like_ratio or None)
    """
    score = 0
    like_ratio = None

    # View-based scoring (0-20 points)
    if view_count is not None:
        if view_count >= 1_000_000:
            score += 20
        elif view_count >= 100_000:
            score += 15
        elif view_count >= 10_000:
            score += 10
        elif view_count >= 1_000:
            score += 5

    # Like ratio scoring (0-20 points)
    if view_count and like_count and view_count > 0:
        like_ratio = like_count / view_count
        # Typical good like ratios are 3-5% of views
        if like_ratio >= 0.05:
            score += 20
        elif like_ratio >= 0.03:
            score += 15
        elif like_ratio >= 0.02:
            score += 10
        elif like_ratio >= 0.01:
            score += 5

    return score, like_ratio


def _calculate_negative_penalty(text: str) -> tuple[int, list[str]]:
    """
    Calculate penalty for non-educational content indicators.

    Returns:
        Tuple of (penalty, list of found negative keywords)
    """
    found_keywords = []
    for keyword in NEGATIVE_KEYWORDS:
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            found_keywords.append(keyword)

    # Each negative keyword adds 15 points penalty, max 50
    penalty = min(50, len(found_keywords) * 15)
    return penalty, found_keywords


def filter_videos_by_score(
    videos: list[dict],
    min_score: int = 30,
    score_field: str = "score",
) -> list[dict]:
    """
    Filter and sort videos by quality score.

    Args:
        videos: List of video dicts with title, view_count, like_count, duration fields
        min_score: Minimum score threshold (default 30)
        score_field: Field name to store score in output (default "score")

    Returns:
        List of videos with score >= min_score, sorted by score descending
    """
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

    # Sort by score descending
    scored_videos.sort(key=lambda v: v[score_field], reverse=True)

    return scored_videos


def is_educational(title: str, description: str | None = None) -> bool:
    """
    Quick check if a video appears to be educational.

    Args:
        title: Video title
        description: Video description (optional)

    Returns:
        True if video appears educational (score >= 40)
    """
    breakdown = score_video(title=title, description=description)
    return breakdown.total >= 40
