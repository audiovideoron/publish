"""Tests for YouTube video quality scoring (scoring.py)."""

import pytest

from distillyzer import scoring


class TestScoreVideo:
    """Tests for the score_video function."""

    def test_educational_title_high_score(self):
        """Test that educational titles get high scores."""
        breakdown = scoring.score_video("Python Tutorial for Beginners")

        assert breakdown.total >= 30  # Base 20 + 10 for "tutorial"
        assert breakdown.educational_score > 0
        assert "tutorial" in breakdown.educational_keywords_found

    def test_multiple_educational_keywords(self):
        """Test that multiple educational keywords increase score."""
        breakdown = scoring.score_video(
            "Learn Python: Complete Guide and Tutorial for Beginners"
        )

        assert breakdown.educational_score >= 30
        assert len(breakdown.educational_keywords_found) >= 3
        assert "learn" in breakdown.educational_keywords_found
        assert "guide" in breakdown.educational_keywords_found
        assert "tutorial" in breakdown.educational_keywords_found

    def test_gameplay_video_low_score(self):
        """Test that gameplay videos get penalized."""
        breakdown = scoring.score_video("Minecraft Gameplay - Let's Play Episode 1")

        assert breakdown.total <= 30
        assert breakdown.negative_penalty > 0
        assert "gameplay" in breakdown.negative_keywords_found
        assert "let's play" in breakdown.negative_keywords_found

    def test_vlog_video_low_score(self):
        """Test that vlogs get penalized."""
        breakdown = scoring.score_video("My Day in Tokyo - Travel Vlog")

        assert breakdown.total <= 30
        assert "vlog" in breakdown.negative_keywords_found

    def test_meme_video_low_score(self):
        """Test that meme videos get penalized."""
        breakdown = scoring.score_video("Funny Programming Memes Compilation")

        assert breakdown.negative_penalty > 0
        assert "meme" in breakdown.negative_keywords_found or "memes" in breakdown.negative_keywords_found

    def test_shorts_penalty(self):
        """Test that shorts (under 60 seconds) get penalized."""
        breakdown = scoring.score_video("Quick Tips", duration=45)

        assert breakdown.is_short is True
        assert breakdown.negative_penalty >= 20

    def test_normal_duration_no_penalty(self):
        """Test that normal duration videos don't get short penalty."""
        breakdown = scoring.score_video("Python Tutorial", duration=600)

        assert breakdown.is_short is False

    def test_neutral_title_base_score(self):
        """Test that neutral titles get base score."""
        breakdown = scoring.score_video("Some Video About Coding")

        # Should get base score (20) but no educational bonus
        assert 15 <= breakdown.total <= 40
        assert breakdown.educational_score == 0
        assert breakdown.negative_penalty == 0


class TestEngagementScoring:
    """Tests for engagement-based scoring."""

    def test_high_view_count_bonus(self):
        """Test that high view counts increase score."""
        breakdown = scoring.score_video("Python Basics", view_count=1_500_000)

        assert breakdown.engagement_score >= 20

    def test_medium_view_count_bonus(self):
        """Test that medium view counts give moderate bonus."""
        breakdown = scoring.score_video("Python Basics", view_count=50_000)

        assert breakdown.engagement_score >= 10

    def test_low_view_count_minimal_bonus(self):
        """Test that low view counts give minimal bonus."""
        breakdown = scoring.score_video("Python Basics", view_count=500)

        assert breakdown.engagement_score == 0

    def test_good_like_ratio_bonus(self):
        """Test that good like ratios increase score."""
        # 5% like ratio (very good)
        breakdown = scoring.score_video(
            "Python Basics",
            view_count=100_000,
            like_count=5_000,
        )

        assert breakdown.engagement_score >= 30
        assert breakdown.like_ratio == 0.05

    def test_poor_like_ratio_no_bonus(self):
        """Test that poor like ratios don't add bonus."""
        # 0.5% like ratio (poor)
        breakdown = scoring.score_video(
            "Python Basics",
            view_count=100_000,
            like_count=500,
        )

        # Should get view bonus but no like ratio bonus
        assert breakdown.like_ratio == 0.005
        assert breakdown.engagement_score <= 20  # Only view bonus


class TestScoreBreakdown:
    """Tests for ScoreBreakdown dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        breakdown = scoring.score_video(
            "Python Tutorial",
            view_count=10_000,
            like_count=500,
        )
        result = breakdown.to_dict()

        assert "total" in result
        assert "educational_score" in result
        assert "engagement_score" in result
        assert "negative_penalty" in result
        assert "educational_keywords_found" in result
        assert "negative_keywords_found" in result
        assert "is_short" in result
        assert "view_count" in result
        assert "like_ratio" in result

    def test_score_clamped_to_100(self):
        """Test that score is clamped to maximum 100."""
        # Video with many positive signals
        breakdown = scoring.score_video(
            "Learn Python: Complete Tutorial Course Guide for Beginners - Deep Dive",
            view_count=10_000_000,
            like_count=1_000_000,
            duration=3600,
        )

        assert breakdown.total <= 100

    def test_score_clamped_to_0(self):
        """Test that score is clamped to minimum 0."""
        # Video with many negative signals
        breakdown = scoring.score_video(
            "Minecraft Let's Play Gameplay Vlog Meme Compilation Funny Fails",
            view_count=100,
            duration=30,
        )

        assert breakdown.total >= 0


class TestFilterVideosByScore:
    """Tests for filter_videos_by_score function."""

    def test_filters_low_score_videos(self):
        """Test that low-score videos are filtered out."""
        videos = [
            {"title": "Python Tutorial for Beginners", "view_count": 100_000},
            {"title": "Gaming Meme Compilation", "view_count": 100_000},
            {"title": "How to Learn JavaScript", "view_count": 50_000},
        ]

        result = scoring.filter_videos_by_score(videos, min_score=30)

        # Only educational videos should remain
        assert len(result) == 2
        titles = [v["title"] for v in result]
        assert "Gaming Meme Compilation" not in titles

    def test_sorts_by_score_descending(self):
        """Test that results are sorted by score descending."""
        videos = [
            {"title": "Basic Video", "view_count": 1_000},
            {"title": "Python Tutorial Guide Course", "view_count": 1_000_000},
            {"title": "JavaScript Tutorial", "view_count": 100_000},
        ]

        result = scoring.filter_videos_by_score(videos, min_score=0)

        # Higher scored videos should come first
        assert result[0]["score"] >= result[-1]["score"]

    def test_adds_score_field(self):
        """Test that score field is added to results."""
        videos = [{"title": "Python Tutorial", "view_count": 10_000}]

        result = scoring.filter_videos_by_score(videos, min_score=0)

        assert "score" in result[0]
        assert "score_breakdown" in result[0]

    def test_custom_score_field_name(self):
        """Test custom score field name."""
        videos = [{"title": "Python Tutorial", "view_count": 10_000}]

        result = scoring.filter_videos_by_score(
            videos, min_score=0, score_field="quality_score"
        )

        assert "quality_score" in result[0]

    def test_empty_list(self):
        """Test with empty video list."""
        result = scoring.filter_videos_by_score([], min_score=30)

        assert result == []

    def test_preserves_original_fields(self):
        """Test that original video fields are preserved."""
        videos = [
            {
                "title": "Python Tutorial",
                "view_count": 10_000,
                "url": "https://youtube.com/watch?v=abc",
                "channel": "Test Channel",
            }
        ]

        result = scoring.filter_videos_by_score(videos, min_score=0)

        assert result[0]["url"] == "https://youtube.com/watch?v=abc"
        assert result[0]["channel"] == "Test Channel"


class TestIsEducational:
    """Tests for is_educational helper function."""

    def test_educational_video(self):
        """Test that educational videos are detected."""
        # Single keyword gets 30 points (base 20 + 10 for tutorial)
        # Use multiple keywords to reach 40 threshold
        assert scoring.is_educational("Python Tutorial: Complete Beginner's Guide") is True

    def test_non_educational_video(self):
        """Test that non-educational videos are detected."""
        assert scoring.is_educational("Gaming Meme Compilation") is False

    def test_with_description(self):
        """Test that description is considered."""
        # Title is neutral but description has educational keywords
        result = scoring.is_educational(
            title="Python Basics",
            description="In this tutorial, we'll learn the fundamentals of Python programming."
        )
        assert result is True


class TestKeywordDetection:
    """Tests for keyword detection edge cases."""

    def test_case_insensitive_matching(self):
        """Test that keyword matching is case insensitive."""
        breakdown = scoring.score_video("PYTHON TUTORIAL FOR BEGINNERS")

        assert "tutorial" in breakdown.educational_keywords_found

    def test_word_boundary_matching(self):
        """Test that keywords match on word boundaries."""
        # "tutorial" should match, not "tutorializing"
        breakdown = scoring.score_video("This is a tutorial video")

        assert "tutorial" in breakdown.educational_keywords_found

    def test_how_to_keyword(self):
        """Test 'how to' keyword detection."""
        breakdown = scoring.score_video("How to Build a REST API")

        assert "how to" in breakdown.educational_keywords_found

    def test_explained_keyword(self):
        """Test 'explained' keyword detection."""
        breakdown = scoring.score_video("Recursion Explained Simply")

        assert "explained" in breakdown.educational_keywords_found

    def test_guide_keyword(self):
        """Test 'guide' keyword detection."""
        breakdown = scoring.score_video("Complete Beginner's Guide to React")

        assert "guide" in breakdown.educational_keywords_found

    def test_learn_keyword(self):
        """Test 'learn' keyword detection."""
        breakdown = scoring.score_video("Learn TypeScript in 10 Minutes")

        assert "learn" in breakdown.educational_keywords_found


class TestNegativeKeywordDetection:
    """Tests for negative keyword detection."""

    def test_lets_play_detection(self):
        """Test 'let's play' keyword detection."""
        breakdown = scoring.score_video("Let's Play Dark Souls III")

        assert "let's play" in breakdown.negative_keywords_found

    def test_shorts_detection(self):
        """Test 'shorts' keyword detection."""
        breakdown = scoring.score_video("Quick Tip #shorts")

        assert "shorts" in breakdown.negative_keywords_found

    def test_reaction_detection(self):
        """Test 'reaction' keyword detection."""
        breakdown = scoring.score_video("My Reaction to the New iPhone")

        assert "reaction" in breakdown.negative_keywords_found

    def test_unboxing_detection(self):
        """Test 'unboxing' keyword detection."""
        breakdown = scoring.score_video("iPhone 16 Unboxing and First Impressions")

        assert "unboxing" in breakdown.negative_keywords_found
