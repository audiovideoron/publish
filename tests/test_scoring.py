"""Tests for YouTube video quality scoring (scoring.py)."""

import pytest

from publishing import scoring


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


# =============================================================================
# GitHub Repository Scoring Tests
# =============================================================================


class TestScoreReadme:
    """Tests for README quality scoring."""

    def test_empty_readme_scores_zero(self):
        """Empty README should score 0."""
        score, details = scoring.score_readme("")
        assert score == 0
        assert details["length"] == 0
        assert details["has_badges"] is False
        assert details["sections_count"] == 0
        assert details["has_code_examples"] is False

    def test_whitespace_readme_scores_zero(self):
        """Whitespace-only README should score 0."""
        score, details = scoring.score_readme("   \n\t  ")
        assert score == 0

    def test_short_readme_low_score(self):
        """Short README gets minimal points."""
        score, details = scoring.score_readme("# Hello\n\nThis is a test.")
        # Very short content (< 50 chars) scores 0 for length
        assert score == 0
        assert details["length"] > 0
        assert details["length"] < 50

    def test_long_readme_high_length_score(self):
        """Long README gets high length score."""
        long_content = "# Project\n\n" + "Lorem ipsum dolor sit amet. " * 500
        score, details = scoring.score_readme(long_content)
        assert score >= 25  # Should get length points
        assert details["length"] >= 5000

    def test_readme_with_badges(self):
        """README with badges gets badge points."""
        content = """# Project

[![Build Status](https://travis-ci.org/user/repo.svg?branch=master)](https://travis-ci.org/user/repo)
![Coverage](https://img.shields.io/codecov/c/github/user/repo.svg)

Some content here.
"""
        score, details = scoring.score_readme(content)
        assert details["has_badges"] is True

    def test_readme_with_shields_io_badge(self):
        """README with shields.io badge is detected."""
        content = "# Project\n\n![](https://shields.io/badge/test-passing-green)"
        score, details = scoring.score_readme(content)
        assert details["has_badges"] is True

    def test_readme_with_sections(self):
        """README with structured sections scores higher."""
        content = """# Project

## Installation
pip install project

## Usage
Just use it!

## Features
- Feature 1
- Feature 2

## Contributing
See CONTRIBUTING.md

## License
MIT
"""
        score, details = scoring.score_readme(content)
        assert details["sections_count"] >= 4
        assert score >= 30  # Sections + some length

    def test_readme_with_code_blocks(self):
        """README with code examples scores higher."""
        content = """# Project

## Installation

```bash
pip install project
```

## Usage

```python
from project import foo
foo.bar()
```

```python
# Another example
result = foo.baz(123)
```
"""
        score, details = scoring.score_readme(content)
        assert details["has_code_examples"] is True

    def test_comprehensive_readme_high_score(self):
        """Comprehensive README with all elements scores highly."""
        content = """# Awesome Project

[![Build](https://img.shields.io/github/actions/workflow/status/user/repo/ci.yml)](https://github.com)
[![Coverage](https://codecov.io/gh/user/repo/branch/main/graph/badge.svg)](https://codecov.io)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A fantastic project that does amazing things.

## Installation

```bash
pip install awesome-project
```

## Usage

```python
from awesome import Project

project = Project()
project.do_amazing_things()
```

## Features

- Fast and efficient
- Easy to use
- Well documented

## Configuration

```yaml
awesome:
  setting: value
```

## API Reference

See the [API docs](https://docs.example.com).

## Examples

Check out the examples/ folder.

## Contributing

PRs welcome! See CONTRIBUTING.md.

## License

MIT License
"""
        score, details = scoring.score_readme(content)
        assert details["has_badges"] is True
        assert details["sections_count"] >= 6
        assert details["has_code_examples"] is True
        assert score >= 70  # Should be high with all elements


class TestScoreDocumentation:
    """Tests for documentation presence scoring."""

    def test_no_docs_scores_zero(self):
        """No documentation folders should score 0."""
        score = scoring.score_documentation(
            has_docs_folder=False,
            has_examples_folder=False,
            has_wiki=False,
        )
        assert score == 0

    def test_docs_folder_adds_50_points(self):
        """docs/ folder adds 50 points."""
        score = scoring.score_documentation(has_docs_folder=True)
        assert score == 50

    def test_examples_folder_adds_30_points(self):
        """examples/ folder adds 30 points."""
        score = scoring.score_documentation(has_examples_folder=True)
        assert score == 30

    def test_wiki_adds_20_points(self):
        """Wiki adds 20 points."""
        score = scoring.score_documentation(has_wiki=True)
        assert score == 20

    def test_all_docs_max_score(self):
        """All documentation presence gives 100 points."""
        score = scoring.score_documentation(
            has_docs_folder=True,
            has_examples_folder=True,
            has_wiki=True,
        )
        assert score == 100


class TestScoreRepoEngagement:
    """Tests for engagement scoring (stars, commits)."""

    def test_no_engagement_scores_zero(self):
        """No stars or commits should score 0."""
        score = scoring.score_engagement(
            star_count=0,
            recent_commits=0,
            days_since_last_commit=None,
        )
        assert score == 0

    def test_low_stars_low_score(self):
        """Few stars get low score."""
        score = scoring.score_engagement(star_count=5)
        assert 0 < score < 20

    def test_medium_stars_medium_score(self):
        """Medium stars get medium score."""
        score = scoring.score_engagement(star_count=100)
        assert 25 <= score <= 35

    def test_high_stars_high_score(self):
        """Many stars get high score."""
        score = scoring.score_engagement(star_count=5000)
        assert score >= 45

    def test_very_high_stars_max_star_score(self):
        """10k+ stars get max star score."""
        score = scoring.score_engagement(star_count=10000)
        assert score == 50  # Max star score

    def test_recent_commits_add_points(self):
        """Recent commits add to score."""
        score_none = scoring.score_engagement(star_count=10)
        score_some = scoring.score_engagement(star_count=10, recent_commits=10)
        assert score_some > score_none

    def test_very_recent_activity_bonus(self):
        """Very recent activity (< 7 days) gets maximum bonus."""
        score = scoring.score_engagement(
            star_count=10,
            recent_commits=5,
            days_since_last_commit=3,
        )
        assert score >= 45  # Stars + commits + recency bonus

    def test_stale_repo_lower_score(self):
        """Stale repo (>180 days) gets no recency bonus."""
        score_fresh = scoring.score_engagement(
            star_count=10,
            recent_commits=5,
            days_since_last_commit=7,
        )
        score_stale = scoring.score_engagement(
            star_count=10,
            recent_commits=5,
            days_since_last_commit=365,
        )
        assert score_fresh > score_stale


class TestNegativeRepoPenalty:
    """Tests for negative signal detection and penalties."""

    def test_no_negatives_no_penalty(self):
        """Clean repo gets no penalty."""
        penalty, indicators = scoring.calculate_negative_penalty(
            is_fork=False,
            is_empty_readme=False,
            repo_name="awesome-project",
            description="A great library",
        )
        assert penalty == 0
        assert indicators == []

    def test_fork_penalty(self):
        """Fork gets -30 penalty."""
        penalty, _ = scoring.calculate_negative_penalty(is_fork=True)
        assert penalty == -30

    def test_empty_readme_penalty(self):
        """Empty README gets -40 penalty."""
        penalty, _ = scoring.calculate_negative_penalty(is_empty_readme=True)
        assert penalty == -40

    def test_joke_repo_by_name(self):
        """Joke repo detected by name gets penalty."""
        penalty, indicators = scoring.calculate_negative_penalty(
            repo_name="my-hello-world",
        )
        assert penalty < 0
        assert "hello-world" in indicators

    def test_joke_repo_by_description(self):
        """Joke repo detected by description gets penalty."""
        penalty, indicators = scoring.calculate_negative_penalty(
            repo_name="project",
            description="This is my homework assignment for CS101",
        )
        assert penalty < 0
        assert any(ind in ["homework", "assignment"] for ind in indicators)

    def test_dotfiles_detected(self):
        """Dotfiles repo detected as low-value."""
        penalty, indicators = scoring.calculate_negative_penalty(
            repo_name="my-dotfiles",
        )
        assert penalty < 0
        assert "dotfiles" in indicators

    def test_multiple_negatives_stack(self):
        """Multiple negative signals stack up."""
        penalty, _ = scoring.calculate_negative_penalty(
            is_fork=True,
            is_empty_readme=True,
        )
        assert penalty <= -70  # -30 + -40

    def test_multiple_joke_indicators_capped(self):
        """Joke penalty is capped at -30."""
        penalty, indicators = scoring.calculate_negative_penalty(
            repo_name="my-hello-world-test-repo-playground-troll",
        )
        # Multiple indicators found but penalty capped
        assert penalty >= -30  # Just joke penalty
        assert len(indicators) >= 3


class TestScoreGithubRepo:
    """Tests for comprehensive repo scoring."""

    def test_empty_repo_low_score(self):
        """Empty repo with nothing should score very low."""
        breakdown = scoring.score_github_repo()
        assert breakdown.total_score <= 10

    def test_basic_repo_moderate_score(self):
        """Basic repo with decent README scores moderately."""
        readme = """# Project

## Installation

pip install project

## Usage

Use it like this. Here is more content to ensure we have enough length for scoring.
This is a great project that does amazing things for your workflow.
"""
        breakdown = scoring.score_github_repo(
            readme_content=readme,
            star_count=50,  # More stars for moderate score
            recent_commits=5,
        )
        assert 15 <= breakdown.total_score <= 50

    def test_high_quality_repo_high_score(self):
        """High quality repo with good metrics scores highly."""
        readme = """# Awesome Project

[![Build](https://img.shields.io/badge/build-passing-green)](link)

Great project.

## Installation

```bash
pip install awesome
```

## Usage

```python
import awesome
```

## Features
- Good
- Great

## API
Documented.

## Contributing
Welcome!
"""
        breakdown = scoring.score_github_repo(
            readme_content=readme,
            has_docs_folder=True,
            has_examples_folder=True,
            star_count=1000,
            recent_commits=20,
            days_since_last_commit=5,
        )
        assert breakdown.total_score >= 70

    def test_fork_penalty_reduces_score(self):
        """Fork penalty reduces total score."""
        # Use a longer README to avoid empty README penalty
        readme = "# Test Project\n\nThis is a test project with enough content to not be empty."
        score_original = scoring.score_github_repo(
            readme_content=readme,
            star_count=100,
        )
        score_fork = scoring.score_github_repo(
            readme_content=readme,
            star_count=100,
            is_fork=True,
        )
        assert score_fork.total_score < score_original.total_score
        assert score_fork.is_fork is True
        assert score_fork.negative_penalty < score_original.negative_penalty

    def test_joke_repo_penalty(self):
        """Joke repo gets penalty in total score."""
        breakdown = scoring.score_github_repo(
            readme_content="# My Hello World",
            star_count=50,
            repo_name="hello-world-tutorial",
        )
        assert breakdown.is_joke_repo is True
        assert breakdown.total_score < 40

    def test_breakdown_contains_all_details(self):
        """Score breakdown contains all expected fields."""
        breakdown = scoring.score_github_repo(
            readme_content="# Test\n\n```python\ncode\n```",
            has_docs_folder=True,
            star_count=100,
            recent_commits=10,
            days_since_last_commit=7,
        )

        # Check component scores are set
        assert breakdown.readme_score > 0
        assert breakdown.docs_score > 0
        assert breakdown.engagement_score > 0

        # Check details are populated
        assert breakdown.readme_length > 0
        assert breakdown.readme_has_code_examples is True
        assert breakdown.has_docs_folder is True
        assert breakdown.star_count == 100

    def test_score_capped_at_100(self):
        """Total score is capped at 100."""
        breakdown = scoring.score_github_repo(
            readme_content="# " + "x" * 10000,  # Long readme
            has_docs_folder=True,
            has_examples_folder=True,
            has_wiki=True,
            star_count=100000,
            recent_commits=100,
            days_since_last_commit=1,
        )
        assert breakdown.total_score <= 100

    def test_score_floor_at_zero(self):
        """Total score doesn't go below 0."""
        breakdown = scoring.score_github_repo(
            readme_content="",
            is_fork=True,
            repo_name="test-repo-playground-joke",
        )
        assert breakdown.total_score >= 0


class TestScoreRepoFromPath:
    """Tests for scoring from local repo path."""

    def test_score_from_path_reads_readme(self, tmp_path):
        """Scoring from path reads README file."""
        # Create repo structure
        (tmp_path / "README.md").write_text("# Test Project\n\n## Installation\nInstall it.")

        breakdown = scoring.score_repo_from_path(tmp_path)
        assert breakdown.readme_length > 0
        assert breakdown.readme_sections_count >= 1

    def test_score_from_path_detects_docs_folder(self, tmp_path):
        """Scoring from path detects docs folder."""
        (tmp_path / "README.md").write_text("# Test")
        (tmp_path / "docs").mkdir()

        breakdown = scoring.score_repo_from_path(tmp_path)
        assert breakdown.has_docs_folder is True

    def test_score_from_path_detects_examples_folder(self, tmp_path):
        """Scoring from path detects examples folder."""
        (tmp_path / "README.md").write_text("# Test")
        (tmp_path / "examples").mkdir()

        breakdown = scoring.score_repo_from_path(tmp_path)
        assert breakdown.has_examples_folder is True

    def test_score_from_path_uses_repo_name(self, tmp_path):
        """Scoring from path uses directory name for joke detection."""
        hello_world = tmp_path / "hello-world"
        hello_world.mkdir()
        (hello_world / "README.md").write_text("# Hello World")

        breakdown = scoring.score_repo_from_path(hello_world)
        assert breakdown.is_joke_repo is True

    def test_score_from_path_no_readme(self, tmp_path):
        """Scoring from path handles missing README."""
        # No README file
        breakdown = scoring.score_repo_from_path(tmp_path)
        assert breakdown.readme_length == 0
        assert breakdown.is_empty_readme is True

    def test_score_from_path_alternate_readme_names(self, tmp_path):
        """Scoring from path checks alternate README names."""
        (tmp_path / "README.rst").write_text("Test Project\n============\n\nContent here.")

        breakdown = scoring.score_repo_from_path(tmp_path)
        assert breakdown.readme_length > 0


class TestRepoScoreBreakdownDataclass:
    """Tests for RepoScoreBreakdown dataclass."""

    def test_total_score_calculation(self):
        """Total score is calculated from components."""
        breakdown = scoring.RepoScoreBreakdown(
            readme_score=100,  # 35 weighted
            docs_score=100,  # 20 weighted
            engagement_score=100,  # 45 weighted
            negative_penalty=0,
        )
        # 100*0.35 + 100*0.20 + 100*0.45 = 100
        assert breakdown.total_score == 100

    def test_total_score_with_penalty(self):
        """Total score includes penalty."""
        breakdown = scoring.RepoScoreBreakdown(
            readme_score=80,
            docs_score=80,
            engagement_score=80,
            negative_penalty=-30,
        )
        # 80*0.35 + 80*0.20 + 80*0.45 - 30 = 80 - 30 = 50
        assert breakdown.total_score == 50

    def test_total_score_capped_at_zero(self):
        """Total score doesn't go below 0."""
        breakdown = scoring.RepoScoreBreakdown(
            readme_score=0,
            docs_score=0,
            engagement_score=0,
            negative_penalty=-100,
        )
        assert breakdown.total_score == 0

    def test_to_dict(self):
        """to_dict produces expected structure."""
        breakdown = scoring.RepoScoreBreakdown(
            readme_score=50,
            readme_length=1000,
            readme_has_badges=True,
            star_count=100,
            is_fork=True,
        )
        result = breakdown.to_dict()

        assert "total_score" in result
        assert "components" in result
        assert result["components"]["readme_score"] == 50
        assert result["readme_details"]["length"] == 1000
        assert result["readme_details"]["has_badges"] is True
        assert result["engagement_details"]["star_count"] == 100
        assert result["negative_signals"]["is_fork"] is True


class TestFilterReposByScore:
    """Tests for filtering repos by score."""

    def test_filter_by_min_score(self):
        """Repos below min score are filtered out."""
        repos = [
            {"name": "high", "score_breakdown": scoring.RepoScoreBreakdown(readme_score=100, engagement_score=100)},
            {"name": "low", "score_breakdown": scoring.RepoScoreBreakdown(readme_score=10, engagement_score=10)},
        ]

        filtered = scoring.filter_repos_by_score(repos, min_score=50)
        assert len(filtered) == 1
        assert filtered[0]["name"] == "high"

    def test_filter_excludes_forks(self):
        """Forks are excluded by default."""
        repos = [
            {"name": "original", "score_breakdown": scoring.RepoScoreBreakdown(readme_score=80, engagement_score=80)},
            {"name": "fork", "score_breakdown": scoring.RepoScoreBreakdown(readme_score=80, engagement_score=80, is_fork=True)},
        ]

        filtered = scoring.filter_repos_by_score(repos, min_score=30)
        assert len(filtered) == 1
        assert filtered[0]["name"] == "original"

    def test_filter_excludes_joke_repos(self):
        """Joke repos are excluded by default."""
        repos = [
            {"name": "serious", "score_breakdown": scoring.RepoScoreBreakdown(readme_score=80, engagement_score=80)},
            {"name": "joke", "score_breakdown": scoring.RepoScoreBreakdown(readme_score=80, engagement_score=80, is_joke_repo=True)},
        ]

        filtered = scoring.filter_repos_by_score(repos, min_score=30)
        assert len(filtered) == 1
        assert filtered[0]["name"] == "serious"

    def test_filter_include_forks_option(self):
        """Forks can be included with option."""
        repos = [
            {"name": "fork", "score_breakdown": scoring.RepoScoreBreakdown(readme_score=80, engagement_score=80, is_fork=True)},
        ]

        filtered = scoring.filter_repos_by_score(repos, min_score=30, exclude_forks=False)
        assert len(filtered) == 1

    def test_filter_sorted_by_score(self):
        """Results are sorted by score descending."""
        repos = [
            {"name": "low", "score_breakdown": scoring.RepoScoreBreakdown(readme_score=30, engagement_score=30)},
            {"name": "high", "score_breakdown": scoring.RepoScoreBreakdown(readme_score=100, engagement_score=100)},
            {"name": "mid", "score_breakdown": scoring.RepoScoreBreakdown(readme_score=60, engagement_score=60)},
        ]

        filtered = scoring.filter_repos_by_score(repos, min_score=0)
        names = [r["name"] for r in filtered]
        assert names == ["high", "mid", "low"]

    def test_filter_handles_dict_breakdown(self):
        """Filter works with dict breakdowns (serialized form)."""
        repos = [
            {
                "name": "repo",
                "score_breakdown": {
                    "total_score": 70,
                    "negative_signals": {"is_fork": False, "is_joke_repo": False},
                },
            },
        ]

        filtered = scoring.filter_repos_by_score(repos, min_score=50)
        assert len(filtered) == 1

    def test_filter_handles_missing_breakdown(self):
        """Repos without breakdown are skipped."""
        repos = [
            {"name": "no_score"},
            {"name": "has_score", "score_breakdown": scoring.RepoScoreBreakdown(readme_score=80, engagement_score=80)},
        ]

        filtered = scoring.filter_repos_by_score(repos, min_score=30)
        assert len(filtered) == 1
        assert filtered[0]["name"] == "has_score"
