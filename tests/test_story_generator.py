"""Tests for story_generator.py — prompt building and loading logic."""

import pytest
import tempfile
from pathlib import Path

from src.story_generator import (
    _build_story_prompt,
    STORY_TOPICS,
    EMOTIONS,
    load_stories,
    StoryGenerationConfig,
)


class TestBuildStoryPrompt:
    def test_contains_emotion_and_topic(self):
        prompt = _build_story_prompt("desperation", "a job interview")
        assert "desperation" in prompt
        assert "a job interview" in prompt

    def test_asks_for_short_paragraph(self):
        prompt = _build_story_prompt("calm", "hiking")
        assert "short paragraph" in prompt.lower() or "3-5 sentences" in prompt

    def test_asks_not_to_label_emotion(self):
        prompt = _build_story_prompt("anger", "traffic")
        assert "Do not label" in prompt


class TestStoryTopics:
    def test_at_least_50_topics(self):
        assert len(STORY_TOPICS) >= 50

    def test_topics_are_unique(self):
        assert len(STORY_TOPICS) == len(set(STORY_TOPICS))


class TestEmotions:
    def test_six_emotions(self):
        assert len(EMOTIONS) == 6

    def test_expected_emotions(self):
        expected = {"desperation", "calm", "anger", "fear", "guilt", "confidence"}
        assert set(EMOTIONS) == expected


class TestLoadStories:
    def test_load_from_disk(self, tmp_path):
        """Create fake story files and verify load_stories reads them."""
        emotion_dir = tmp_path / "anger"
        emotion_dir.mkdir()
        (emotion_dir / "story_000.txt").write_text("The man slammed his fist on the table.")
        (emotion_dir / "story_001.txt").write_text("She could barely contain her rage.")

        stories = load_stories(str(tmp_path))
        assert "anger" in stories
        assert len(stories["anger"]) == 2

    def test_skips_empty_files(self, tmp_path):
        emotion_dir = tmp_path / "calm"
        emotion_dir.mkdir()
        (emotion_dir / "story_000.txt").write_text("A quiet afternoon.")
        (emotion_dir / "story_001.txt").write_text("")  # empty

        stories = load_stories(str(tmp_path))
        assert len(stories["calm"]) == 1

    def test_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            load_stories("/nonexistent/path")

    def test_filter_by_emotion(self, tmp_path):
        for emo in ["anger", "calm", "fear"]:
            d = tmp_path / emo
            d.mkdir()
            (d / "story_000.txt").write_text(f"A {emo} story.")

        stories = load_stories(str(tmp_path), emotions=["anger", "fear"])
        assert "anger" in stories
        assert "fear" in stories
        assert "calm" not in stories


class TestStoryGenerationConfig:
    def test_defaults(self):
        cfg = StoryGenerationConfig()
        assert cfg.num_stories_per_emotion == 100
        assert cfg.temperature == 0.9
        assert cfg.seed == 42
