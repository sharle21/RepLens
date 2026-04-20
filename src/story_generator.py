from __future__ import annotations

"""
story_generator.py — Generate emotion stories for vector extraction.

Matches Anthropic's methodology: prompt the model to write short stories about
characters experiencing specific emotions across diverse topics. These stories
are then fed back through the model for activation extraction.

Why stories instead of direct prompts:
- Stories spread emotional content across many tokens (better for mean-pooling)
- The model's own generation style produces text it "naturally" associates with
  the emotion, giving cleaner extraction signal
- Anthropic validated this approach on Claude; we replicate it on open-source models
"""

import json
import os
import random
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

import torch
from tqdm import tqdm

from src.model_adapter import ModelAdapter, ModelConfig


# Diverse topics to pair with each emotion. Variety prevents the model from
# associating the emotion vector with any single domain.
STORY_TOPICS = [
    "a job interview",
    "a hospital waiting room",
    "a chess tournament",
    "moving to a new city",
    "a first day at school",
    "a long road trip alone",
    "a family dinner",
    "preparing for a wedding",
    "a court hearing",
    "losing a wallet in a foreign country",
    "a midnight phone call",
    "a startup pitch to investors",
    "waiting for exam results",
    "a hiking trip gone wrong",
    "a reunion with an old friend",
    "cooking for a large group",
    "a power outage at home",
    "a child's birthday party",
    "receiving unexpected news",
    "the last day at a beloved job",
    "a crowded subway commute",
    "an important video call",
    "visiting a childhood home",
    "a delayed flight at the airport",
    "a performance review at work",
    "shopping for groceries on a tight budget",
    "a visit to the dentist",
    "moving a parent into assisted living",
    "a surprise inspection at work",
    "finding a lost pet",
    "a neighborhood dispute",
    "the morning of a marathon",
    "a blind date",
    "a student defending their thesis",
    "a soldier returning home",
    "an artist finishing a painting",
    "a farmer watching the weather forecast",
    "a musician before a concert",
    "a nurse finishing a double shift",
    "a scientist waiting for lab results",
    "a parent at a school play",
    "a climber near the summit",
    "a writer facing a deadline",
    "a coach before the championship game",
    "a refugee arriving in a new country",
    "a teacher on the first day of school",
    "a firefighter after a call",
    "a pilot during turbulence",
    "a developer deploying to production",
    "a photographer at golden hour",
]


# The 6 emotions from our experiment config
EMOTIONS = ["desperation", "calm", "anger", "fear", "guilt", "confidence"]


@dataclass
class StoryGenerationConfig:
    """Configuration for story generation."""
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    num_stories_per_emotion: int = 100
    max_new_tokens: int = 150
    temperature: float = 0.9
    output_dir: str = "data/stories"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    seed: int = 42


OPENING_CONSTRAINTS = [
    "Begin with a line of dialogue.",
    "Begin with a sensory detail (sound, smell, texture).",
    "Begin with a thought the character is having.",
    "Begin with a physical action the character takes.",
    "Begin with a description of the environment.",
    "Begin with a question the character asks themselves.",
    "Begin with a memory that flashes through the character's mind.",
    "Begin with a metaphor or comparison.",
    "Begin with the character noticing a small detail.",
    "Begin with a statement about time.",
]


def _build_story_prompt(emotion: str, topic: str, index: int = 0) -> str:
    """Build the prompt that asks the model to write an emotion story.

    Args:
        emotion: The target emotion (e.g. "desperation").
        topic: The scenario topic (e.g. "a job interview").
        index: Story index, used to rotate opening constraints.

    Returns:
        A user-turn prompt string (not yet wrapped in chat template).
    """
    constraint = OPENING_CONSTRAINTS[index % len(OPENING_CONSTRAINTS)]
    return (
        f"Write a short paragraph (3-5 sentences) about a person experiencing "
        f"{emotion} during {topic}. Focus on their internal feelings and thoughts. "
        f"Do not label the emotion explicitly — show it through their experience. "
        f"{constraint}"
    )


def generate_stories(
    config: StoryGenerationConfig | None = None,
) -> dict[str, list[dict]]:
    """Generate emotion stories by prompting the model.

    For each emotion, generates `num_stories_per_emotion` stories using diverse
    topics. Stories are saved to disk as individual text files plus a metadata
    JSON for reproducibility.

    Args:
        config: Generation parameters. Uses defaults if None.

    Returns:
        Dict mapping emotion -> list of story dicts with keys:
            "text", "topic", "emotion", "prompt", "tokens_generated"
    """
    if config is None:
        config = StoryGenerationConfig()

    torch.manual_seed(config.seed)
    adapter = ModelAdapter(ModelConfig(
        name=config.model_name,
        device=config.device,
        dtype=config.dtype,
    ))

    # Select topics — cycle if we need more stories than topics
    topics = STORY_TOPICS
    num_stories = config.num_stories_per_emotion

    all_stories: dict[str, list[dict]] = {}

    for emotion in EMOTIONS:
        print(f"\n{'='*60}")
        print(f"Generating {num_stories} stories for: {emotion}")
        print(f"{'='*60}")

        emotion_dir = Path(config.output_dir) / emotion
        emotion_dir.mkdir(parents=True, exist_ok=True)

        stories = []
        for i in tqdm(range(num_stories), desc=f"  {emotion}"):
            topic = topics[i % len(topics)]
            prompt = _build_story_prompt(emotion, topic, index=i)

            text = adapter.generate(
                prompt,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
            )

            story = {
                "text": text.strip(),
                "topic": topic,
                "emotion": emotion,
                "prompt": prompt,
                "index": i,
                "tokens_generated": len(adapter.tokenizer.encode(text)),
            }
            stories.append(story)

            # Save individual story
            story_path = emotion_dir / f"story_{i:03d}.txt"
            story_path.write_text(text.strip())

        all_stories[emotion] = stories

        # Save metadata for this emotion
        metadata = {
            "emotion": emotion,
            "model": config.model_name,
            "num_stories": len(stories),
            "temperature": config.temperature,
            "max_new_tokens": config.max_new_tokens,
            "seed": config.seed,
            "generated_at": datetime.now().isoformat(),
            "stories": stories,
        }
        meta_path = emotion_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"\nAll stories saved to {config.output_dir}/")
    return all_stories


def load_stories(
    output_dir: str = "data/stories",
    emotions: list[str] | None = None,
) -> dict[str, list[str]]:
    """Load previously generated stories from disk.

    Args:
        output_dir: Root directory containing per-emotion subdirectories.
        emotions: Which emotions to load. Defaults to all found on disk.

    Returns:
        Dict mapping emotion -> list of story texts (ready for extraction).
    """
    root = Path(output_dir)
    if not root.exists():
        raise FileNotFoundError(f"Story directory not found: {output_dir}")

    if emotions is None:
        emotions = [d.name for d in root.iterdir() if d.is_dir()]

    stories: dict[str, list[str]] = {}
    for emotion in sorted(emotions):
        emotion_dir = root / emotion
        if not emotion_dir.exists():
            print(f"  Warning: no stories found for {emotion}")
            continue

        texts = []
        for story_file in sorted(emotion_dir.glob("story_*.txt")):
            text = story_file.read_text().strip()
            if text:
                texts.append(text)

        stories[emotion] = texts
        print(f"  Loaded {len(texts)} stories for {emotion}")

    return stories


def inspect_stories(
    output_dir: str = "data/stories",
    num_samples: int = 3,
) -> None:
    """Print a few sample stories per emotion for manual quality checking.

    Anthropic manually inspected stories. This helper makes it easy to do the same.

    Args:
        output_dir: Root directory containing stories.
        num_samples: How many stories to print per emotion.
    """
    stories = load_stories(output_dir)
    for emotion, texts in stories.items():
        print(f"\n{'='*60}")
        print(f"  {emotion.upper()} ({len(texts)} total)")
        print(f"{'='*60}")
        for i, text in enumerate(texts[:num_samples]):
            print(f"\n  [{i+1}] {text[:300]}")
            if len(text) > 300:
                print(f"      ... ({len(text)} chars total)")
        print()
