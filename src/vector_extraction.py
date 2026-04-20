from __future__ import annotations

"""
vector_extraction.py — Extract refusal and emotion vectors from LLM activations.

Implements the contrastive activation approach from:
- Arditi et al. (2024) for refusal vectors
- Anthropic (2026) story-based approach for emotion vectors

Two token modes are supported:
- "last": extract the last token position only (standard for refusal vectors)
- "mean_from_n": average all token positions from token N onward (Anthropic's approach
  for story-based emotion vectors, where N=50 skips boilerplate story preamble)
"""

import torch
import numpy as np
from typing import Optional, Literal, Callable
from dataclasses import dataclass, field
from tqdm import tqdm

from src.model_adapter import ModelAdapter, ModelConfig


@dataclass
class ExtractionConfig:
    """Configuration for vector extraction.

    Only controls extraction behavior — model loading is handled by ModelAdapter.
    """
    target_layers: list[int] = field(default_factory=lambda: list(range(0, 32)))
    max_length: int = 256
    batch_size: int = 8


class ActivationCollector:
    """Hook-based collector for intermediate activations from transformer layers.

    Model-agnostic: accepts a `get_layer` callable instead of reaching into
    model internals, so it works across architectures (Llama, Qwen, Gemma, etc.).

    Args:
        get_layer: Callable that takes a layer index and returns the nn.Module.
            Typically `model_adapter.get_layer`.
        layer_indices: Which transformer layers to record.
        token_mode: How to reduce the sequence dimension.
            "last" — take only the final token position (standard).
            "mean_from_n" — average all positions from `mean_from_token` onward
            (matches Anthropic's story-based extraction approach).
        mean_from_token: Start index for mean pooling when token_mode="mean_from_n".
            Anthropic used ~50 to skip story boilerplate.
    """

    def __init__(
        self,
        get_layer: Callable[[int], torch.nn.Module],
        layer_indices: list[int],
        token_mode: Literal["last", "mean_from_n"] = "last",
        mean_from_token: int = 50,
    ):
        self.get_layer = get_layer
        self.layer_indices = layer_indices
        self.token_mode = token_mode
        self.mean_from_token = mean_from_token
        self.activations: dict[int, list[torch.Tensor]] = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for idx in self.layer_indices:
            layer = self.get_layer(idx)
            hook = layer.register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)
            if layer_idx not in self.activations:
                self.activations[layer_idx] = []

            if self.token_mode == "last":
                extracted = hidden[:, -1, :].detach().cpu()
            else:
                start = min(self.mean_from_token, hidden.shape[1] - 1)
                extracted = hidden[:, start:, :].mean(dim=1).detach().cpu()

            self.activations[layer_idx].append(extracted)
        return hook_fn

    def clear(self):
        self.activations = {}

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_stacked(self, layer_idx: int) -> torch.Tensor:
        """Return all collected activations for a layer as (N, hidden_dim)."""
        return torch.cat(self.activations[layer_idx], dim=0)


class VectorExtractor:
    """Extract refusal and emotion vectors via contrastive activation differences.

    Uses ModelAdapter for model-agnostic layer access and tokenization.
    """

    def __init__(self, adapter: ModelAdapter, config: ExtractionConfig | None = None):
        self.adapter = adapter
        self.config = config or ExtractionConfig(
            target_layers=list(range(adapter.num_layers)),
        )

    def _collect_activations(
        self,
        prompts: list[str],
        desc: str = "Collecting",
        token_mode: Literal["last", "mean_from_n"] = "last",
        mean_from_token: int = 50,
    ) -> dict[int, torch.Tensor]:
        """Run prompts through the model and collect per-layer activations.

        Args:
            prompts: Input texts (already formatted with chat template if needed).
            desc: Progress bar label.
            token_mode: "last" or "mean_from_n" — see ActivationCollector.
            mean_from_token: Used when token_mode="mean_from_n".

        Returns:
            Dict mapping layer index -> stacked activations (N, hidden_dim).
        """
        collector = ActivationCollector(
            self.adapter.get_layer,
            self.config.target_layers,
            token_mode=token_mode,
            mean_from_token=mean_from_token,
        )

        batch_size = self.config.batch_size
        for i in tqdm(range(0, len(prompts), batch_size), desc=desc):
            batch = prompts[i : i + batch_size]
            inputs = self.adapter.tokenize(batch, max_length=self.config.max_length)

            with torch.no_grad():
                self.adapter.model(**inputs)

        result = {
            layer: collector.get_stacked(layer) for layer in self.config.target_layers
        }
        collector.remove_hooks()
        return result

    def extract_refusal_vector(
        self,
        harmful_prompts: list[str],
        harmless_prompts: list[str],
    ) -> dict[int, torch.Tensor]:
        """
        Extract refusal direction via difference-in-means (Arditi et al., 2024).

        The refusal vector at each layer is:
            r_l = mean(activations_harmful) - mean(activations_harmless)

        This direction, when present, causes the model to refuse.
        When ablated, the model complies with harmful requests.

        Returns:
            Dictionary mapping layer index -> refusal vector (hidden_dim,)
        """
        print("Extracting refusal vectors...")
        print(f"  Harmful prompts: {len(harmful_prompts)}")
        print(f"  Harmless prompts: {len(harmless_prompts)}")

        harmful_acts = self._collect_activations(harmful_prompts, "Harmful prompts")
        harmless_acts = self._collect_activations(harmless_prompts, "Harmless prompts")

        refusal_vectors = {}
        for layer in self.config.target_layers:
            mean_harmful = harmful_acts[layer].mean(dim=0)
            mean_harmless = harmless_acts[layer].mean(dim=0)
            refusal_vec = mean_harmful - mean_harmless
            # Normalize
            refusal_vec = refusal_vec / refusal_vec.norm()
            refusal_vectors[layer] = refusal_vec

        print("Refusal vectors extracted.")
        return refusal_vectors

    def extract_emotion_vectors(
        self,
        emotion_prompts: dict[str, list[str]],
        neutral_prompts: list[str] | None = None,
        token_mode: Literal["last", "mean_from_n"] = "last",
        mean_from_token: int = 50,
        use_cross_emotion_baseline: bool = False,
    ) -> dict[str, dict[int, torch.Tensor]]:
        """Extract emotion vectors via contrastive activation differences.

        Two baseline modes are supported:

        1. Neutral baseline (default, token_mode="last"):
           e_l = mean(emotion_activations) - mean(neutral_activations)
           Used for quick descriptive extraction from single-turn prompts.

        2. Cross-emotion baseline (Anthropic's method, use_cross_emotion_baseline=True):
           e_l = mean(this_emotion) - mean(ALL_emotions)
           Isolates what is unique to each emotion, removing shared variance.
           Should be paired with token_mode="mean_from_n" and story-based prompts.

        Args:
            emotion_prompts: Dict mapping emotion name -> list of prompts/stories.
            neutral_prompts: Baseline prompts. Required when use_cross_emotion_baseline=False.
            token_mode: "last" for single-turn prompts, "mean_from_n" for stories.
            mean_from_token: Token index to start averaging from (default 50).
            use_cross_emotion_baseline: If True, use cross-emotion baseline (Anthropic method).

        Returns:
            Dict mapping emotion -> {layer -> normalized emotion vector (hidden_dim,)}
        """
        if not use_cross_emotion_baseline and neutral_prompts is None:
            raise ValueError("neutral_prompts required when use_cross_emotion_baseline=False")

        print(f"Extracting emotion vectors (mode={'cross-emotion' if use_cross_emotion_baseline else 'neutral'}, token={token_mode})...")

        # Collect activations for every emotion
        all_acts: dict[str, dict[int, torch.Tensor]] = {}
        for emotion, prompts in emotion_prompts.items():
            print(f"\n  Processing emotion: {emotion} ({len(prompts)} prompts)")
            all_acts[emotion] = self._collect_activations(
                prompts, f"  {emotion}", token_mode=token_mode, mean_from_token=mean_from_token
            )

        # Compute baseline
        if use_cross_emotion_baseline:
            # Cross-emotion baseline: mean over all emotions at each layer
            cross_baseline: dict[int, torch.Tensor] = {}
            for layer in self.config.target_layers:
                stacked = torch.stack(
                    [all_acts[em][layer].mean(dim=0) for em in emotion_prompts], dim=0
                )
                cross_baseline[layer] = stacked.mean(dim=0)
        else:
            neutral_acts = self._collect_activations(
                neutral_prompts, "Neutral baseline", token_mode=token_mode, mean_from_token=mean_from_token  # type: ignore[arg-type]
            )

        emotion_vectors: dict[str, dict[int, torch.Tensor]] = {}
        for emotion in emotion_prompts:
            vectors: dict[int, torch.Tensor] = {}
            for layer in self.config.target_layers:
                mean_emotion = all_acts[emotion][layer].mean(dim=0)
                if use_cross_emotion_baseline:
                    baseline = cross_baseline[layer]
                else:
                    baseline = neutral_acts[layer].mean(dim=0)  # type: ignore[union-attr]
                vec = mean_emotion - baseline
                vec = vec / vec.norm()
                vectors[layer] = vec
            emotion_vectors[emotion] = vectors

        print("\nEmotion vectors extracted.")
        return emotion_vectors


def compute_interaction_metrics(
    refusal_vectors: dict[int, torch.Tensor],
    emotion_vectors: dict[str, dict[int, torch.Tensor]],
    layers: Optional[list[int]] = None,
) -> dict:
    """
    Compute geometric interaction metrics between emotion and refusal vectors.

    Metrics computed per (emotion, layer):
    - cosine_similarity: alignment between emotion and refusal directions
    - projection_magnitude: how much of the emotion vector lies in refusal direction
    - orthogonal_component: magnitude of emotion vector orthogonal to refusal

    Returns:
        Nested dict: {emotion: {layer: {metric: value}}}
    """
    if layers is None:
        layers = sorted(refusal_vectors.keys())

    results = {}
    for emotion, evecs in emotion_vectors.items():
        results[emotion] = {}
        for layer in layers:
            r = refusal_vectors[layer].float()
            e = evecs[layer].float()

            cos_sim = torch.nn.functional.cosine_similarity(
                r.unsqueeze(0), e.unsqueeze(0)
            ).item()

            # Scalar projection of emotion onto refusal direction
            projection = torch.dot(e, r).item()

            # Orthogonal component magnitude
            parallel = projection * r
            orthogonal = e - parallel
            orth_magnitude = orthogonal.norm().item()

            results[emotion][layer] = {
                "cosine_similarity": cos_sim,
                "projection_magnitude": projection,
                "orthogonal_component": orth_magnitude,
            }

    return results


def find_best_layer(
    refusal_vectors: dict[int, torch.Tensor],
    emotion_vectors: dict[str, dict[int, torch.Tensor]],
    target_emotion: str = "desperation",
) -> int:
    """
    Find the layer where the interaction between a target emotion
    and refusal is strongest (highest absolute cosine similarity).
    """
    metrics = compute_interaction_metrics(
        refusal_vectors, {target_emotion: emotion_vectors[target_emotion]}
    )

    best_layer = max(
        metrics[target_emotion].keys(),
        key=lambda l: abs(metrics[target_emotion][l]["cosine_similarity"]),
    )
    return best_layer
