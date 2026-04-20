"""Tests for vector_extraction.py — uses mock model layers."""

import pytest
import torch
from unittest.mock import MagicMock

from src.vector_extraction import (
    ActivationCollector,
    ExtractionConfig,
    compute_interaction_metrics,
    find_best_layer,
)


HIDDEN_DIM = 64
SEQ_LEN = 20


def make_fake_layer():
    """Create a fake nn.Module that supports register_forward_hook."""
    layer = MagicMock()
    layer._forward_hooks = {}
    hooks = []

    def register_forward_hook(fn):
        hook = MagicMock()
        hooks.append((fn, hook))
        # Actually store so we can trigger it
        layer._hooks = hooks
        return hook

    layer.register_forward_hook = register_forward_hook
    return layer, hooks


class TestActivationCollector:
    def test_last_token_mode(self):
        """Collector in 'last' mode should extract only the final token."""
        layer = torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        get_layer = lambda idx: layer

        collector = ActivationCollector(
            get_layer, layer_indices=[0], token_mode="last"
        )
        collector.clear()

        fake_hidden = torch.randn(1, SEQ_LEN, HIDDEN_DIM)
        fake_output = (fake_hidden,)

        hook_fn = collector._make_hook(0)
        hook_fn(None, None, fake_output)

        result = collector.get_stacked(0)
        assert result.shape == (1, HIDDEN_DIM)
        # Should be the last token
        torch.testing.assert_close(result[0], fake_hidden[0, -1, :].cpu())

        collector.remove_hooks()

    def test_mean_from_n_mode(self):
        """Collector in 'mean_from_n' mode should average from token N onward."""
        layer = torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        get_layer = lambda idx: layer

        mean_from = 5
        collector = ActivationCollector(
            get_layer, layer_indices=[0],
            token_mode="mean_from_n", mean_from_token=mean_from,
        )
        collector.clear()

        fake_hidden = torch.randn(1, SEQ_LEN, HIDDEN_DIM)
        fake_output = (fake_hidden,)

        hook_fn = collector._make_hook(0)
        hook_fn(None, None, fake_output)

        result = collector.get_stacked(0)
        expected = fake_hidden[0, mean_from:, :].mean(dim=0)
        torch.testing.assert_close(result[0], expected.cpu())

        collector.remove_hooks()

    def test_mean_from_n_clamps_to_seq_len(self):
        """If mean_from_token > seq_len, should use seq_len - 1."""
        layer = torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        collector = ActivationCollector(
            lambda idx: layer, layer_indices=[0],
            token_mode="mean_from_n", mean_from_token=999,
        )
        collector.clear()

        fake_hidden = torch.randn(1, 10, HIDDEN_DIM)
        hook_fn = collector._make_hook(0)
        hook_fn(None, None, (fake_hidden,))

        result = collector.get_stacked(0)
        # start = min(999, 10-1) = 9, so just the last token
        expected = fake_hidden[0, 9:, :].mean(dim=0)
        torch.testing.assert_close(result[0], expected.cpu())

        collector.remove_hooks()

    def test_multiple_batches_stack(self):
        """Multiple forward passes should stack activations."""
        layer = torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        collector = ActivationCollector(
            lambda idx: layer, layer_indices=[0], token_mode="last"
        )
        collector.clear()

        hook_fn = collector._make_hook(0)
        # Two batches of size 2
        hook_fn(None, None, (torch.randn(2, SEQ_LEN, HIDDEN_DIM),))
        hook_fn(None, None, (torch.randn(2, SEQ_LEN, HIDDEN_DIM),))

        result = collector.get_stacked(0)
        assert result.shape == (4, HIDDEN_DIM)

        collector.remove_hooks()


class TestComputeInteractionMetrics:
    def test_identical_vectors_have_cosine_one(self):
        vec = torch.randn(HIDDEN_DIM)
        vec = vec / vec.norm()

        refusal = {0: vec}
        emotion = {"test": {0: vec}}

        metrics = compute_interaction_metrics(refusal, emotion, layers=[0])
        cos_sim = metrics["test"][0]["cosine_similarity"]
        assert abs(cos_sim - 1.0) < 1e-5

    def test_orthogonal_vectors_have_cosine_zero(self):
        v1 = torch.zeros(HIDDEN_DIM)
        v1[0] = 1.0
        v2 = torch.zeros(HIDDEN_DIM)
        v2[1] = 1.0

        refusal = {0: v1}
        emotion = {"test": {0: v2}}

        metrics = compute_interaction_metrics(refusal, emotion, layers=[0])
        cos_sim = metrics["test"][0]["cosine_similarity"]
        assert abs(cos_sim) < 1e-5

    def test_opposite_vectors_have_cosine_negative_one(self):
        vec = torch.randn(HIDDEN_DIM)
        vec = vec / vec.norm()

        refusal = {0: vec}
        emotion = {"test": {0: -vec}}

        metrics = compute_interaction_metrics(refusal, emotion, layers=[0])
        cos_sim = metrics["test"][0]["cosine_similarity"]
        assert abs(cos_sim + 1.0) < 1e-5

    def test_metrics_have_all_keys(self):
        refusal = {0: torch.randn(HIDDEN_DIM)}
        emotion = {"emo1": {0: torch.randn(HIDDEN_DIM)}}

        metrics = compute_interaction_metrics(refusal, emotion, layers=[0])
        assert "cosine_similarity" in metrics["emo1"][0]
        assert "projection_magnitude" in metrics["emo1"][0]
        assert "orthogonal_component" in metrics["emo1"][0]


class TestFindBestLayer:
    def test_finds_layer_with_highest_alignment(self):
        refusal = {}
        emotion_vecs = {"test": {}}

        # Layer 5 has perfect alignment, others are random
        for i in range(10):
            refusal[i] = torch.randn(HIDDEN_DIM)
            refusal[i] = refusal[i] / refusal[i].norm()
            if i == 5:
                emotion_vecs["test"][i] = refusal[i].clone()  # cos_sim = 1.0
            else:
                # Make orthogonal-ish
                v = torch.randn(HIDDEN_DIM)
                v = v - torch.dot(v, refusal[i]) * refusal[i]
                v = v / v.norm()
                emotion_vecs["test"][i] = v

        best = find_best_layer(refusal, emotion_vecs, "test")
        assert best == 5


class TestExtractionConfig:
    def test_defaults(self):
        cfg = ExtractionConfig()
        assert cfg.max_length == 256
        assert cfg.batch_size == 8
        assert len(cfg.target_layers) == 32

    def test_no_model_name_field(self):
        """ExtractionConfig should NOT have model_name — that's ModelAdapter's job."""
        assert not hasattr(ExtractionConfig(), "model_name")
