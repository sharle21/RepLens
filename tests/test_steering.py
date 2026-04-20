
"""Tests for steering.py — verifies hook math without a real model."""

import pytest
import torch

from src.steering import SteeringConfig, SteeringHook


HIDDEN_DIM = 64
SEQ_LEN = 10
BATCH = 2


class FakeLayer(torch.nn.Module):
    """Minimal module that passes input through and supports hooks."""
    def forward(self, x):
        return (x,)


class TestSteeringHookAdd:
    def test_add_all_tokens(self):
        """Adding a vector to all tokens should shift every position."""
        layer = FakeLayer()
        vec = torch.ones(HIDDEN_DIM) * 0.5
        cfg = SteeringConfig(vector=vec, layer=0, strength=2.0, method="add", token_positions="all")

        hidden = torch.zeros(BATCH, SEQ_LEN, HIDDEN_DIM)

        with SteeringHook(lambda idx: layer, [cfg]):
            output = layer(hidden)

        modified = output[0]
        expected = 2.0 * 0.5  # strength * vec value
        assert torch.allclose(modified, torch.full_like(modified, expected), atol=1e-6)

    def test_add_last_token_only(self):
        """Adding to 'last' should only modify the final token position."""
        layer = FakeLayer()
        vec = torch.ones(HIDDEN_DIM)
        cfg = SteeringConfig(vector=vec, layer=0, strength=1.0, method="add", token_positions="last")

        hidden = torch.zeros(BATCH, SEQ_LEN, HIDDEN_DIM)

        with SteeringHook(lambda idx: layer, [cfg]):
            output = layer(hidden)

        modified = output[0]
        # All tokens except last should be zero
        assert torch.allclose(modified[:, :-1, :], torch.zeros(BATCH, SEQ_LEN - 1, HIDDEN_DIM))
        # Last token should be 1.0
        assert torch.allclose(modified[:, -1, :], torch.ones(BATCH, HIDDEN_DIM))

    def test_zero_strength_is_noop(self):
        """Strength 0 should not change activations."""
        layer = FakeLayer()
        vec = torch.randn(HIDDEN_DIM)
        cfg = SteeringConfig(vector=vec, layer=0, strength=0.0, method="add")

        hidden = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        original = hidden.clone()

        with SteeringHook(lambda idx: layer, [cfg]):
            output = layer(hidden)

        torch.testing.assert_close(output[0], original)


class TestSteeringHookAblate:
    def test_ablate_removes_component(self):
        """Ablating a direction should remove that component from activations."""
        layer = FakeLayer()

        # Create a vector aligned with dimension 0
        vec = torch.zeros(HIDDEN_DIM)
        vec[0] = 1.0
        cfg = SteeringConfig(vector=vec, layer=0, strength=1.0, method="ablate", token_positions="all")

        # Hidden state with known component along dim 0
        hidden = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)

        with SteeringHook(lambda idx: layer, [cfg]):
            output = layer(hidden)

        modified = output[0]
        # The component along dim 0 should be ~0
        assert torch.allclose(modified[:, :, 0], torch.zeros(BATCH, SEQ_LEN), atol=1e-5)
        # Other dimensions should be unchanged
        torch.testing.assert_close(modified[:, :, 1:], hidden[:, :, 1:])

    def test_ablate_last_token(self):
        """Ablating at 'last' should only affect the final position."""
        layer = FakeLayer()
        vec = torch.zeros(HIDDEN_DIM)
        vec[0] = 1.0
        cfg = SteeringConfig(vector=vec, layer=0, method="ablate", token_positions="last")

        hidden = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        original = hidden.clone()

        with SteeringHook(lambda idx: layer, [cfg]):
            output = layer(hidden)

        modified = output[0]
        # Non-last tokens unchanged
        torch.testing.assert_close(modified[:, :-1, :], original[:, :-1, :])
        # Last token dim 0 should be ~0
        assert torch.allclose(modified[:, -1, 0], torch.zeros(BATCH), atol=1e-5)


class TestSteeringHookCleanup:
    def test_hooks_removed_after_context(self):
        """Hooks should be cleaned up after exiting context manager."""
        layer = FakeLayer()
        vec = torch.randn(HIDDEN_DIM)
        cfg = SteeringConfig(vector=vec, layer=0, strength=1.0, method="add")

        hook_obj = SteeringHook(lambda idx: layer, [cfg])

        with hook_obj:
            assert len(hook_obj.hooks) == 1

        assert len(hook_obj.hooks) == 0

    def test_multi_layer_configs(self):
        """Multiple configs on different layers should register multiple hooks."""
        layers = {0: FakeLayer(), 5: FakeLayer()}
        vec = torch.randn(HIDDEN_DIM)
        configs = [
            SteeringConfig(vector=vec, layer=0, strength=1.0, method="add"),
            SteeringConfig(vector=vec, layer=5, strength=-1.0, method="add"),
        ]

        with SteeringHook(lambda idx: layers[idx], configs) as hook:
            assert len(hook.hooks) == 2
