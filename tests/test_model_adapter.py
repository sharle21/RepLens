"""Tests for model_adapter.py — no GPU needed, uses mocks."""

import pytest
from unittest.mock import MagicMock, patch
from src.model_adapter import detect_family, ModelConfig, CHAT_TEMPLATES


class TestDetectFamily:
    def test_llama(self):
        assert detect_family("meta-llama/Llama-3.1-8B-Instruct") == "llama"

    def test_qwen2(self):
        assert detect_family("Qwen/Qwen2.5-7B-Instruct") == "qwen2"

    def test_gemma(self):
        assert detect_family("google/gemma-7b-it") == "gemma"

    def test_gemma2(self):
        assert detect_family("google/gemma2-9b-it") == "gemma2"

    def test_mistral(self):
        assert detect_family("mistralai/Mistral-7B-Instruct-v0.2") == "mistral"

    def test_phi(self):
        assert detect_family("microsoft/phi-3-mini") == "phi"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Could not detect"):
            detect_family("some-unknown-model/v1")


class TestChatTemplates:
    def test_llama_template_wraps_prompt(self):
        t = CHAT_TEMPLATES["llama"]
        result = t["prefix"] + "hello" + t["suffix"]
        assert "user" in result
        assert "hello" in result
        assert "assistant" in result

    def test_all_families_have_prefix_suffix(self):
        for family, template in CHAT_TEMPLATES.items():
            assert "prefix" in template, f"{family} missing prefix"
            assert "suffix" in template, f"{family} missing suffix"


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig(name="test-model")
        assert cfg.family == "auto"
        assert cfg.max_length == 256
        assert cfg.batch_size == 4
