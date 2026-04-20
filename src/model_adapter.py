"""
model_adapter.py — Model-agnostic interface for representation engineering.

Abstracts away architecture differences between model families so the same
extraction and steering code works across Llama, Qwen, Gemma, etc.

This is important for:
1. Cross-model experiments (do findings generalize?)
2. Demonstrating engineering maturity (not hardcoded to one model)
"""

import torch
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for loading and interacting with a model."""
    name: str
    family: str = "auto"  # "llama", "qwen", "gemma", or "auto" to detect
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    max_length: int = 256
    batch_size: int = 4


# Architecture-specific layer access paths
LAYER_ACCESSORS = {
    "llama": lambda model: model.model.layers,
    "qwen": lambda model: model.model.layers,
    "qwen2": lambda model: model.model.layers,
    "gemma": lambda model: model.model.layers,
    "gemma2": lambda model: model.model.layers,
    "mistral": lambda model: model.model.layers,
    "phi": lambda model: model.model.layers,
}

# Chat templates per family
CHAT_TEMPLATES = {
    "llama": {
        "prefix": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
        "suffix": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "qwen": {
        "prefix": "<|im_start|>user\n",
        "suffix": "<|im_end|>\n<|im_start|>assistant\n",
    },
    "qwen2": {
        "prefix": "<|im_start|>user\n",
        "suffix": "<|im_end|>\n<|im_start|>assistant\n",
    },
    "gemma": {
        "prefix": "<start_of_turn>user\n",
        "suffix": "<end_of_turn>\n<start_of_turn>model\n",
    },
    "gemma2": {
        "prefix": "<start_of_turn>user\n",
        "suffix": "<end_of_turn>\n<start_of_turn>model\n",
    },
    "mistral": {
        "prefix": "[INST] ",
        "suffix": " [/INST]",
    },
    "phi": {
        "prefix": "<|user|>\n",
        "suffix": "<|end|>\n<|assistant|>\n",
    },
}


def detect_family(model_name: str) -> str:
    """Detect model family from the model name/path."""
    name_lower = model_name.lower()
    families = ["llama", "qwen2", "qwen", "gemma2", "gemma", "mistral", "phi"]
    for family in families:
        if family in name_lower:
            return family
    raise ValueError(
        f"Could not detect model family from '{model_name}'. "
        f"Please specify family explicitly in ModelConfig."
    )


class ModelAdapter:
    """
    Unified interface for working with different LLM architectures.

    Provides:
    - Model and tokenizer loading
    - Architecture-aware layer access
    - Chat template formatting
    - Hook registration on the correct modules
    """

    def __init__(self, config: ModelConfig):
        self.config = config

        if config.family == "auto":
            self.family = detect_family(config.name)
        else:
            self.family = config.family

        print(f"Loading {config.name} (family: {self.family})")

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.name, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.name,
            torch_dtype=config.dtype,
            device_map=config.device,
        )
        self.model.eval()

        self._layers = self._get_layers()
        self.num_layers = len(self._layers)
        self.hidden_dim = self.model.config.hidden_size

        print(f"Loaded: {self.num_layers} layers, hidden_dim={self.hidden_dim}")

    def _get_layers(self):
        """Get the transformer layers using architecture-specific accessor."""
        accessor = LAYER_ACCESSORS.get(self.family)
        if accessor is None:
            raise ValueError(f"Unsupported model family: {self.family}")
        return accessor(self.model)

    def get_layer(self, idx: int):
        """Get a specific transformer layer."""
        return self._layers[idx]

    def format_prompt(self, text: str) -> str:
        """Format a prompt using the model's chat template."""
        template = CHAT_TEMPLATES.get(self.family)
        if template is None:
            return f"User: {text}\nAssistant:"
        return template["prefix"] + text + template["suffix"]

    def format_prompts(self, texts: list[str]) -> list[str]:
        """Format a batch of prompts."""
        return [self.format_prompt(t) for t in texts]

    def tokenize(
        self, texts: list[str], max_length: Optional[int] = None
    ) -> dict:
        """Tokenize a batch of texts."""
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length or self.config.max_length,
        ).to(self.config.device)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Generate a response to a prompt."""
        formatted = self.format_prompt(prompt)
        inputs = self.tokenize([formatted])
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                **kwargs,
            )
        return self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

    def get_model_info(self) -> dict:
        """Return model metadata for experiment logging."""
        return {
            "name": self.config.name,
            "family": self.family,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "dtype": str(self.config.dtype),
            "device": self.config.device,
            "num_parameters": sum(
                p.numel() for p in self.model.parameters()
            ),
        }
